//! Walking the graph.
//!
//! Two rules shape this module. `run.json` is written *before* the transition
//! it describes, so a crash re-executes a node rather than losing it — safe
//! only because documents are immutable and attempts are numbered. And ready
//! nodes are taken one at a time: the pipeline is sequential by nature, and a
//! second 14B model does not fit in this machine's memory anyway.

use crate::backend::{ModelBackend, TokenSink};
use crate::chat::{self, CallSource, Message};
use crate::clarify::{self, ClarificationAuthority, SELF_RESOLUTION_PASSES};
use crate::dispatcher::{Dispatcher, Outcome};
use crate::document::{self, Artifact, Document};
use crate::graph::{Graph, Node, NodeId, NodeKind};
use crate::node::{format_contract, now_utc, strip_outer_fence, MAX_ATTEMPTS};
use crate::ollama::Options;
use crate::tokens::Tokenizer;
use crate::toolcalls::{self, Step, TOOL_INSTRUCTIONS};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeState {
    Idle,
    Queued,
    WaitingMemory,
    Running,
    Compacting,
    AwaitingDecision,
    Done,
    Failed,
    Skipped,
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleSpec {
    pub name: String,
    pub slot: String,
    pub window: u32,
    pub temperature: f32,
    pub artifact: String,
    pub system: String,
    /// Documents read in full. Everything else in the run arrives as a digest.
    #[serde(default)]
    pub primary_inputs: Vec<String>,
    /// When true the role works in steps with tools instead of answering in
    /// one shot. Anything that has to touch files needs this: a model cannot
    /// write a correct change to a file it has not read.
    #[serde(default)]
    pub tools: bool,
    /// Procedure the role follows, by name. Loaded from the skills directory.
    #[serde(default)]
    pub skill: Option<String>,
    /// How many tool steps before the node gives up.
    #[serde(default)]
    pub max_steps: Option<u32>,
    /// Room for this role's answer, in tokens. A role that sends whole files
    /// needs more than one that writes a paragraph, and the same number is
    /// held back from the context budget — so the two can never disagree.
    /// Falls back to a constant until the model's measured profile is read
    /// (T-052).
    #[serde(default)]
    pub max_output: Option<u32>,
    /// When true, this role's document is refused unless the journal shows it
    /// actually wrote a file. A role that changes code and reports success
    /// without touching anything is the failure this exists to catch.
    #[serde(default)]
    pub must_write: bool,
    /// When true, this role's unknowns run the clarification ladder: two
    /// self-resolution passes, then the human is asked.
    #[serde(default)]
    pub clarifies: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestBaseline {
    pub total: u32,
    pub failed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunState {
    pub run_id: String,
    pub workflow: String,
    pub states: BTreeMap<NodeId, NodeState>,
    pub attempts: BTreeMap<NodeId, u32>,
    /// Node -> the document file it produced.
    pub documents: BTreeMap<NodeId, String>,
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// Test counts taken before any agent acted. Without them a final "2 tests
    /// pass" cannot be told apart from a run that destroyed the other four.
    #[serde(default)]
    pub baseline: Option<TestBaseline>,
    #[serde(default)]
    pub finished: Option<String>,
    #[serde(default)]
    pub failure: Option<String>,
}

impl RunState {
    pub fn new(run_id: &str, workflow: &str, graph: &Graph) -> Self {
        Self {
            run_id: run_id.to_string(),
            workflow: workflow.to_string(),
            states: graph.nodes.iter().map(|n| (n.id.clone(), NodeState::Idle)).collect(),
            attempts: BTreeMap::new(),
            documents: BTreeMap::new(),
            checkpoint: None,
            baseline: None,
            finished: None,
            failure: None,
        }
    }

    pub fn state(&self, id: &str) -> NodeState {
        self.states.get(id).copied().unwrap_or(NodeState::Idle)
    }

    /// Atomic: written to a temporary file, then renamed. A half-written
    /// run.json would be worse than none.
    pub fn save(&self, run_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(run_dir)?;
        let tmp = run_dir.join("run.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self)?)?;
        std::fs::rename(tmp, run_dir.join("run.json"))?;
        Ok(())
    }

    pub fn load(run_dir: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(run_dir.join("run.json"))?;
        Ok(serde_json::from_str(&text)?)
    }
}

/// A prompt together with its measurement.
///
/// SDD §7 forbids "it will probably fit", and until now nothing in the engine
/// counted what it was about to send: the tokenizer was used on documents and
/// on the compaction threshold, never on the prompt itself.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Context {
    pub text: String,
    /// Named parts in assembly order, each sized by how much it grew the
    /// whole. Sized that way on purpose: measuring the pieces separately and
    /// adding them up is not the count of the joined text under any tokenizer
    /// that rounds, and it is the joined text that gets sent.
    pub parts: Vec<(&'static str, usize)>,
    pub total: usize,
}

impl Context {
    pub fn build(parts: &[(&'static str, String)], tokenizer: &dyn Tokenizer) -> Self {
        let mut out = Context::default();
        for (name, text) in parts {
            if text.is_empty() {
                continue;
            }
            out.text.push_str(text);
            let now = tokenizer.count(&out.text);
            out.parts.push((name, now.saturating_sub(out.total)));
            out.total = now;
        }
        out
    }

    /// One line for the run log: `system 61 + contract 210 + inputs 640 = 911 tokens`.
    /// An overflow has to be visible in the record, not deduced from how the
    /// model behaved.
    pub fn breakdown(&self) -> String {
        let parts: Vec<String> = self.parts.iter().map(|(n, c)| format!("{n} {c}")).collect();
        format!("{} = {} tokens", parts.join(" + "), self.total)
    }
}

/// Room held back for the answer, in tokens: the role's own number if it has
/// one, otherwise the constant this used to be everywhere. The same value is
/// given to the model as `num_predict` and subtracted from the context budget,
/// so a role cannot be promised room the budget does not leave it.
fn answer_room(role: &RoleSpec, fallback: u32) -> u32 {
    role.max_output.unwrap_or(fallback)
}

/// One line of the "what else this run produced" block, with what shedding
/// needs to know about it.
struct Digest {
    file: String,
    line: String,
    is_task: bool,
    kept: bool,
}

pub enum Event<'a> {
    NodeState { id: &'a str, state: NodeState },
    Token(&'a str),
    Note(String),
}

pub struct Engine<'a> {
    pub graph: Graph,
    pub roles: BTreeMap<String, RoleSpec>,
    /// Slot name -> concrete model. Replacing Qwen is a change here, not in
    /// every role.
    pub slots: BTreeMap<String, String>,
    pub backend: &'a dyn ModelBackend,
    pub tokenizer: &'a dyn Tokenizer,
    pub run_dir: PathBuf,
    /// Where role procedures live. Absent means roles name no skills.
    pub skills_dir: Option<PathBuf>,
    /// How this project runs its tests. Present means a baseline is taken and
    /// regressions are refused.
    pub test_command: Option<Vec<String>>,
    /// The project root, for accumulated knowledge. Absent means answers are
    /// used for this run only and not remembered.
    pub project_dir: Option<PathBuf>,
    /// Who answers a clarification. Absent means unknowns are recorded and the
    /// run proceeds, which is the honest behaviour when nobody can be asked.
    pub clarifier: Option<&'a dyn ClarificationAuthority>,
    back_edges: BTreeSet<(NodeId, NodeId)>,
    /// Edges a Branch declined. Readiness ignores them.
    declined: BTreeSet<(NodeId, NodeId)>,
    /// Tool nodes act through this, so a command is gated and journalled like
    /// anything else an agent asks for. Absent means the graph may not contain
    /// Tool nodes, which is checked rather than assumed.
    pub dispatcher: Option<Dispatcher<'a>>,
}

struct Sink<'f>(&'f mut (dyn FnMut(Event<'_>) + Send));
impl TokenSink for Sink<'_> {
    fn token(&mut self, t: &str) {
        (self.0)(Event::Token(t));
    }
}

impl<'a> Engine<'a> {
    pub fn new(
        graph: Graph,
        roles: BTreeMap<String, RoleSpec>,
        slots: BTreeMap<String, String>,
        backend: &'a dyn ModelBackend,
        tokenizer: &'a dyn Tokenizer,
        run_dir: PathBuf,
    ) -> Self {
        let back_edges = graph.back_edges();
        Self {
            graph,
            roles,
            slots,
            backend,
            tokenizer,
            run_dir,
            skills_dir: None,
            test_command: None,
            project_dir: None,
            clarifier: None,
            back_edges,
            declined: BTreeSet::new(),
            dispatcher: None,
        }
    }

    fn forward_predecessors(&self, id: &str) -> Vec<NodeId> {
        self.graph
            .edges
            .iter()
            .filter(|e| e.to == id)
            .filter(|e| !self.back_edges.contains(&(e.from.clone(), e.to.clone())))
            .filter(|e| !self.declined.contains(&(e.from.clone(), e.to.clone())))
            .map(|e| e.from.clone())
            .collect()
    }

    fn ready(&self, run: &RunState) -> Option<NodeId> {
        self.graph
            .nodes
            .iter()
            .filter(|n| run.state(&n.id) == NodeState::Idle)
            .find(|n| {
                let preds = self.forward_predecessors(&n.id);
                if n.kind == NodeKind::Input {
                    return true;
                }
                if preds.is_empty() {
                    // Every way in was declined by a branch.
                    return false;
                }
                preds.iter().all(|p| matches!(run.state(p), NodeState::Done | NodeState::Skipped))
            })
            .map(|n| n.id.clone())
    }

    /// Marks a node and, transitively, anything left with no live way in.
    fn skip_from(&self, run: &mut RunState, id: &str) {
        run.states.insert(id.to_string(), NodeState::Skipped);
        for s in self.graph.successors(id).into_iter().cloned().collect::<Vec<_>>() {
            if run.state(&s) != NodeState::Idle {
                continue;
            }
            let preds = self.forward_predecessors(&s);
            let all_gone = !preds.is_empty()
                && preds.iter().all(|p| run.state(p) == NodeState::Skipped);
            if all_gone {
                self.skip_from(run, &s);
            }
        }
    }

    /// Runs to completion, or to the first failure.
    pub async fn run(&mut self, run: &mut RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        while let Some(id) = self.ready(run) {
            let node = self.graph.node(&id).cloned().ok_or_else(|| anyhow!("no node {id}"))?;

            // The transition is persisted before it is announced, so a crash
            // between the two re-executes rather than skips.
            run.states.insert(id.clone(), NodeState::Running);
            run.save(&self.run_dir)?;
            on_event(Event::NodeState { id: &id, state: NodeState::Running });

            let outcome = self.execute(&node, run, on_event).await;

            match outcome {
                Ok(()) => {
                    run.states.insert(id.clone(), NodeState::Done);
                    run.save(&self.run_dir)?;
                    on_event(Event::NodeState { id: &id, state: NodeState::Done });
                }
                Err(e) => {
                    run.states.insert(id.clone(), NodeState::Failed);
                    run.failure = Some(format!("{id}: {e}"));
                    run.save(&self.run_dir)?;
                    on_event(Event::NodeState { id: &id, state: NodeState::Failed });
                    return Err(e);
                }
            }
        }
        run.finished = Some(now_utc());
        run.save(&self.run_dir)?;
        Ok(())
    }

    async fn execute(&mut self, node: &Node, run: &mut RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        match node.kind {
            NodeKind::Input | NodeKind::Merge | NodeKind::Loop | NodeKind::Gate => Ok(()),
            NodeKind::Output => self.compose_report(run, on_event),
            NodeKind::Branch => self.branch(node, run, on_event),
            NodeKind::Tool => self.tool(node, run, on_event),
            NodeKind::Agent | NodeKind::Foreman => self.agent(node, run, on_event).await,
        }
    }

    fn branch(&mut self, node: &Node, run: &mut RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        let outgoing: Vec<_> = self.graph.edges.iter().filter(|e| e.from == node.id).cloned().collect();
        let mut chosen: Option<NodeId> = None;

        for edge in outgoing.iter().filter(|e| e.when.is_some()) {
            let cond = edge.when.as_ref().expect("filtered");
            let Some(doc_node) = run.documents.iter().find(|(_, f)| **f == cond.document).map(|(n, _)| n.clone()) else {
                continue;
            };
            let path = self.run_dir.join(&run.documents[&doc_node]);
            let text = std::fs::read_to_string(&path)?;
            let doc = document::parse(&text, self.tokenizer).map_err(|e| anyhow!("{e}"))?;
            let actual = doc.header.results.get(&cond.field).cloned().unwrap_or_default();
            if cond.holds(&actual) {
                on_event(Event::Note(format!("branch: {}.{} = {} -> {}", cond.document, cond.field, actual, edge.to)));
                chosen = Some(edge.to.clone());
                break;
            }
        }

        if chosen.is_none() {
            chosen = outgoing.iter().find(|e| e.when.is_none()).map(|e| e.to.clone());
            if let Some(c) = &chosen {
                on_event(Event::Note(format!("branch: no condition held, taking the fallback -> {c}")));
            }
        }

        let chosen = chosen.ok_or_else(|| anyhow!("branch `{}` has no edge to take and no fallback", node.id))?;

        for edge in outgoing {
            if edge.to != chosen {
                self.declined.insert((edge.from.clone(), edge.to.clone()));
                if run.state(&edge.to) == NodeState::Idle {
                    let preds = self.forward_predecessors(&edge.to);
                    if preds.is_empty() {
                        self.skip_from(run, &edge.to);
                    }
                }
            }
        }
        Ok(())
    }

    /// The final report is assembled by the harness from digests, with no model
    /// involved. Nothing can distort earlier conclusions on the last step, it
    /// costs neither memory nor tokens, and it is instant.
    fn compose_report(&mut self, run: &RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        let mut out = String::new();
        out.push_str(&format!("# Run report — {}\n\n", run.run_id));
        out.push_str(&format!("Workflow: {}\n", run.workflow));
        out.push_str(&format!("Assembled: {}\n\n", now_utc()));

        out.push_str("## What each step concluded\n\n");
        for (node, file) in &run.documents {
            let text = std::fs::read_to_string(self.run_dir.join(file))?;
            match document::parse(&text, self.tokenizer) {
                Ok(doc) => {
                    out.push_str(&format!("### {node} — `{file}`\n\n"));
                    out.push_str(&format!("{}\n\n", doc.header.digest.trim()));
                    if !doc.header.results.is_empty() {
                        let pairs: Vec<String> =
                            doc.header.results.iter().map(|(k, v)| format!("{k}: {v}")).collect();
                        out.push_str(&format!("`{}`\n\n", pairs.join(" · ")));
                    }
                }
                Err(e) => out.push_str(&format!("### {node} — `{file}`\n\nUnreadable: {e}\n\n")),
            }
        }

        let failed: Vec<&String> = run
            .states
            .iter()
            .filter(|(_, s)| **s == NodeState::Failed)
            .map(|(id, _)| id)
            .collect();
        let skipped: Vec<&String> = run
            .states
            .iter()
            .filter(|(_, s)| **s == NodeState::Skipped)
            .map(|(id, _)| id)
            .collect();

        out.push_str("## Course of the run\n\n");
        out.push_str(&format!("- Documents produced: {}\n", run.documents.len()));
        out.push_str(&format!("- Repeated attempts: {}\n", run.attempts.values().filter(|a| **a > 1).count()));
        if !skipped.is_empty() {
            out.push_str(&format!("- Not taken: {}\n", skipped.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
        if !failed.is_empty() {
            out.push_str(&format!("- **Failed: {}**\n", failed.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")));
        }
        if let Some(c) = &run.checkpoint {
            out.push_str(&format!("- Checkpoint: `{c}` — this run is reversible in one action\n"));
        }

        std::fs::create_dir_all(&self.run_dir)?;
        std::fs::write(self.run_dir.join("report.md"), out)?;
        on_event(Event::Note("harness assembled report.md from digests".into()));
        Ok(())
    }

    /// A Tool node runs a command through the dispatcher — so it is classified,
    /// gated where the classification says so, and journalled — and the harness
    /// turns the result into a document without a model.
    fn tool(&mut self, node: &Node, run: &mut RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        let command = node
            .command
            .clone()
            .filter(|c| !c.is_empty())
            .ok_or_else(|| anyhow!("`{}` is a tool node with no command", node.id))?;
        let output = node.output.clone().ok_or_else(|| anyhow!("`{}` declares no output", node.id))?;
        let dispatcher = self
            .dispatcher
            .as_mut()
            .ok_or_else(|| anyhow!("`{}` needs a dispatcher, and none is configured", node.id))?;

        dispatcher.for_node(&node.id, 1);
        on_event(Event::Note(format!("{} runs `{}`", node.id, command.join(" "))));

        let outcome = dispatcher.dispatch(
            crate::sandbox::ToolCall::RunCommand {
                program: command[0].clone(),
                args: command[1..].to_vec(),
            },
            node.gate,
        )?;

        let (exit_code, stdout, stderr) = match outcome {
            Outcome::Ran { exit_code, stdout, stderr } => (exit_code, stdout, stderr),
            Outcome::Refused(why) => return Err(anyhow!("`{}` was refused: {why}", node.id)),
            other => return Err(anyhow!("`{}` produced an unexpected outcome: {other:?}", node.id)),
        };

        let combined = if stderr.trim().is_empty() { stdout.clone() } else { format!("{stdout}\n{stderr}") };
        let doc = tool_document(&run.run_id, &node.id, &command.join(" "), exit_code, &combined);
        std::fs::create_dir_all(&self.run_dir)?;
        std::fs::write(self.run_dir.join(&output), doc)?;
        run.documents.insert(node.id.clone(), output.clone());
        on_event(Event::Note(format!("{} wrote {output} (exit {exit_code})", node.id)));
        Ok(())
    }

    async fn agent(&mut self, node: &Node, run: &mut RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        let role_name = node.role.clone().ok_or_else(|| anyhow!("`{}` names no role", node.id))?;
        let role = self.roles.get(&role_name).cloned().ok_or_else(|| anyhow!("no role `{role_name}`"))?;
        let slot = node.slot.clone().unwrap_or_else(|| role.slot.clone());
        let model = self
            .slots
            .get(&slot)
            .cloned()
            .ok_or_else(|| anyhow!("slot `{slot}` is not bound to a model"))?;
        let artifact = Artifact::parse(&role.artifact).ok_or_else(|| anyhow!("unknown artifact `{}`", role.artifact))?;
        let output = node.output.clone().ok_or_else(|| anyhow!("`{}` declares no output", node.id))?;

        if role.tools {
            return self.agent_with_tools(node, &role, &model, artifact, &output, run, on_event).await;
        }

        let mut correction: Option<String> = None;
        let start_attempt = run.attempts.get(&node.id).copied().unwrap_or(0) + 1;

        for offset in 0..MAX_ATTEMPTS {
            let attempt = start_attempt + offset;
            run.attempts.insert(node.id.clone(), attempt);

            let room = answer_room(&role, 1400);
            let context = self.assemble(&node.id, &role, run, attempt, &model, artifact, room as usize, on_event)?;
            let mut prompt = context.text;
            if let Some(c) = &correction {
                prompt.push_str("\n\nCORRECTION\n");
                prompt.push_str(c);
            }

            let opts = Options {
                num_ctx: role.window,
                num_predict: room as i32,
                temperature: role.temperature,
                repeat_penalty: 1.1,
                seed: None,
            };

            on_event(Event::Note(format!("{} attempt {attempt} on {model}", node.id)));
            let completion = {
                let mut sink = Sink(on_event);
                self.backend.generate(&model, &prompt, &opts, "300s", &mut sink).await?
            };

            // A cut-off answer is not a malformed one. Saying "the document was
            // not recognised" here blames the model for the harness's own
            // limit, and the correction it produces — "mind the format" — is
            // advice about the wrong problem.
            if completion.truncated {
                on_event(Event::Note(format!(
                    "{}: the answer was cut off at {room} tokens",
                    node.id
                )));
                correction = Some(format!(
                    "Your previous reply was cut off after {room} tokens: it never reached the end of the \
                     document. Nothing is wrong with the format. Answer again and make it fit — keep the body \
                     to what matters and the digest to two or three sentences."
                ));
                continue;
            }

            let raw = strip_outer_fence(&completion.text);

            match document::parse(&raw, self.tokenizer) {
                Ok(mut doc) => {
                    stamp(&mut doc, &run.run_id, &node.id, attempt, &model);
                    let name = if attempt == 1 { output.clone() } else { numbered(&output, attempt) };
                    std::fs::create_dir_all(&self.run_dir)?;
                    std::fs::write(self.run_dir.join(&name), document::render(&doc))?;
                    run.documents.insert(node.id.clone(), name.clone());
                    on_event(Event::Note(format!("{} wrote {name}", node.id)));

                    if role.clarifies {
                        self.clarification_ladder(&node.id, &role, &model, artifact, &name, run, on_event)
                            .await?;
                    }

                    // A Patch is the one artifact that is not finished when it
                    // is written: it describes a change that has to reach the
                    // files. Applying it goes through the dispatcher like any
                    // other effect, so it is classified, gated and journalled.
                    if artifact == Artifact::Patch {
                        // A patch that does not apply is a correctable mistake,
                        // not a dead end: git says exactly what it choked on,
                        // and that goes back to the model like any other
                        // rejection. Failing the node here would waste the two
                        // remaining attempts the design already grants.
                        match self.apply_patch(&node.id, &doc, on_event) {
                            Ok(()) => return Ok(()),
                            Err(e) => {
                                on_event(Event::Note(format!("{}: patch did not apply — {e}", node.id)));
                                correction = Some(format!(
                                    "The previous reply was rejected: the diff did not apply. git said: {e}\n\
                                     Reply with the corrected document only. Re-read the file contents you were \
                                     given and quote context lines exactly as they appear there, including \
                                     indentation. Do not guess line numbers — write the hunk header as \
                                     @@ -1 +1 @@ and let the context lines carry the position."
                                ));
                                continue;
                            }
                        }
                    }
                    return Ok(());
                }
                Err(e) => {
                    on_event(Event::Note(format!("{} rejected: {e}", node.id)));
                    correction = Some(e.guidance());
                }
            }
        }
        Err(anyhow!("`{}` produced no valid document in {MAX_ATTEMPTS} attempts", node.id))
    }

    /// Two self-resolution passes, then the human — and never the other way
    /// round. Most unknowns dissolve on reading, and interrupting first spends
    /// the one resource the product cannot replace.
    #[allow(clippy::too_many_arguments)]
    async fn clarification_ladder(
        &mut self,
        node_id: &str,
        role: &RoleSpec,
        model: &str,
        artifact: Artifact,
        file: &str,
        run: &mut RunState,
        on_event: &mut (dyn FnMut(Event<'_>) + Send),
    ) -> Result<()> {
        for pass in 1..=SELF_RESOLUTION_PASSES {
            let doc = self.read_document(file)?;
            let left = clarify::unknown_count(&doc);
            if left == 0 {
                return Ok(());
            }
            let questions = clarify::extract_questions(&doc.body);
            if questions.is_empty() {
                return Ok(());
            }
            on_event(Event::Note(format!(
                "{node_id}: {left} unknown(s) — self-resolution pass {pass} of {SELF_RESOLUTION_PASSES}"
            )));
            let instruction = clarify::self_resolution_prompt(&questions);
            self.rerun_agent(node_id, role, model, artifact, file, &instruction, run, on_event).await?;
        }

        let doc = self.read_document(file)?;
        let left = clarify::unknown_count(&doc);
        if left == 0 {
            return Ok(());
        }
        let questions = clarify::extract_questions(&doc.body);
        if questions.is_empty() {
            return Ok(());
        }

        let Some(clarifier) = self.clarifier else {
            on_event(Event::Note(format!(
                "{node_id}: {left} unknown(s) remain and nobody can be asked — they stay recorded in {file}"
            )));
            return Ok(());
        };

        on_event(Event::Note(format!("{node_id}: asking the human {} question(s)", questions.len())));
        let answers = clarifier.ask(node_id, &questions);
        let pairs: Vec<(String, String)> = questions
            .iter()
            .cloned()
            .zip(answers.into_iter())
            .filter(|(_, a)| !a.trim().is_empty())
            .collect();

        if pairs.is_empty() {
            on_event(Event::Note(format!("{node_id}: no answers given, the unknowns stay recorded")));
            return Ok(());
        }

        if let Some(project) = &self.project_dir {
            clarify::append_knowledge(project, &pairs)?;
            on_event(Event::Note(format!("{node_id}: {} answer(s) kept for later runs", pairs.len())));
        }

        let instruction = clarify::answered_prompt(&pairs);
        self.rerun_agent(node_id, role, model, artifact, file, &instruction, run, on_event).await
    }

    /// Runs the project's tests and reads its counts. Returns None when the
    /// output carries no summary, because a missing count is not a zero.
    pub fn measure_tests(&mut self, command: &[String]) -> Option<TestBaseline> {
        let dispatcher = self.dispatcher.as_mut()?;
        dispatcher.for_node("baseline", 1);
        let outcome = dispatcher
            .dispatch(
                crate::sandbox::ToolCall::RunCommand {
                    program: command[0].clone(),
                    args: command[1..].to_vec(),
                },
                false,
            )
            .ok()?;
        let crate::dispatcher::Outcome::Ran { stdout, stderr, .. } = outcome else { return None };
        let counts = parse_test_output(&format!("{stdout}\n{stderr}"));
        counts.conclusive.then_some(TestBaseline { total: counts.total, failed: counts.failed })
    }

    /// Taken before any agent acts, which is the only moment it means anything.
    pub fn take_baseline(&mut self, run: &mut RunState, on_event: &mut (dyn FnMut(Event<'_>) + Send)) {
        let Some(cmd) = self.test_command.clone() else { return };
        match self.measure_tests(&cmd) {
            Some(b) => {
                on_event(Event::Note(format!("baseline: {} tests, {} failing", b.total, b.failed)));
                run.baseline = Some(b);
            }
            None => on_event(Event::Note(
                "baseline: the test command produced no summary, so regressions cannot be detected".into(),
            )),
        }
    }

    /// The test command as an agent has to send it, if this project has one.
    /// The harness stores its own shell wrapper in the argv; `run_command` adds
    /// one itself, so what is shown is the command line inside it.
    fn test_command_line(&self) -> Option<String> {
        self.test_command.as_deref().map(chat::command_line)
    }

    fn read_document(&self, file: &str) -> Result<Document> {
        let text = std::fs::read_to_string(self.run_dir.join(file))?;
        document::parse(&text, self.tokenizer).map_err(|e| anyhow!("{file}: {e}"))
    }

    /// Re-runs a role over its own last document, replacing it in place. Used
    /// only by the ladder, where the point is a better answer to the same
    /// question rather than a record of a second attempt.
    #[allow(clippy::too_many_arguments)]
    async fn rerun_agent(
        &mut self,
        node_id: &str,
        role: &RoleSpec,
        model: &str,
        artifact: Artifact,
        file: &str,
        instruction: &str,
        run: &mut RunState,
        on_event: &mut (dyn FnMut(Event<'_>) + Send),
    ) -> Result<()> {
        let room = answer_room(role, 1400);
        let base = self.assemble(node_id, role, run, 1, model, artifact, room as usize, on_event)?;
        let prompt = format!("{}\n\n{}\n\n{instruction}", base.text, self.load_skill(role));
        let opts = Options {
            num_ctx: role.window,
            num_predict: room as i32,
            temperature: role.temperature,
            repeat_penalty: 1.1,
            seed: None,
        };
        let reply = {
            let mut sink = Sink(on_event);
            self.backend.generate(model, &prompt, &opts, "300s", &mut sink).await?
        };
        // A rewrite that ran out of room is not an improvement on what is
        // already on disk, even if what arrived happens to parse.
        if reply.truncated {
            on_event(Event::Note(format!(
                "{node_id}: the rewrite was cut off at {room} tokens, keeping the previous document"
            )));
            return Ok(());
        }
        match document::parse(&strip_outer_fence(&reply.text), self.tokenizer) {
            Ok(mut doc) => {
                stamp(&mut doc, &run.run_id, node_id, 1, model);
                std::fs::write(self.run_dir.join(file), document::render(&doc))?;
                Ok(())
            }
            Err(e) => {
                // The document already on disk is valid; a malformed rewrite is
                // discarded rather than allowed to replace it.
                on_event(Event::Note(format!("{node_id}: the rewrite was malformed ({e}), keeping the previous document")));
                Ok(())
            }
        }
    }

    /// An agent that works in steps: it reads, changes, runs, looks at what
    /// happened, and only then writes its document. One shot cannot do this —
    /// a model cannot correctly change a file it has not read, and cannot know
    /// whether its change works without running something.
    #[allow(clippy::too_many_arguments)]
    async fn agent_with_tools(
        &mut self,
        node: &Node,
        role: &RoleSpec,
        model: &str,
        artifact: Artifact,
        output: &str,
        run: &mut RunState,
        on_event: &mut (dyn FnMut(Event<'_>) + Send),
    ) -> Result<()> {
        let max_steps = role.max_steps.unwrap_or(12);
        // A tool-using role sends whole files, so it needs more room than one
        // that answers in prose — which is exactly what the role can now say.
        let room = answer_room(role, 1600);
        let base = self.assemble(&node.id, role, run, 1, model, artifact, room as usize, on_event)?;
        let skill = self.load_skill(role);
        // The harness measured the baseline with a command it used to keep to
        // itself. The model then guessed: six steps of the run of 2026-08-19
        // went on `python -m unittest` on a machine that has only `python3`
        // (finding 37).
        let tests = match self.test_command_line() {
            Some(line) => format!(
                "\n\nThe tests of this project are run by exactly this command line and no other:\n\
                 {line}\n\
                 Send that to run_command as it stands. It is the command this harness measured with \
                 before you started, so any other one counts something else."
            ),
            None => String::new(),
        };

        // Identical calls that came to nothing, and how many times each. Counted
        // over the whole node: repeating in step 6 what failed in step 2 is the
        // same standing still.
        let mut failures: BTreeMap<String, usize> = BTreeMap::new();

        // The message structure the model was trained on: the role and its
        // procedure are the system message, the task and its documents are the
        // user message, and tool results come back in their own role. Sending
        // one concatenated blob instead competes with that training.
        let mut messages = vec![
            Message::system(format!(
                "{}{skill}\n\nWork in steps. Use the tools to read and change files and to run the tests. \
                 Always read a file before you change it, and always send a whole file when you write. \
                 When the work is done and the tests pass, reply with your document and no tool call.\n\n\
                 The documents you were given are already below in full. They are not files on disk — \
                 read_file is for files of the project only.{tests}",
                role.system
            )),
            Message::user(base.text),
        ];

        for step in 1..=max_steps {
            let opts = Options {
                num_ctx: role.window,
                num_predict: room as i32,
                temperature: role.temperature,
                repeat_penalty: 1.1,
                seed: None,
            };
            on_event(Event::Note(format!("{} step {step}/{max_steps} on {model}", node.id)));

            let reply = {
                let mut sink = Sink(on_event);
                self.backend
                    .chat(model, &messages, chat::tool_schemas(), &opts, "300s", &mut sink)
                    .await?
            };

            // A reply cut off at the limit is not a reply. Its tool call, if it
            // had started one, is half-written; parsing what arrived and
            // reporting "no valid document" would blame the model for the size
            // of the room it was given.
            if reply.truncated {
                on_event(Event::Note(format!(
                    "{}: the answer was cut off at {room} tokens",
                    node.id
                )));
                messages.push(Message::assistant(reply.text.clone()));
                messages.push(Message::user(format!(
                    "Your reply was cut off after {room} tokens, so nothing in it could be used. Do not send \
                     whole files: call replace_in_file with the smallest fragment that identifies the place, \
                     one call at a time."
                )));
                continue;
            }

            // Three layers, in order of how much the model is trusted to have
            // got the mechanics right. qwen2.5-coder declares the tools
            // capability and then writes its calls into the text, so the second
            // layer is not theoretical.
            let (requests, source) = if !reply.tool_calls.is_empty() {
                (reply.tool_calls.clone(), CallSource::Native)
            } else {
                let recovered = chat::recover_from_text(&reply.text);
                if !recovered.is_empty() {
                    (recovered, CallSource::JsonInText)
                } else {
                    (Vec::new(), CallSource::LineSyntax)
                }
            };

            if requests.is_empty() {
                match document::parse(&strip_outer_fence(&reply.text), self.tokenizer) {
                    Ok(mut doc) => {
                        // The claim is checked against the record before it is
                        // accepted. An agent reporting work it did not do is
                        // worse than one that fails: a failure is visible.
                        if role.must_write {
                            let written = self
                                .dispatcher
                                .as_ref()
                                .map(|d| {
                                    crate::journal::Journal::load(d.journal().path())
                                        .map(|es| crate::journal::Journal::writes_by(&es, &node.id))
                                        .unwrap_or(0)
                                })
                                .unwrap_or(0);
                            if written == 0 {
                                on_event(Event::Note(format!(
                                    "{}: reported completion but the journal shows no file written — refused",
                                    node.id
                                )));
                                messages.push(Message::assistant(reply.text.clone()));
                                messages.push(Message::user(
                                    "You reported the work as finished, but no file has been written. The record of \
                                     this run shows only reads. Do the work: call write_file with the whole new \
                                     contents of each file you need to change, then run the tests, then reply with \
                                     your document.",
                                ));
                                continue;
                            }
                        }
                        // Writing something is not the same as not destroying
                        // something. A run that adds two tests and deletes four
                        // has gone backwards, and only the baseline can say so.
                        let baseline_cmd = self.test_command.clone();
                        if let (Some(base), Some(cmd)) = (run.baseline, baseline_cmd) {
                            if let Some(now) = self.measure_tests(&cmd) {
                                if now.total < base.total {
                                    on_event(Event::Note(format!(
                                        "{}: {} tests before, {} now — refused",
                                        node.id, base.total, now.total
                                    )));
                                    messages.push(Message::assistant(reply.text.clone()));
                                    messages.push(Message::user(format!(
                                        "There were {} tests before you started and there are {} now. You have \
                                         deleted tests that existed. Restore every test that was there and keep \
                                         your own as well, then run the tests again.",
                                        base.total, now.total
                                    )));
                                    continue;
                                }
                            }
                        }
                        stamp(&mut doc, &run.run_id, &node.id, step, model);
                        std::fs::create_dir_all(&self.run_dir)?;
                        std::fs::write(self.run_dir.join(output), document::render(&doc))?;
                        run.documents.insert(node.id.clone(), output.to_string());
                        run.attempts.insert(node.id.clone(), step);
                        on_event(Event::Note(format!("{} wrote {output} after {step} steps", node.id)));
                        return Ok(());
                    }
                    Err(e) => {
                        on_event(Event::Note(format!("{} not a document yet: {e}", node.id)));
                        messages.push(Message::assistant(reply.text.clone()));
                        messages.push(Message::user(format!(
                            "That was neither a tool call nor a valid document, so the step did nothing. \
                             Either call a tool now, or reply with the document. {}",
                            e.guidance()
                        )));
                        continue;
                    }
                }
            }

            if source != CallSource::Native {
                on_event(Event::Note(format!(
                    "{}: the call came as {source:?} rather than through the tool channel",
                    node.id
                )));
            }
            // The calls go into the turn that made them. Without them the
            // model is shown an empty assistant turn followed by results it
            // never asked for, and it asks again — five reads of one file in
            // five steps, in the run of 2026-08-16.
            messages.push(Message::assistant_with_calls(reply.text.clone(), requests.clone()));

            for req in &requests {
                let call = match chat::to_tool_call(req) {
                    Ok(c) => c,
                    Err(why) => {
                        on_event(Event::Note(format!("{}: unusable call — {why}", node.id)));
                        messages.push(Message::tool_result(&req.name, why));
                        continue;
                    }
                };
                let summary = describe(&call);
                let outcome = {
                    let dispatcher = self
                        .dispatcher
                        .as_mut()
                        .ok_or_else(|| anyhow!("`{}` needs tools and no dispatcher is configured", node.id))?;
                    dispatcher.for_node(&node.id, step);
                    dispatcher.dispatch(call.clone(), node.gate)
                };
                let mut performed = true;
                // Why the call came to nothing, if it did. A command that ran
                // and failed is as fruitless as one that was refused, and a
                // model repeats both the same way.
                let mut fruitless = None;
                let mut text = match outcome {
                    Ok(crate::dispatcher::Outcome::Read(t)) => truncate(&t, 8000),
                    Ok(crate::dispatcher::Outcome::Wrote) => "written".to_string(),
                    Ok(crate::dispatcher::Outcome::Deleted) => "deleted".to_string(),
                    Ok(crate::dispatcher::Outcome::Ran { exit_code, stdout, stderr }) => {
                        if exit_code != 0 {
                            fruitless = Some(format!("exit {exit_code}"));
                        }
                        format!("exit {exit_code}\n{}", truncate(&format!("{stdout}{stderr}"), 4000))
                    }
                    Ok(crate::dispatcher::Outcome::Refused(why)) => {
                        performed = false;
                        fruitless = Some("a refusal".to_string());
                        format!("REFUSED: {why}")
                    }
                    Err(e) => {
                        performed = false;
                        fruitless = Some("an error".to_string());
                        format!("ERROR: {e}")
                    }
                };
                on_event(Event::Note(if performed {
                    format!("{}: {summary}", node.id)
                } else {
                    format!("{}: {summary} — NOT DONE, {}", node.id, text.lines().next().unwrap_or(""))
                }));
                // A call that failed and is sent again unchanged fails again.
                // Only the harness sees the repetition: the model has just been
                // shown the failure and sent the same thing anyway.
                if let Some(how) = fruitless {
                    let times = *failures
                        .entry(format!("{call:?}"))
                        .and_modify(|n| *n += 1)
                        .or_insert(1);
                    if times >= 3 {
                        let instead = match (&call, self.test_command_line()) {
                            (crate::sandbox::ToolCall::RunCommand { .. }, Some(line)) => {
                                format!(" The tests of this project are run by: {line}")
                            }
                            _ => String::new(),
                        };
                        on_event(Event::Note(format!(
                            "{}: {summary} — the same call has now failed {times} times",
                            node.id
                        )));
                        text.push_str(&format!(
                            "\n\nThe same call has now failed {times} times, every time with {how}. It will \
                             not start working: send something different, or do something else.{instead}"
                        ));
                    }
                }
                messages.push(Message::tool_result(&req.name, text));
            }

            // The conversation is the context here, so it must not outgrow the
            // window. The system message and the task are never dropped.
            let total: usize = messages.iter().map(|m| self.tokenizer.count(&m.content)).sum();
            if total > (role.window as usize) * 2 / 3 && messages.len() > 4 {
                let tail = messages.split_off(messages.len() - 2);
                messages.truncate(2);
                messages.push(Message::user("[earlier steps were compacted away]"));
                messages.extend(tail);
                on_event(Event::Note(format!("{} compacted its step history", node.id)));
            }
        }

        Err(anyhow!("`{}` used all {max_steps} steps without producing its document", node.id))
    }

    fn load_skill(&self, role: &RoleSpec) -> String {
        let (Some(dir), Some(name)) = (&self.skills_dir, &role.skill) else { return String::new() };
        match std::fs::read_to_string(dir.join(format!("{name}.md"))) {
            Ok(t) => format!("\n\nHOW TO DO THIS WELL\n{t}"),
            Err(_) => String::new(),
        }
    }

    /// Sends a Coder's diff to the files. The gate the workflow puts after the
    /// node is what the user actually sees here.
    fn apply_patch(&mut self, node_id: &str, doc: &Document, on_event: &mut (dyn FnMut(Event<'_>) + Send)) -> Result<()> {
        let diff = extract_diff(&doc.body);
        if diff.trim().is_empty() {
            return Err(anyhow!("`{node_id}` produced a Patch document with no diff in it"));
        }
        let dispatcher = self
            .dispatcher
            .as_mut()
            .ok_or_else(|| anyhow!("`{node_id}` wants to change files and no dispatcher is configured"))?;
        dispatcher.for_node(node_id, 1);

        match dispatcher.dispatch(crate::sandbox::ToolCall::ApplyPatch { diff }, true)? {
            crate::dispatcher::Outcome::Wrote => {
                on_event(Event::Note(format!("{node_id}: patch applied to the working tree")));
                Ok(())
            }
            crate::dispatcher::Outcome::Refused(why) => {
                Err(anyhow!("the patch from `{node_id}` was refused: {why}"))
            }
            other => Err(anyhow!("applying the patch from `{node_id}` gave {other:?}")),
        }
    }

    /// Full bodies of declared primary inputs, digests of everything else —
    /// measured as it is built, and the measurement put in the run log.
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        &self,
        node_id: &str,
        role: &RoleSpec,
        run: &RunState,
        attempt: u32,
        model: &str,
        artifact: Artifact,
        // Tokens held back for the answer. Hardcoded at each call site until
        // T-006 takes it from the role, which is where it belongs.
        reserve: usize,
        on_event: &mut (dyn FnMut(Event<'_>) + Send),
    ) -> Result<Context> {
        // Everything the owner has answered before. Without this the ladder
        // asks the same questions every day and becomes a tax rather than a
        // safeguard — it is written after every gate, so it must be read
        // before every run.
        let mut knowledge = String::new();
        if let Some(project) = &self.project_dir {
            let known = crate::clarify::read_knowledge(project);
            if !known.trim().is_empty() {
                knowledge.push_str("\n\nWHAT THIS PROJECT'S OWNER HAS ALREADY TOLD US\n");
                knowledge.push_str(&known);
                knowledge.push_str("\nTreat these as settled. Do not ask them again.\n");
            }
        }

        let inputs: Vec<String> = run.documents.values().cloned().collect();
        let contract = format_contract(artifact, &run.run_id, &role.name, attempt, model, &inputs);

        let mut full = String::new();
        // Digests carry the file they came from and whether they are the task,
        // because shedding has to know both. The order is the order of the file
        // names: documents are numbered by the workflow that writes them, and
        // that numbering is the only production order `run.json` records —
        // `documents` is a map keyed by node id, which sorts by node name.
        let mut digests: Vec<Digest> = Vec::new();
        for (_node, file) in run.documents.iter() {
            let text = std::fs::read_to_string(self.run_dir.join(file))?;
            let Ok(doc) = document::parse(&text, self.tokenizer) else { continue };
            if role.primary_inputs.iter().any(|p| p == file) {
                full.push_str(&format!("\n\n--- {file} (in full) ---\n{}", doc.body));
            } else {
                digests.push(Digest {
                    file: file.clone(),
                    line: format!("\n- {file}: {}", doc.header.digest.replace('\n', " ")),
                    is_task: doc.header.artifact == Artifact::Task,
                    kept: true,
                });
            }
        }
        digests.sort_by(|a, b| a.file.cmp(&b.file));
        if !full.is_empty() {
            full = format!("\n\nYOUR INPUTS{full}");
        }

        // The window minus room for the answer. Ollama silently drops whatever
        // does not fit into `num_ctx`, and what it drops first is the front of
        // the prompt — which is where the format contract lives. So the
        // shedding is done here, in the open, in a stated order.
        let budget = (role.window as usize).saturating_sub(reserve);
        let mut keep_knowledge = true;

        loop {
            let block: String = digests.iter().filter(|d| d.kept).map(|d| d.line.as_str()).collect();
            let block = if block.is_empty() {
                String::new()
            } else {
                format!("\n\nWHAT ELSE THIS RUN HAS PRODUCED (digests only)\n{block}")
            };

            let context = Context::build(
                &[
                    ("system", role.system.clone()),
                    ("knowledge", if keep_knowledge { knowledge.clone() } else { String::new() }),
                    ("contract", contract.clone()),
                    ("digests", block),
                    ("inputs", full.clone()),
                ],
                self.tokenizer,
            );

            if context.total <= budget {
                on_event(Event::Note(format!(
                    "{node_id}: context {} of a {budget} token budget",
                    context.breakdown()
                )));
                return Ok(context);
            }

            // Oldest digest first, and never the task: a node may not be made
            // to work on a different job than the one it was given.
            if let Some(d) = digests.iter_mut().find(|d| d.kept && !d.is_task) {
                d.kept = false;
                on_event(Event::Note(format!(
                    "{node_id}: {} tokens against a {budget} budget — dropping the digest of {}",
                    context.total, d.file
                )));
                continue;
            }
            if keep_knowledge && !knowledge.is_empty() {
                keep_knowledge = false;
                on_event(Event::Note(format!(
                    "{node_id}: {} tokens against a {budget} budget — dropping what the owner has told us before",
                    context.total
                )));
                continue;
            }

            return Err(anyhow!(
                "`{node_id}`: the task does not fit in the window — {} against a budget of {budget} ({}). \
                 Nothing is left to drop but the task and the inputs this node was given.",
                context.total,
                context.breakdown()
            ));
        }
    }
}

/// Pulls the unified diff out of a Patch document body, whether the model
/// fenced it or not. Models fence code far more often than they are told not
/// to, and rejecting the document over punctuation would waste an attempt.
pub fn extract_diff(body: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    let mut saw_fence = false;
    for line in body.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            saw_fence = true;
            continue;
        }
        if in_fence {
            out.push_str(line);
            out.push('\n');
        }
    }
    if saw_fence && !out.trim().is_empty() {
        return out;
    }
    // Unfenced: keep from the first diff marker onwards.
    let start = body
        .lines()
        .position(|l| l.starts_with("diff --git") || l.starts_with("--- "))
        .unwrap_or(0);
    body.lines().skip(start).collect::<Vec<_>>().join("\n") + "\n"
}

/// Counts read out of a test runner's own output.
///
/// The exit code alone is not trustworthy: a shell pipeline reports the last
/// stage, so `... | tail` turns a failing suite into a success. Reading the
/// runner's own summary is what the numbers in `results` have to come from,
/// because branches are written against them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TestCounts {
    pub total: u32,
    pub failed: u32,
    pub conclusive: bool,
}

pub fn parse_test_output(output: &str) -> TestCounts {
    let mut c = TestCounts::default();
    for line in output.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Ran ") {
            if let Some(n) = rest.split_whitespace().next().and_then(|n| n.parse::<u32>().ok()) {
                c.total = n;
                c.conclusive = true;
            }
        }
        if t == "OK" || t.starts_with("OK (") {
            c.failed = 0;
            c.conclusive = true;
        }
        if t.starts_with("FAILED") {
            c.conclusive = true;
            let digits: u32 = t
                .split(|ch: char| !ch.is_ascii_digit())
                .filter(|p| !p.is_empty())
                .filter_map(|p| p.parse::<u32>().ok())
                .sum();
            c.failed = digits.max(1);
        }
    }
    c
}

/// A tool result becomes a document without a model touching it, so nothing
/// can be embellished on the way in.
pub fn tool_document(run_id: &str, node: &str, command: &str, exit_code: i32, output: &str) -> String {
    let counts = parse_test_output(output);
    let head: String = output.lines().take(200).collect::<Vec<_>>().join("\n");
    let digest = if counts.conclusive {
        format!("`{command}` ran {} tests, {} failed (exit {exit_code}).", counts.total, counts.failed)
    } else {
        format!(
            "`{command}` exited {exit_code}. The output carries no test summary, so the counts below are not a verdict."
        )
    };
    format!(
        "---\nartifact: TestReport\nrun: {run_id}\nnode: {node}\nattempt: 1\nmodel: harness\ncreated: {}\ninputs: []\nresults:\n  total: {}\n  passed: {}\n  failed: {}\n  baseline_total: 0\n  baseline_passed: 0\ndigest: |\n  {digest}\n---\n\n$ {command}\n\n{head}\n",
        now_utc(),
        counts.total,
        counts.total.saturating_sub(counts.failed),
        counts.failed,
    )
}

fn describe(call: &crate::sandbox::ToolCall) -> String {
    use crate::sandbox::ToolCall::*;
    match call {
        ReadFile { path } => format!("read {}", path.display()),
        WriteFile { path, content } => format!("wrote {} ({} lines)", path.display(), content.lines().count()),
        ReplaceInFile { path, replace, .. } => {
            format!("replaced a fragment in {} ({} lines in)", path.display(), replace.lines().count())
        }
        AppendToFile { path, content } => {
            format!("appended {} lines to {}", content.lines().count(), path.display())
        }
        DeleteFile { path } => format!("deleted {}", path.display()),
        RunCommand { args, .. } => format!("ran {}", args.last().cloned().unwrap_or_default()),
        ApplyPatch { .. } => "applied a patch".into(),
        Search { query, .. } => format!("searched for {query}"),
    }
}

/// Cuts to `max` characters and says how many were dropped.
///
/// Both halves are counted the same way now. They were not: the test was in
/// bytes and the cut in characters, so any text that is not ASCII was reported
/// as shortened while being handed over whole — the step told the model a
/// number that was not true of what it was shown (finding 31).
fn truncate(s: &str, max: usize) -> String {
    let total = s.chars().count();
    if total <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}\n[... {} more characters]", total - max)
}

fn stamp(doc: &mut Document, run_id: &str, node: &str, attempt: u32, model: &str) {
    doc.header.run = run_id.to_string();
    doc.header.node = node.to_string();
    doc.header.attempt = attempt;
    doc.header.model = model.to_string();
    doc.header.created = now_utc();
    doc.header.authored_by_human = false;
}

/// `03_changes.diff` at attempt 2 becomes `03_changes.a2.diff`. Documents are
/// immutable, so a repeat never overwrites.
pub fn numbered(name: &str, attempt: u32) -> String {
    match name.rsplit_once('.') {
        Some((stem, ext)) => format!("{stem}.a{attempt}.{ext}"),
        None => format!("{name}.a{attempt}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::CharRatioTokenizer;

    #[test]
    fn a_context_is_measured_part_by_part_and_the_parts_are_the_whole() {
        let tok = CharRatioTokenizer::default();
        let parts = [
            ("system", "you are a careful engineer".to_string()),
            ("knowledge", String::new()),
            ("contract", "answer as a document".to_string()),
            ("inputs", "x".repeat(500)),
        ];
        let c = Context::build(&parts, &tok);

        assert_eq!(
            c.parts.iter().map(|(n, _)| *n).collect::<Vec<_>>(),
            vec!["system", "contract", "inputs"],
            "order is the order given, and an empty part is not recorded as zero"
        );
        assert_eq!(
            c.parts.iter().map(|(_, n)| *n).sum::<usize>(),
            c.total,
            "the parts add up to the whole"
        );
        assert_eq!(c.total, tok.count(&c.text), "and the whole is what the model will be sent");

        // Why the parts are sized by growth rather than measured alone: this
        // tokenizer rounds up, so measuring the pieces separately overshoots.
        let separately: usize = parts.iter().map(|(_, t)| tok.count(t)).sum();
        assert!(separately > c.total, "the naive sum should differ, or this test proves nothing");

        assert!(c.breakdown().starts_with("system "), "the log line names the parts: {}", c.breakdown());
        assert!(c.breakdown().ends_with(&format!("= {} tokens", c.total)));
    }

    use crate::backend::ReplayBackend;
    use crate::graph::Graph;

    fn a_role(window: u32, primary: &[&str]) -> RoleSpec {
        RoleSpec {
            name: "analyst".into(),
            slot: "reasoning".into(),
            window,
            temperature: 0.2,
            artifact: "Requirements".into(),
            system: "you are an analyst".into(),
            primary_inputs: primary.iter().map(|s| s.to_string()).collect(),
            tools: false,
            skill: None,
            max_steps: None,
            max_output: None,
            must_write: false,
            clarifies: false,
        }
    }

    fn a_document(dir: &Path, file: &str, artifact: &str, digest: &str, body: &str) {
        // Every artifact demands its own results keys, so the fixture supplies
        // the ones the parser will ask for.
        let results: &[(&str, &str)] = match artifact {
            "Requirements" => &[("requirements", "1"), ("unknowns", "0")],
            "Plan" => &[("steps", "1"), ("risks", "0"), ("verdict", "ok")],
            _ => &[],
        };
        let mut r = String::new();
        for (k, v) in results {
            r.push_str(&format!("  {k}: {v}\n"));
        }
        std::fs::write(
            dir.join(file),
            format!(
                "---\nartifact: {artifact}\nrun: r1\nnode: n\nattempt: 1\nmodel: m\ncreated: now\ninputs: []\nresults:\n{r}digest: |\n  {digest}\n---\n\n{body}\n"
            ),
        )
        .unwrap();
    }

    #[test]
    fn the_fixture_documents_parse_before_anything_is_asserted_about_them() {
        // Without this, a fixture the parser rejects would look exactly like a
        // context that shed everything, and the shedding tests would pass for
        // the wrong reason.
        let dir = tempfile::tempdir().unwrap();
        let tok = CharRatioTokenizer::default();
        for (file, artifact) in [("t.md", "Task"), ("r.md", "Requirements"), ("p.md", "Plan")] {
            a_document(dir.path(), file, artifact, "a digest of substance", "a body");
            let text = std::fs::read_to_string(dir.path().join(file)).unwrap();
            document::parse(&text, &tok).unwrap_or_else(|e| panic!("{file} does not parse: {e}"));
        }
    }

    /// An engine with nothing but a run directory: `assemble` needs no model.
    fn an_engine<'a>(
        backend: &'a ReplayBackend,
        tok: &'a dyn Tokenizer,
        dir: &Path,
    ) -> Engine<'a> {
        Engine::new(
            Graph { nodes: vec![], edges: vec![] },
            BTreeMap::new(),
            BTreeMap::new(),
            backend,
            tok,
            dir.to_path_buf(),
        )
    }

    #[test]
    fn a_context_over_budget_sheds_digests_oldest_first_and_never_the_task() {
        let dir = tempfile::tempdir().unwrap();
        let backend = ReplayBackend::new(Vec::<String>::new());
        let tok = CharRatioTokenizer::default();
        let mut e = an_engine(&backend, &tok, dir.path());

        let g = Graph { nodes: vec![], edges: vec![] };
        let mut run = RunState::new("r1", "test", &g);

        // The task is a document like any other, and by default it arrives as a
        // digest — so shedding could drop the job itself.
        a_document(dir.path(), "00_task.md", "Task", "add parse_duration and tests for it", "the task in full");
        run.documents.insert("task".into(), "00_task.md".into());

        for i in 1..=8 {
            let file = format!("{i:02}_step.md");
            a_document(dir.path(), &file, "Requirements", &"digest ".repeat(60), "body");
            run.documents.insert(format!("n{i}"), file);
        }
        // The primary input is read in full and may never be shed.
        a_document(dir.path(), "09_input.md", "Plan", "the plan digest", &"plan body ".repeat(40));
        run.documents.insert("n9".into(), "09_input.md".into());

        let role = a_role(1024, &["09_input.md"]);
        let mut notes = Vec::new();
        let ctx = e
            .assemble("a", &role, &run, 1, "m", Artifact::Requirements, 256, &mut |ev| {
                if let Event::Note(n) = ev {
                    notes.push(n);
                }
            })
            .expect("it fits once the digests are shed");

        assert!(ctx.total <= 1024 - 256, "over budget: {}", ctx.breakdown());
        assert!(ctx.text.contains("add parse_duration"), "the task was shed: {notes:?}");
        assert!(ctx.text.contains("plan body"), "a primary input was shed");

        // Oldest first: the low-numbered documents go before the high-numbered.
        let dropped: Vec<&String> = notes.iter().filter(|n| n.contains("dropping the digest")).collect();
        assert!(!dropped.is_empty(), "nothing was reported dropped: {notes:?}");
        assert!(dropped[0].contains("01_step.md"), "shedding did not start at the oldest: {dropped:?}");
        let order: Vec<usize> = dropped
            .iter()
            .filter_map(|n| n.split("dropping the digest of ").nth(1))
            .filter_map(|f| f.get(0..2).and_then(|d| d.parse().ok()))
            .collect();
        assert!(order.windows(2).all(|w| w[0] < w[1]), "not in order: {order:?}");
    }

    #[test]
    fn a_task_that_cannot_fit_is_refused_rather_than_sent() {
        let dir = tempfile::tempdir().unwrap();
        let backend = ReplayBackend::new(Vec::<String>::new());
        let tok = CharRatioTokenizer::default();
        let mut e = an_engine(&backend, &tok, dir.path());

        let g = Graph { nodes: vec![], edges: vec![] };
        let mut run = RunState::new("r1", "test", &g);
        a_document(dir.path(), "00_task.md", "Task", "small task", "the task");
        run.documents.insert("task".into(), "00_task.md".into());
        a_document(dir.path(), "01_huge.md", "Plan", "a plan", &"body ".repeat(4000));
        run.documents.insert("n1".into(), "01_huge.md".into());

        let role = a_role(2048, &["01_huge.md"]);
        let err = e
            .assemble("a", &role, &run, 1, "m", Artifact::Requirements, 512, &mut |_| {})
            .expect_err("a prompt that cannot fit must not be sent");
        let text = err.to_string();
        assert!(text.contains("does not fit in the window"), "{text}");
        assert!(text.contains("1536"), "the budget it was measured against belongs in the message: {text}");
    }

    #[test]
    fn truncate_counts_what_it_cuts_and_cuts_what_it_counts() {
        // Both halves used to be measured differently: the test in bytes, the
        // cut in characters. On text that is not ASCII the step then told the
        // model a number that was not true of the text it was shown.
        let cyrillic = "я".repeat(100); // 100 characters, 200 bytes
        let out = truncate(&cyrillic, 150);
        assert_eq!(out, cyrillic, "nothing was over the limit, so nothing may be said about it");

        let out = truncate(&cyrillic, 40);
        let (head, tail) = out.split_once('\n').expect("a cut says so on its own line");
        assert_eq!(head.chars().count(), 40, "the head is the limit, in characters");
        assert_eq!(tail, "[... 60 more characters]", "the number is what was cut");

        // ASCII keeps behaving exactly as before.
        assert_eq!(truncate("abcdef", 3), "abc\n[... 3 more characters]");
        assert_eq!(truncate("abc", 3), "abc");
    }
}
