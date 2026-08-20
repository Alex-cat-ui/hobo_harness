//! Runs a workflow against live models, with no interface. The honest
//! end-to-end check: everything the product does except draw itself.

use anyhow::{Context, Result};
use minions_core::checkpoint;
use minions_core::clarify::ClarificationAuthority;
use minions_core::dispatcher::{Dispatcher, GateAuthority, GateDecision};
use minions_core::graph::Graph;
use minions_core::journal::Journal;
use minions_core::ollama::Ollama;
use minions_core::run::{Engine, Event, RoleSpec, RunState};
use minions_core::sandbox::{GateReason, PermissionMode, ToolCall};
use minions_core::tokens::CharRatioTokenizer;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
struct Workflow {
    name: String,
    graph: Graph,
    roles: BTreeMap<String, RoleSpec>,
}

/// Approves from the terminal. The product asks in the interface; this asks here.
struct Terminal;

impl GateAuthority for Terminal {
    fn ask(&self, call: &ToolCall, reason: GateReason, consentable: bool) -> GateDecision {
        let what = match call {
            ToolCall::RunCommand { program, args } => format!("run `{program} {}`", args.join(" ")),
            ToolCall::WriteFile { path, .. } => format!("write {}", path.display()),
            ToolCall::DeleteFile { path } => format!("delete {}", path.display()),
            ToolCall::ApplyPatch { .. } => "apply a patch".to_string(),
            other => format!("{other:?}"),
        };
        eprintln!("\n  GATE  {what}\n        reason: {reason:?}, consentable: {consentable}");
        eprint!("        approve? [y/N] ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_ok() && line.trim().eq_ignore_ascii_case("y") {
            GateDecision::Approve
        } else {
            GateDecision::Reject { note: "declined at the terminal".into() }
        }
    }
}

/// Used with --yes so an unattended run does not stall. It approves ordinary
/// work and the workflow's own checkpoints, and refuses the floor.
///
/// A node gate is not a floor category: it is a stop the workflow author put
/// there for a human, and it is marked not-consentable so that no scoped
/// consent can silently swallow it. Treating "not consentable" as "on the
/// floor" made every gated node fail under --yes while the log claimed the
/// files had been written.
struct AutoWithinFloor;

impl GateAuthority for AutoWithinFloor {
    fn ask(&self, _call: &ToolCall, reason: GateReason, consentable: bool) -> GateDecision {
        if reason == GateReason::NodeGate {
            eprintln!("      (--yes: passing the workflow's own gate)");
            return GateDecision::Approve;
        }
        if consentable {
            GateDecision::Approve
        } else {
            GateDecision::Reject { note: format!("{reason:?} is on the floor and needs a human") }
        }
    }
}

/// Asks the human at the terminal. In the product this is a gate in the
/// interface; here it is stdin.
struct TerminalClarifier {
    auto: bool,
}

impl ClarificationAuthority for TerminalClarifier {
    fn ask(&self, node: &str, questions: &[String]) -> Vec<String> {
        eprintln!("\n  ─── {node} needs answers before it can go on ───");
        for (i, q) in questions.iter().enumerate() {
            eprintln!("  {}. {q}", i + 1);
        }
        if self.auto {
            eprintln!("  (--yes: nobody is here to answer, so the unknowns stay recorded)");
            return vec![String::new(); questions.len()];
        }
        let mut answers = Vec::new();
        for (i, q) in questions.iter().enumerate() {
            eprintln!("\n  {}. {q}", i + 1);
            eprint!("     > ");
            let _ = std::io::stderr().flush();
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            answers.push(line.trim().to_string());
        }
        answers
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let auto = args.iter().any(|a| a == "--yes");
    let task = args
        .iter()
        .position(|a| a == "--task")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "Describe this project.".to_string());
    let positional: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| {
            !a.starts_with("--")
                && !args.get(i.wrapping_sub(1)).map(|p| p == "--task").unwrap_or(false)
        })
        .map(|(_, a)| a)
        .collect();
    let wf_path = positional.first().map(|s| s.as_str()).unwrap_or("workflows/analysis.json");
    let project = positional
        .get(1)
        .map(|s| PathBuf::from(s.as_str()))
        .unwrap_or_else(|| std::env::current_dir().expect("cwd"));

    let text = std::fs::read_to_string(wf_path).with_context(|| format!("reading {wf_path}"))?;
    let wf: Workflow = serde_json::from_str(&text).context("parsing the workflow")?;

    let errors = wf.graph.validate();
    if !errors.is_empty() {
        eprintln!("the workflow does not validate:");
        for e in &errors {
            eprintln!("  - {e}");
        }
        std::process::exit(2);
    }

    let ollama = Ollama::local()?;
    if !ollama.reachable().await {
        anyhow::bail!("ollama is not reachable — run tools/ollama.sh up");
    }

    // The coding slot is bound to the general model, not the coding-tuned one.
    // Measured, not assumed: qwen2.5-coder:14b wrote no file in three runs and
    // forty steps, never used the tool channel and never held the document
    // format, while qwen2.5:14b did all three (STATUS §6). Its coding tuning
    // appears to have cost it agentic behaviour, and this harness is agentic.
    let slots = BTreeMap::from([
        ("coding".to_string(), "qwen2.5:14b".to_string()),
        ("reasoning".to_string(), "qwen2.5:14b".to_string()),
        ("chat".to_string(), "qwen2.5:7b".to_string()),
        ("embedding".to_string(), "nomic-embed-text".to_string()),
    ]);

    let stamp = std::process::Command::new("date").args(["-u", "+%Y-%m-%dT%H-%M-%S"]).output()?;
    let run_id = format!("{}_{}", String::from_utf8_lossy(&stamp.stdout).trim(), wf.name.to_lowercase().replace(' ', "-"));
    let run_dir = project.join(".minions/runs").join(&run_id);

    println!("workflow : {}", wf.name);
    println!("project  : {}", project.display());
    println!("run      : {run_id}");
    println!("gates    : {}\n", if auto { "auto within the floor" } else { "terminal" });

    let tok = CharRatioTokenizer::default();
    let journal = Journal::create(&run_dir)?;
    let terminal = Terminal;
    let automatic = AutoWithinFloor;
    let authority: &dyn GateAuthority = if auto { &automatic } else { &terminal };
    let dispatcher = Dispatcher::new(&project, PermissionMode::AskForEverything, vec![], authority, journal)?;

    let mut engine = Engine::new(wf.graph.clone(), wf.roles, slots, &ollama, &tok, run_dir.clone());
    engine.dispatcher = Some(dispatcher);
    let skills = std::env::current_dir()?.join("skills");
    if skills.is_dir() {
        engine.skills_dir = Some(skills);
    }
    engine.project_dir = Some(project.clone());
    // Detected for now from the workflow's own test node; the product detects
    // it when a project is opened.
    engine.test_command = Some(vec![
        "bash".to_string(),
        "-lc".to_string(),
        "set -o pipefail; python3 -m unittest discover -s tests 2>&1 | tail -25".to_string(),
    ]);
    let clarifier = TerminalClarifier { auto };
    engine.clarifier = Some(&clarifier);

    let mut run = RunState::new(&run_id, &wf.name, &wf.graph);

    // Before anything can write. A run whose every effect is reversible in one
    // action carries a very different risk from one whose effects are
    // permanent, and gates alone protect against a bad action, not a bad run.
    if checkpoint::is_repository(&project) {
        let cp = checkpoint::snapshot(&project, &run_id).context("taking a checkpoint")?;
        println!("checkpoint: {} ({})", &cp.commit[..12], cp.reference);
        run.checkpoint = Some(cp.commit.clone());
    } else {
        println!("checkpoint: NONE — this folder is not a git repository, so this run cannot be rolled back");
    }

    // The task is the run's first document. Everything downstream sees it the
    // same way it sees any other: in full if declared, as a digest otherwise.
    std::fs::create_dir_all(&run_dir)?;
    let digest: String = task.chars().take(300).collect();
    std::fs::write(
        run_dir.join("00_task.md"),
        format!(
            "---\nartifact: Task\nrun: {run_id}\nnode: input\nattempt: 1\nmodel: human\ncreated: now\ninputs: []\nresults:\ndigest: |\n  {digest}\n---\n\n{task}\n"
        ),
    )?;
    run.documents.insert("task".to_string(), "00_task.md".to_string());
    println!("task     : {task}\n");
    // Before any agent acts, which is the only moment it means anything.
    engine.take_baseline(&mut run, &mut |ev| {
        if let Event::Note(n) = ev {
            println!("  · {n}");
        }
    });

    let started = std::time::Instant::now();
    let mut streamed = 0usize;

    let result = engine
        .run(&mut run, &mut |ev| match ev {
            Event::Note(n) => println!("\n  · {n}"),
            Event::NodeState { id, state } => println!("\n[{id}] {state:?}"),
            Event::Token(t) => {
                streamed += t.len();
                print!("{t}");
                let _ = std::io::stdout().flush();
            }
        })
        .await;

    println!("\n\n=== summary ===");
    println!("elapsed   : {:.1}s", started.elapsed().as_secs_f64());
    println!("streamed  : {streamed} bytes");
    println!("documents : {}", run.documents.len());
    for (node, file) in &run.documents {
        println!("  {node:<10} {file}");
    }
    match result {
        Ok(()) => println!("report    : {}", run_dir.join("report.md").display()),
        Err(e) => {
            println!("FAILED    : {e}");
            std::process::exit(1);
        }
    }
    Ok(())
}
