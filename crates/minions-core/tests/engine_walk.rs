//! The engine, driven entirely by recorded replies. No model, no network, no
//! memory pressure — so these results never vary.

use minions_core::backend::ReplayBackend;
use minions_core::graph::*;
use minions_core::run::*;
use minions_core::tokens::CharRatioTokenizer;
use std::collections::BTreeMap;

fn doc(artifact: &str, results: &[(&str, &str)], body: &str) -> String {
    let mut r = String::new();
    for (k, v) in results {
        r.push_str(&format!("  {k}: {v}\n"));
    }
    format!(
        "---\nartifact: {artifact}\nrun: x\nnode: x\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n{r}digest: |\n  A short digest of substance.\n---\n\n{body}\n"
    )
}

fn agent(id: &str, role: &str, out: &str) -> Node {
    Node {
        id: id.into(),
        kind: NodeKind::Agent,
        role: Some(role.into()),
        slot: Some("reasoning".into()),
        output: Some(out.into()),
        loop_limit: None,
        command: None,
        gate: false,
    }
}
fn plain(id: &str, kind: NodeKind) -> Node {
    Node { id: id.into(), kind, role: None, slot: None, output: None, loop_limit: None, command: None, gate: false }
}
fn edge(a: &str, b: &str) -> Edge {
    Edge { from: a.into(), to: b.into(), when: None }
}

fn roles() -> BTreeMap<String, RoleSpec> {
    let mut m = BTreeMap::new();
    for (name, artifact) in [("analyst", "Requirements"), ("reviewer", "Findings")] {
        m.insert(
            name.to_string(),
            RoleSpec {
                name: name.into(),
                slot: "reasoning".into(),
                window: 8192,
                temperature: 0.2,
                artifact: artifact.into(),
                system: format!("You are the {name}."),
                primary_inputs: vec![],
                tools: false,
                skill: None,
                max_steps: None,
                clarifies: false,
                must_write: false,
            },
        );
    }
    m
}

fn slots() -> BTreeMap<String, String> {
    BTreeMap::from([("reasoning".to_string(), "stub-model".to_string())])
}

#[tokio::test]
async fn a_linear_run_writes_documents_and_finishes() {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };
    assert_eq!(g.validate(), vec![]);

    let backend = ReplayBackend::new([doc("Requirements", &[("requirements", "3"), ("unknowns", "0")], "Body.")]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r1", "test", &g);

    let mut notes = Vec::new();
    e.run(&mut run, &mut |ev| {
        if let Event::Note(n) = ev {
            notes.push(n);
        }
    })
    .await
    .unwrap();

    assert_eq!(run.state("a"), NodeState::Done);
    assert_eq!(run.state("out"), NodeState::Done);
    assert!(run.finished.is_some());
    assert_eq!(run.documents.get("a").unwrap(), "01_req.md");
    assert!(dir.path().join("01_req.md").exists());
    assert!(notes.iter().any(|n| n.contains("wrote 01_req.md")), "{notes:?}");
}

#[tokio::test]
async fn run_json_is_written_before_the_transition_it_describes() {
    // A crash between writing and announcing must re-execute, never skip.
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };
    let backend = ReplayBackend::new([doc("Requirements", &[("requirements", "1"), ("unknowns", "0")], "B.")]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r2", "test", &g);

    let mut seen_running_on_disk = false;
    e.run(&mut run, &mut |ev| {
        if let Event::NodeState { id, state: NodeState::Running } = ev {
            if id == "a" {
                let saved = RunState::load(dir.path()).unwrap();
                seen_running_on_disk = saved.state("a") == NodeState::Running;
            }
        }
    })
    .await
    .unwrap();

    assert!(seen_running_on_disk, "the transition was announced before it was persisted");
    let saved = RunState::load(dir.path()).unwrap();
    assert_eq!(saved.state("a"), NodeState::Done);
}

#[tokio::test]
async fn a_malformed_reply_is_retried_with_guidance_then_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };
    let backend = ReplayBackend::new([
        "I think the requirements are these.".to_string(), // no header at all
        doc("Requirements", &[("requirements", "2"), ("unknowns", "0")], "Body."),
    ]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r3", "test", &g);

    e.run(&mut run, &mut |_| {}).await.unwrap();

    assert_eq!(run.state("a"), NodeState::Done);
    assert_eq!(run.attempts.get("a"), Some(&2), "the retry should be counted");
    // Attempt 2 must not overwrite attempt 1's filename.
    assert!(dir.path().join("01_req.a2.md").exists(), "a repeat attempt must write its own file");

    let prompts = backend.prompts();
    assert_eq!(prompts.len(), 2);
    assert!(prompts[1].contains("CORRECTION"), "the retry carried no guidance");
    assert!(prompts[1].contains("rejected"), "the guidance did not say what happened");
}

#[tokio::test]
async fn a_node_fails_after_three_malformed_replies() {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };
    let backend = ReplayBackend::new(["junk".to_string(), "junk".to_string(), "junk".to_string()]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r4", "test", &g);

    let err = e.run(&mut run, &mut |_| {}).await.unwrap_err();
    assert!(err.to_string().contains("no valid document"), "{err}");
    assert_eq!(run.state("a"), NodeState::Failed);
    assert_eq!(run.state("out"), NodeState::Idle, "downstream must not run");
    let saved = RunState::load(dir.path()).unwrap();
    assert!(saved.failure.is_some(), "the failure must survive on disk");
}

#[tokio::test]
async fn a_branch_routes_on_the_results_block_and_skips_the_other_way() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("rev", "reviewer", "04_findings.md"),
            plain("br", NodeKind::Branch),
            agent("fix", "analyst", "05_fix.md"),
            agent("ship", "analyst", "05_ship.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "rev"), edge("rev", "br")],
    };
    g.edges.push(Edge {
        from: "br".into(),
        to: "fix".into(),
        when: Some(Condition { document: "04_findings.md".into(), field: "breaking".into(), op: Op::Gt, value: "0".into() }),
    });
    g.edges.push(edge("br", "ship"));
    g.edges.push(edge("fix", "out"));
    g.edges.push(edge("ship", "out"));

    // No breaking findings -> the fallback (ship) is taken, fix is skipped.
    let backend = ReplayBackend::new([
        doc("Findings", &[("breaking", "0"), ("risky", "1"), ("minor", "0"), ("unmet_requirements", "0")], "Nothing serious."),
        doc("Requirements", &[("requirements", "1"), ("unknowns", "0")], "Shipping."),
    ]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r5", "test", &g);

    let mut notes = Vec::new();
    e.run(&mut run, &mut |ev| {
        if let Event::Note(n) = ev {
            notes.push(n);
        }
    })
    .await
    .unwrap();

    assert_eq!(run.state("ship"), NodeState::Done);
    assert_eq!(run.state("fix"), NodeState::Skipped, "the untaken branch must be skipped, not left idle");
    assert_eq!(run.state("out"), NodeState::Done, "the merge point must still run");
    assert!(notes.iter().any(|n| n.contains("fallback")), "{notes:?}");
}

#[tokio::test]
async fn a_branch_takes_the_conditional_way_when_the_field_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let mut g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("rev", "reviewer", "04_findings.md"),
            plain("br", NodeKind::Branch),
            agent("fix", "analyst", "05_fix.md"),
            agent("ship", "analyst", "05_ship.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "rev"), edge("rev", "br")],
    };
    g.edges.push(Edge {
        from: "br".into(),
        to: "fix".into(),
        when: Some(Condition { document: "04_findings.md".into(), field: "breaking".into(), op: Op::Gt, value: "0".into() }),
    });
    g.edges.push(edge("br", "ship"));
    g.edges.push(edge("fix", "out"));
    g.edges.push(edge("ship", "out"));

    let backend = ReplayBackend::new([
        doc("Findings", &[("breaking", "2"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "1")], "Two breakages."),
        doc("Requirements", &[("requirements", "1"), ("unknowns", "0")], "Fixing."),
    ]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r6", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    assert_eq!(run.state("fix"), NodeState::Done, "breaking > 0 must route to the fix");
    assert_eq!(run.state("ship"), NodeState::Skipped);
}

#[tokio::test]
async fn later_nodes_receive_digests_and_not_whole_bodies() {
    // The claim the whole file bus rests on: context does not grow by carrying
    // every earlier document in full.
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("a", "analyst", "01_req.md"),
            agent("b", "reviewer", "04_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "a"), edge("a", "b"), edge("b", "out")],
    };
    let long_body = "This sentence is part of a deliberately long body. ".repeat(80);
    let backend = ReplayBackend::new([
        doc("Requirements", &[("requirements", "9"), ("unknowns", "0")], &long_body),
        doc("Findings", &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")], "Fine."),
    ]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r7", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    let second = &backend.prompts()[1];
    assert!(second.contains("digests only"), "the digest section is missing");
    assert!(second.contains("A short digest of substance"), "the digest itself is missing");
    assert!(
        !second.contains(&long_body),
        "the second node received the first document in full, which is what digests exist to prevent"
    );
}

#[tokio::test]
async fn accumulated_answers_reach_the_next_run() {
    // The ladder writes knowledge.md after every clarification gate. If nothing
    // read it back, the same questions would be asked every day and the ladder
    // would be a tax rather than a safeguard.
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().to_path_buf();
    minions_core::clarify::append_knowledge(
        &project,
        &[("Which runtime?".to_string(), "Our own interpreter, written in Rust.".to_string())],
    )
    .unwrap();

    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };
    let backend = ReplayBackend::new([doc("Requirements", &[("requirements", "1"), ("unknowns", "0")], "Body.")]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, project.join("run"));
    e.project_dir = Some(project);
    let mut run = RunState::new("r-knowledge", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    let prompt = &backend.prompts()[0];
    assert!(prompt.contains("Our own interpreter"), "the accumulated answer never reached the model");
    assert!(prompt.contains("Do not ask them again"), "the model was not told these are settled");
}

/// A tool-using node has to see its own calls on the next step. Ollama renders
/// an assistant turn without them as a call to nothing, and the model then asks
/// again for what it already has — measured 2026-08-19, see `chat::wire`.
#[tokio::test]
async fn an_assistant_turn_carries_the_calls_it_made() {
    use minions_core::dispatcher::{Dispatcher, GateAuthority, GateDecision};
    use minions_core::journal::Journal;
    use minions_core::sandbox::{GateReason, PermissionMode, ToolCall};

    struct Approve;
    impl GateAuthority for Approve {
        fn ask(&self, _c: &ToolCall, _r: GateReason, _consentable: bool) -> GateDecision {
            GateDecision::Approve
        }
    }

    let project = tempfile::tempdir().unwrap();
    std::fs::write(project.path().join("a.txt"), "hello from the file\n").unwrap();
    let run_dir = project.path().join(".minions/runs/r1");

    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };
    assert_eq!(g.validate(), vec![]);

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(4);
    roles.insert("worker".into(), worker);

    let backend = ReplayBackend::new([
        "NATIVE:{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}".to_string(),
        doc(
            "Findings",
            &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
            "Nothing is wrong with a file that says hello.",
        ),
    ]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    let mut run = RunState::new("r1", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    assert_eq!(run.state("worker"), NodeState::Done);
    let prompts = backend.prompts();
    assert_eq!(prompts.len(), 2, "the node should have taken two steps");
    assert!(
        prompts[1].contains("<calls: read_file>"),
        "the second step must show the call the first step made:\n{}",
        prompts[1]
    );
    assert!(
        prompts[1].contains("hello from the file"),
        "and the result that came back for it:\n{}",
        prompts[1]
    );
}
