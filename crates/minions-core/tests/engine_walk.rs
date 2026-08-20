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
    // The sections the artifact requires (T-013). A fixture that would be
    // rejected in a run proves nothing about a run.
    let sections = match artifact {
        "Requirements" => "Statement: s.\n\nRequirements:\n1. one\n\nOut of scope:\n- nothing\n\n",
        "Plan" => "Approach: a.\n\nRejected alternatives: b.\n\nSteps:\n1. one\n\n",
        _ => "",
    };
    format!(
        "---\nartifact: {artifact}\nrun: x\nnode: x\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n{r}digest: |\n  A short digest of substance.\n---\n\n{sections}{body}\n"
    )
}

/// A document built without the helper's sections, for the tests that are about
/// the sections themselves.
fn raw_doc(artifact: &str, results: &[(&str, &str)], body: &str) -> String {
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
        artifact: None,
        requires_success: false,
        gate: false,
    }
}
fn plain(id: &str, kind: NodeKind) -> Node {
    Node {
        id: id.into(),
        kind,
        role: None,
        slot: None,
        output: None,
        loop_limit: None,
        command: None,
        artifact: None,
        requires_success: false,
        gate: false,
    }
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
                max_output: None,
                clarifies: false,
                must_write: false,
                requires_green: false,
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

#[tokio::test]
async fn the_run_log_says_how_big_the_prompt_was_and_of_what() {
    // SDD §7 forbids "it will probably fit". Until T-003 the engine sent the
    // prompt without ever counting it, so an overflow could only be guessed at
    // from how the model behaved.
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };
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

    let line = notes
        .iter()
        .find(|n| n.starts_with("a: context "))
        .unwrap_or_else(|| panic!("no context measurement in the log: {notes:?}"));

    // The line has to carry the parts, a total and the budget it is measured
    // against — a number with nothing to compare it to says nothing.
    assert!(line.contains("system "), "{line}");
    assert!(line.contains("contract "), "{line}");
    assert!(line.contains(" tokens of a "), "{line}");
    assert!(line.ends_with(" token budget"), "{line}");

    let total: usize = line
        .split_once("= ")
        .and_then(|(_, t)| t.split_once(" tokens"))
        .and_then(|(n, _)| n.parse().ok())
        .unwrap_or_else(|| panic!("no total in {line}"));
    assert!(total > 0, "an empty measurement is not a measurement: {line}");
}

#[tokio::test]
async fn a_cut_off_answer_is_named_as_one_and_the_role_says_how_much_room_it_needs() {
    // The failure this prevents: a reply that ran out of num_predict parses as
    // nothing, the node reports "no valid document", and the correction it
    // sends back is advice about the format — which was never the problem.
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };

    let backend = ReplayBackend::new([
        // What a cut-off answer looks like: the beginning of a document.
        "CUT:---\nartifact: Requirements\nrun: x\nnode: x\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n  requi".to_string(),
        doc("Requirements", &[("requirements", "3"), ("unknowns", "0")], "Body."),
    ]);
    let tok = CharRatioTokenizer::default();

    let mut roles = roles();
    roles.get_mut("analyst").unwrap().max_output = Some(64);

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r1", "test", &g);

    let mut notes = Vec::new();
    e.run(&mut run, &mut |ev| {
        if let Event::Note(n) = ev {
            notes.push(n);
        }
    })
    .await
    .unwrap();

    assert!(
        notes.iter().any(|n| n.contains("cut off at 64 tokens")),
        "the cut-off was not named: {notes:?}"
    );
    assert!(
        !notes.iter().any(|n| n.contains("rejected")),
        "the model was blamed for the format instead: {notes:?}"
    );
    assert_eq!(run.state("a"), NodeState::Done, "the next attempt must still be allowed to succeed");

    // The role's number is what was actually asked for, both times.
    assert_eq!(backend.predicts(), vec![64, 64], "num_predict did not come from the role");
}

/// The harness measures the baseline with one command and, until T-009, kept it
/// to itself. In the live run of 2026-08-19 the coder spent six of its fourteen
/// steps on `python -m unittest discover tests` — exit 127 every time, on a
/// machine that has `python3` and no `python` — while the harness held the
/// working command the whole time (finding 37).
#[tokio::test]
async fn a_tool_role_is_told_the_command_the_harness_measures_with() {
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
    let run_dir = project.path().join(".minions/runs/r1");
    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(4);
    roles.insert("worker".into(), worker);

    let backend = ReplayBackend::new([doc(
        "Findings",
        &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
        "Nothing to change.",
    )]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    // The shape mrun stores: the harness's own shell wrapper around the command.
    e.test_command =
        Some(vec!["bash".into(), "-lc".into(), "python3 -m unittest discover -s tests".into()]);
    let mut run = RunState::new("r1", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    let first = &backend.prompts()[0];
    assert!(
        first.contains("python3 -m unittest discover -s tests"),
        "the role was never told the command the harness measures with:\n{first}"
    );
    assert!(
        !first.contains("bash -lc"),
        "the agent was shown the harness's own shell wrapper, which run_command adds itself:\n{first}"
    );
}

/// Six identical failures in a row is how the run of 2026-08-19 burned its
/// steps. A repeated call that failed the same way will not start working, and
/// the harness is the only party that can see the repetition.
#[tokio::test]
async fn a_call_that_keeps_failing_the_same_way_is_named_out_loud() {
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
    let run_dir = project.path().join(".minions/runs/r1");
    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(6);
    roles.insert("worker".into(), worker);

    // A command that is missing on every machine, so the failure is the run's
    // own and not the operating system's mood: exit 127, three times running.
    let wrong = "NATIVE:{\"name\":\"run_command\",\"arguments\":{\"command\":\"unittest-runner-that-does-not-exist -s tests\"}}"
        .to_string();
    let backend = ReplayBackend::new([
        wrong.clone(),
        wrong.clone(),
        wrong.clone(),
        doc(
            "Findings",
            &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
            "Gave up on that command.",
        ),
    ]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    e.test_command =
        Some(vec!["bash".into(), "-lc".into(), "python3 -m unittest discover -s tests".into()]);
    let mut run = RunState::new("r1", "test", &g);

    let mut notes = Vec::new();
    e.run(&mut run, &mut |ev| {
        if let Event::Note(n) = ev {
            notes.push(n);
        }
    })
    .await
    .unwrap();

    assert_eq!(run.state("worker"), NodeState::Done);
    let prompts = backend.prompts();
    assert_eq!(prompts.len(), 4, "the node should have taken four steps");
    assert!(
        !prompts[1].contains("same call has now failed"),
        "one failure is not a repetition:\n{}",
        prompts[1]
    );
    assert!(
        prompts[3].contains("same call has now failed 3 times"),
        "the third identical failure was not put to the model:\n{}",
        prompts[3]
    );
    assert!(
        notes.iter().any(|n| n.contains("same call has now failed 3 times")),
        "and it was not said out loud in the log: {notes:?}"
    );
}

/// Finding 2, the half the earlier check could not see. A run that deletes
/// tests is caught by counting them; a run that leaves all six in place and
/// turns three of them red is not, and it is the more likely of the two.
#[tokio::test]
async fn a_node_that_breaks_the_tests_without_deleting_any_is_refused() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    // A suite whose verdict depends on a file the agent is about to write: six
    // tests either way, three of them failing once `broken` is there.
    let suite = "if [ -f broken ]; then printf 'Ran 6 tests\\n\\nFAILED (failures=3)\\n'; \
                 else printf 'Ran 6 tests\\n\\nOK\\n'; fi";

    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(2);
    roles.insert("worker".into(), worker);

    let backend = ReplayBackend::new([
        "NATIVE:{\"name\":\"write_file\",\"arguments\":{\"path\":\"broken\",\"content\":\"x\"}}".to_string(),
        doc(
            "Findings",
            &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
            "Everything is in order.",
        ),
    ]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    e.test_command = Some(vec!["bash".into(), "-lc".into(), suite.into()]);

    let mut run = RunState::new("r1", "test", &g);
    e.take_baseline(&mut run, &mut |_| {});
    assert_eq!(
        run.baseline.map(|b| (b.total, b.failed)),
        Some((6, 0)),
        "the baseline itself has to be green, or the test proves nothing"
    );

    let mut notes = Vec::new();
    let err = e
        .run(&mut run, &mut |ev| {
            if let Event::Note(n) = ev {
                notes.push(n);
            }
        })
        .await
        .unwrap_err();

    assert_eq!(
        run.state("worker"),
        NodeState::Failed,
        "a document written over three broken tests must not close the node: {notes:?}"
    );
    assert!(err.to_string().contains("used all 2 steps"), "{err}");
    assert!(
        notes.iter().any(|n| n.contains("0 failing before, 3 now")),
        "the refusal never named what was broken: {notes:?}"
    );
}

/// A tool node takes steps, not attempts. Until T-012 the step counter was
/// written into both `run.attempts` and the document header, so `03_changes.md`
/// of the run of 16.08 carried `attempt: 9` for a node that had been asked
/// once, and the report counted it as a repeat.
#[tokio::test]
async fn a_tool_node_counts_its_steps_as_steps_and_its_attempt_as_one() {
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
    std::fs::write(project.path().join("a.txt"), "first file\n").unwrap();
    std::fs::write(project.path().join("b.txt"), "second file\n").unwrap();
    let run_dir = project.path().join(".minions/runs/r1");

    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(6);
    roles.insert("worker".into(), worker);

    let backend = ReplayBackend::new([
        "NATIVE:{\"name\":\"read_file\",\"arguments\":{\"path\":\"a.txt\"}}".to_string(),
        "NATIVE:{\"name\":\"read_file\",\"arguments\":{\"path\":\"b.txt\"}}".to_string(),
        doc(
            "Findings",
            &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
            "Two files, nothing wrong with either.",
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

    assert_eq!(
        run.attempts.get("worker"),
        Some(&1),
        "the node was asked once and took three steps; only the asking is an attempt"
    );

    let document = std::fs::read_to_string(run_dir.join("01_findings.md")).unwrap();
    assert!(document.contains("attempt: 1"), "the header carries the step count as an attempt:\n{document}");
    assert!(
        document.contains("tool_steps: 3"),
        "the steps went nowhere, so nothing says how much work the node took:\n{document}"
    );

    let report = std::fs::read_to_string(run_dir.join("report.md")).unwrap();
    assert!(
        report.contains("took more than one attempt: 0"),
        "the report counts steps as repeats:\n{report}"
    );
}

/// Documents are immutable, so a node on its second attempt writes its own
/// file. The single-answer path has always done this; the tool path wrote over
/// the first document, and nothing said so.
#[tokio::test]
async fn a_tool_node_on_a_second_attempt_writes_its_own_file() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(3);
    roles.insert("worker".into(), worker);

    let backend = ReplayBackend::new([doc(
        "Findings",
        &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
        "Second time around.",
    )]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    let mut run = RunState::new("r1", "test", &g);
    // One attempt already spent — what an outer retry loop will leave behind.
    run.attempts.insert("worker".to_string(), 1);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    assert_eq!(run.attempts.get("worker"), Some(&2));
    assert!(
        run_dir.join("01_findings.a2.md").exists(),
        "a repeat attempt must write its own file"
    );
    assert!(
        !run_dir.join("01_findings.md").exists(),
        "and must not stand where the first attempt's document would be"
    );
    assert_eq!(run.documents.get("worker").unwrap(), "01_findings.a2.md");
}

/// The body used to be checked for one thing: that it was not empty. The next
/// node reads sections out of it, so a document that skips one is not a
/// document — it is a document-shaped reply.
#[tokio::test]
async fn a_document_that_skips_a_required_section_is_sent_back() {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), agent("a", "analyst", "01_req.md"), plain("out", NodeKind::Output)],
        edges: vec![edge("in", "a"), edge("a", "out")],
    };

    // Statement and Requirements, and then it stops: no "Out of scope", which
    // is the section the architect downstream reads to know what not to do.
    let half = raw_doc(
        "Requirements",
        &[("requirements", "2"), ("unknowns", "0")],
        "Statement: parse durations.\n\nRequirements:\n1. Parse hours.\n2. Parse minutes.",
    );
    let whole = raw_doc(
        "Requirements",
        &[("requirements", "2"), ("unknowns", "0")],
        "Statement: parse durations.\n\nRequirements:\n1. Parse hours.\n2. Parse minutes.\n\nOut of scope:\n- No user interface.",
    );

    let backend = ReplayBackend::new([half, whole]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r1", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    assert_eq!(run.state("a"), NodeState::Done);
    assert_eq!(run.attempts.get("a"), Some(&2), "the half document was accepted");

    let prompts = backend.prompts();
    assert!(
        prompts[1].contains("Out of scope"),
        "the correction did not name the section that was missing:\n{}",
        prompts[1]
    );
}

/// The reviewer of the run 2026-08-16T18-41-51 answered with its own header as
/// JSON. Everything checked out — the header was perfect and the body was not
/// empty — so `results.breaking: 1` went into the report as a finding nobody
/// had made.
#[tokio::test]
async fn a_body_that_is_the_header_said_again_is_sent_back() {
    let dir = tempfile::tempdir().unwrap();
    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("rev", "reviewer", "04_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "rev"), edge("rev", "out")],
    };

    let numbers = [("breaking", "1"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")];
    let echo = raw_doc(
        "Findings",
        &numbers,
        "{\n  \"artifact\": \"Findings\",\n  \"node\": \"reviewer\",\n  \"results\": {\n    \"breaking\": 1\n  },\n  \"digest\": \"One breaking issue found in src/durations.py.\"\n}",
    );
    let real = raw_doc(
        "Findings",
        &numbers,
        "- `src/durations.py`, `parse_duration`: optional groups make the regex match anything.\n  - Failure scenario: `parse_duration(\"abc\")` returns 0 instead of raising.\n  - Severity: breaking",
    );

    let backend = ReplayBackend::new([echo, real]);
    let tok = CharRatioTokenizer::default();
    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, dir.path().to_path_buf());
    let mut run = RunState::new("r1", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    assert_eq!(run.state("rev"), NodeState::Done);
    assert_eq!(run.attempts.get("rev"), Some(&2), "the header said twice was accepted as findings");

    let prompts = backend.prompts();
    assert!(
        prompts[1].contains("Do not repeat the header"),
        "the correction did not say what was wrong with it:\n{}",
        prompts[1]
    );
}

/// A tool node that ran something which is not a test suite used to produce a
/// `TestReport` full of zeros, and a branch reading `failed` off it took the
/// green way. `00_context.md` of the run 2026-08-16T18-41-51 is that document:
/// a file dump, recorded as "0 failed".
#[tokio::test]
async fn a_branch_on_a_report_that_measured_nothing_stops_the_run() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    let tests = Node {
        id: "tests".into(),
        kind: NodeKind::Tool,
        role: None,
        slot: None,
        output: Some("03_tests.md".into()),
        loop_limit: None,
        // Not a test runner: no summary in the output, so nothing was measured.
        command: Some(vec!["bash".into(), "-lc".into(), "echo listing the files".into()]),
        artifact: None,
        requires_success: false,
        gate: false,
    };
    let mut g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            tests,
            plain("br", NodeKind::Branch),
            agent("fix", "analyst", "05_fix.md"),
            agent("ship", "analyst", "05_ship.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "tests"), edge("tests", "br")],
    };
    g.edges.push(Edge {
        from: "br".into(),
        to: "fix".into(),
        when: Some(Condition {
            document: "03_tests.md".into(),
            field: "failed".into(),
            op: Op::Gt,
            value: "0".into(),
        }),
    });
    g.edges.push(edge("br", "ship"));
    g.edges.push(edge("fix", "out"));
    g.edges.push(edge("ship", "out"));
    assert_eq!(g.validate(), vec![]);

    let backend = ReplayBackend::new([doc("Requirements", &[("requirements", "1"), ("unknowns", "0")], "Shipping.")]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    let mut run = RunState::new("r1", "test", &g);

    let err = e.run(&mut run, &mut |_| {}).await.unwrap_err();
    assert!(
        err.to_string().contains("measured no tests") || err.to_string().contains("conclusive"),
        "the run went on as if the tests had passed: {err}"
    );
    assert_eq!(run.state("ship"), NodeState::Idle, "the green way must not be taken on a report that measured nothing");

    let report = std::fs::read_to_string(run_dir.join("03_tests.md")).unwrap();
    assert!(report.contains("conclusive: no"), "the report does not say it measured nothing:\n{report}");
    assert!(
        !report.contains("failed: 0"),
        "a number that was never measured is written as zero, which is what a branch reads:\n{report}"
    );
}

/// "When the work is done and the tests pass, reply with your document" is a
/// sentence in a system prompt — a request. A role that declares
/// `requires_green` cannot close while the harness's own measurement shows
/// failures, and the refusal carries the output rather than a number.
#[tokio::test]
async fn a_node_that_requires_green_cannot_close_on_red_tests() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    // Red until the agent writes `fixed`, green after: the suite is the thing
    // that decides, and the agent has to change the tree to change its mind.
    let suite = "if [ -f fixed ]; then printf 'Ran 3 tests\\n\\nOK\\n'; \
                 else printf 'F..\\nRan 3 tests\\n\\nFAILED (failures=1)\\n'; fi";

    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(5);
    worker.requires_green = true;
    roles.insert("worker".into(), worker);

    let finished = doc(
        "Findings",
        &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
        "All done.",
    );
    let backend = ReplayBackend::new([
        finished.clone(),
        "NATIVE:{\"name\":\"write_file\",\"arguments\":{\"path\":\"fixed\",\"content\":\"x\"}}".to_string(),
        finished,
    ]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    e.test_command = Some(vec!["bash".into(), "-lc".into(), suite.into()]);
    let mut run = RunState::new("r1", "test", &g);

    let mut notes = Vec::new();
    e.run(&mut run, &mut |ev| {
        if let Event::Note(n) = ev {
            notes.push(n);
        }
    })
    .await
    .unwrap();

    assert_eq!(run.state("worker"), NodeState::Done);
    let prompts = backend.prompts();
    assert_eq!(prompts.len(), 3, "the first document was accepted over a failing suite: {notes:?}");
    assert!(
        prompts[1].contains("1 of 3"),
        "the refusal did not say how many tests are failing:\n{}",
        prompts[1]
    );
    assert!(
        prompts[1].contains("FAILED (failures=1)"),
        "the refusal did not carry what the tests actually printed:\n{}",
        prompts[1]
    );

    let document = std::fs::read_to_string(run_dir.join("01_findings.md")).unwrap();
    assert!(document.contains("tool_steps: 3"), "the document should be the one written on the third step");
}

/// Until loops iterate, a red suite has to stop the run rather than flow into a
/// review of broken code. The tool node that ran the tests says so itself.
#[tokio::test]
async fn a_tool_node_that_must_succeed_stops_the_run_when_it_does_not() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    let tests = Node {
        id: "tests".into(),
        kind: NodeKind::Tool,
        role: None,
        slot: None,
        output: Some("04_test-report.md".into()),
        loop_limit: None,
        command: Some(vec![
            "bash".into(),
            "-lc".into(),
            "printf 'F..\\nRan 3 tests\\n\\nFAILED (failures=1)\\n'; exit 1".into(),
        ]),
        artifact: None,
        requires_success: true,
        gate: false,
    };
    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            tests,
            agent("rev", "reviewer", "05_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "tests"), edge("tests", "rev"), edge("rev", "out")],
    };
    assert_eq!(g.validate(), vec![]);

    let backend = ReplayBackend::new([doc(
        "Findings",
        &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
        "Nothing to report.",
    )]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    let mut run = RunState::new("r1", "test", &g);

    let err = e.run(&mut run, &mut |_| {}).await.unwrap_err();
    assert!(err.to_string().contains("tests"), "the failure does not name the node: {err}");
    assert!(err.to_string().contains("exit 1"), "the failure does not say what the command did: {err}");
    assert_eq!(run.state("rev"), NodeState::Idle, "the review ran over a red suite");

    // The report is written all the same: what happened is recorded before the
    // run stops on it.
    let report = std::fs::read_to_string(run_dir.join("04_test-report.md")).unwrap();
    assert!(report.contains("conclusive: yes"), "{report}");
    assert!(report.contains("failed: 1"), "{report}");
}

/// A tool node that lists files is not a test report. Every one of them was
/// called `TestReport` until the kind could be declared (finding 26).
#[tokio::test]
async fn a_tool_node_says_what_kind_of_document_it_writes() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    let read = Node {
        id: "read".into(),
        kind: NodeKind::Tool,
        role: None,
        slot: None,
        output: Some("00_context.md".into()),
        loop_limit: None,
        command: Some(vec!["bash".into(), "-lc".into(), "echo 'the files of the project'".into()]),
        artifact: Some("Report".into()),
        requires_success: false,
        gate: false,
    };
    let g = Graph {
        nodes: vec![plain("in", NodeKind::Input), read, plain("out", NodeKind::Output)],
        edges: vec![edge("in", "read"), edge("read", "out")],
    };
    assert_eq!(g.validate(), vec![]);

    let backend = ReplayBackend::new([]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles(), slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    let mut run = RunState::new("r1", "test", &g);
    e.run(&mut run, &mut |_| {}).await.unwrap();

    let document = std::fs::read_to_string(run_dir.join("00_context.md")).unwrap();
    assert!(document.contains("artifact: Report"), "a file listing is still called a test report:\n{document}");
    assert!(!document.contains("conclusive"), "a listing has nothing to be conclusive about:\n{document}");
}

/// SPEC §5.11 asks for the movement, not a number: "6 of 6 passing before, 5 of
/// 6 now" is a fact about the run, "5 tests pass" is a fact about a moment. The
/// report used to say neither.
#[tokio::test]
async fn the_report_says_where_the_tests_were_and_where_they_are() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    // One failure until something writes `fixed`, green after: the movement
    // this test is about is the one a run is supposed to make.
    let suite = "if [ -f fixed ]; then printf '...\\nRan 3 tests\\n\\nOK\\n'; \
                 else printf 'F..\\nRan 3 tests\\n\\nFAILED (failures=1)\\n'; fi";

    let tests = Node {
        id: "tests".into(),
        kind: NodeKind::Tool,
        role: None,
        slot: None,
        output: Some("04_test-report.md".into()),
        loop_limit: None,
        command: Some(vec!["bash".into(), "-lc".into(), suite.into()]),
        artifact: None,
        requires_success: false,
        gate: false,
    };
    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            tests,
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "tests"), edge("tests", "out")],
    };
    assert_eq!(g.validate(), vec![]);

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    worker.max_steps = Some(4);
    roles.insert("worker".into(), worker);

    let backend = ReplayBackend::new([
        "NATIVE:{\"name\":\"write_file\",\"arguments\":{\"path\":\"fixed\",\"content\":\"x\"}}".to_string(),
        doc(
            "Findings",
            &[("breaking", "0"), ("risky", "0"), ("minor", "0"), ("unmet_requirements", "0")],
            "The failing test passes now.",
        ),
    ]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    e.test_command = Some(vec!["bash".into(), "-lc".into(), suite.into()]);
    let mut run = RunState::new("r1", "test", &g);
    e.take_baseline(&mut run, &mut |_| {});
    assert_eq!(run.baseline.map(|b| (b.total, b.failed)), Some((3, 1)), "the run has to start from a red suite");

    e.run(&mut run, &mut |_| {}).await.unwrap();

    let report = std::fs::read_to_string(run_dir.join("report.md")).unwrap();
    assert!(
        report.contains("was 2/3 passing, now 3/3 passing"),
        "the report does not say where the tests moved:\n{report}"
    );

    // The report the harness wrote carries the baseline it was measured against,
    // so the document itself says what the movement was.
    let test_report = std::fs::read_to_string(run_dir.join("04_test-report.md")).unwrap();
    assert!(test_report.contains("baseline_total: 3"), "{test_report}");
    assert!(test_report.contains("baseline_failed: 1"), "{test_report}");
}

/// Finding 40, measured on the run of 2026-08-20 evening: the harness named the
/// repetition sixteen times, the model went on repeating, and the node spent
/// its fourteen steps and fifteen minutes on it. Naming is not acting.
#[tokio::test]
async fn a_node_stops_when_the_same_call_keeps_giving_the_same_failure() {
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
    let run_dir = project.path().join(".minions/runs/r1");

    let g = Graph {
        nodes: vec![
            plain("in", NodeKind::Input),
            agent("worker", "worker", "01_findings.md"),
            plain("out", NodeKind::Output),
        ],
        edges: vec![edge("in", "worker"), edge("worker", "out")],
    };

    let mut roles = roles();
    let mut worker = roles.get("reviewer").cloned().unwrap();
    worker.name = "worker".into();
    worker.tools = true;
    // Fourteen steps were available in the live run; the node must not need
    // them to work out that nothing is happening.
    worker.max_steps = Some(14);
    roles.insert("worker".into(), worker);

    // The same command, the same failure, every time — a suite that fails to
    // import, which is exactly what the live run kept re-running.
    let same = "NATIVE:{\"name\":\"run_command\",\"arguments\":{\"command\":\"printf 'ImportError: no module named durations\\\\n' >&2; exit 1\"}}"
        .to_string();
    let backend = ReplayBackend::new(vec![same; 8]);
    let tok = CharRatioTokenizer::default();
    let authority = Approve;
    let journal = Journal::create(&run_dir).unwrap();
    let dispatcher =
        Dispatcher::new(project.path(), PermissionMode::AskForEverything, vec![], &authority, journal).unwrap();

    let mut e = Engine::new(g.clone(), roles, slots(), &backend, &tok, run_dir.clone());
    e.dispatcher = Some(dispatcher);
    let mut run = RunState::new("r1", "test", &g);

    let err = e.run(&mut run, &mut |_| {}).await.unwrap_err();
    assert!(
        err.to_string().contains("same") && err.to_string().contains("worker"),
        "the node did not say why it stopped: {err}"
    );
    assert_eq!(run.state("worker"), NodeState::Failed);
    assert!(
        backend.remaining() >= 4,
        "the node spent {} of the 8 scripted steps on a call that never changed",
        8 - backend.remaining()
    );
}
