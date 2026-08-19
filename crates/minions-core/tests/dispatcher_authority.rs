//! The dispatcher is the only thing that acts, so these tests are about what it
//! refuses to do and what it records having done.

use minions_core::dispatcher::*;
use minions_core::journal::{Effect, Journal};
use minions_core::sandbox::*;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

struct Scripted {
    answers: Mutex<Vec<GateDecision>>,
    asked: Mutex<Vec<(GateReason, bool)>>,
}

impl Scripted {
    fn new(answers: Vec<GateDecision>) -> Self {
        Self { answers: Mutex::new(answers), asked: Mutex::new(Vec::new()) }
    }
    fn asked(&self) -> Vec<(GateReason, bool)> {
        self.asked.lock().unwrap().clone()
    }
}

impl GateAuthority for Scripted {
    fn ask(&self, _call: &ToolCall, reason: GateReason, consentable: bool) -> GateDecision {
        self.asked.lock().unwrap().push((reason, consentable));
        let mut a = self.answers.lock().unwrap();
        if a.is_empty() {
            GateDecision::Reject { note: "no answer scripted".into() }
        } else {
            a.remove(0)
        }
    }
}

fn setup(answers: Vec<GateDecision>) -> (tempfile::TempDir, Scripted) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("Sources")).unwrap();
    (dir, Scripted::new(answers))
}

fn dispatcher<'a>(root: &Path, mode: PermissionMode, auth: &'a Scripted, run: &Path) -> Dispatcher<'a> {
    let j = Journal::create(run).unwrap();
    let mut d = Dispatcher::new(root, mode, vec![PathBuf::from("Sources")], auth, j).unwrap();
    d.for_node("coder", 1);
    d
}

#[test]
fn a_write_inside_the_root_is_asked_about_and_then_performed() {
    let (dir, auth) = setup(vec![GateDecision::Approve]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::AskForEverything, &auth, &run);

    let target = dir.path().join("Sources/new.swift");
    let out = d
        .dispatch(ToolCall::WriteFile { path: target.clone(), content: "code".into() }, false)
        .unwrap();

    assert_eq!(out, Outcome::Wrote);
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "code");
    assert_eq!(auth.asked().len(), 1);

    let entries = Journal::load(d.journal().path()).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].effect, Effect::Wrote);
    assert_eq!(entries[0].verdict, "Approved");
    assert_eq!(entries[0].existed_before, vec![false], "a new file must be recorded as new");
}

#[test]
fn a_rejected_gate_leaves_the_disk_untouched_and_tells_the_agent() {
    let (dir, auth) = setup(vec![GateDecision::Reject { note: "not this file".into() }]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::AskForEverything, &auth, &run);

    let target = dir.path().join("Sources/new.swift");
    let out = d
        .dispatch(ToolCall::WriteFile { path: target.clone(), content: "code".into() }, false)
        .unwrap();

    assert!(matches!(out, Outcome::Refused(ref s) if s.contains("not this file")));
    assert!(!target.exists(), "a rejected write must not reach the disk");
    let entries = Journal::load(d.journal().path()).unwrap();
    assert_eq!(entries[0].verdict, "Rejected");
}

#[test]
fn reading_outside_the_root_is_refused_without_being_asked() {
    let (dir, auth) = setup(vec![]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::DoNotAskInsideSandbox, &auth, &run);

    let out = d.dispatch(ToolCall::ReadFile { path: PathBuf::from("/etc/hosts") }, false).unwrap();
    assert!(matches!(out, Outcome::Refused(_)));
    assert!(auth.asked().is_empty(), "a forbidden read must not reach a human");
}

#[test]
fn consent_is_refused_for_floor_categories_even_when_offered() {
    let (dir, auth) = setup(vec![GateDecision::ApproveWithConsent { prefix: PathBuf::from("/") }]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::DoNotAskInsideSandbox, &auth, &run);

    let victim = dir.path().join("Sources/old.swift");
    std::fs::write(&victim, "x").unwrap();

    let out = d.dispatch(ToolCall::DeleteFile { path: victim.clone() }, false).unwrap();
    assert_eq!(out, Outcome::Deleted);
    assert!(d.consents().is_empty(), "the floor must not accept a consent");

    let (reason, consentable) = auth.asked()[0];
    assert_eq!(reason, GateReason::Delete);
    assert!(!consentable);
}

#[test]
fn granted_consent_stops_the_second_question() {
    let (dir, auth) = setup(vec![GateDecision::ApproveWithConsent { prefix: PathBuf::new() }]);
    let run = dir.path().join("run");
    let root = dir.path().canonicalize().unwrap();
    let j = Journal::create(&run).unwrap();
    let auth2 = Scripted::new(vec![GateDecision::ApproveWithConsent { prefix: root.join("Sources") }]);
    let mut d = Dispatcher::new(&root, PermissionMode::AskForEverything, vec![], &auth2, j).unwrap();
    d.for_node("coder", 1);
    let _ = &auth;

    d.dispatch(ToolCall::WriteFile { path: root.join("Sources/a.swift"), content: "a".into() }, false).unwrap();
    d.dispatch(ToolCall::WriteFile { path: root.join("Sources/b.swift"), content: "b".into() }, false).unwrap();

    assert_eq!(auth2.asked().len(), 1, "the second write should be covered by consent");
    assert!(root.join("Sources/b.swift").exists());
}

#[test]
fn an_explicit_node_gate_is_asked_even_in_the_permissive_mode() {
    let (dir, auth) = setup(vec![GateDecision::Approve]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::DoNotAskInsideSandbox, &auth, &run);

    d.dispatch(ToolCall::WriteFile { path: dir.path().join("Sources/x.swift"), content: "x".into() }, true)
        .unwrap();

    assert_eq!(auth.asked().len(), 1);
    assert_eq!(auth.asked()[0].0, GateReason::NodeGate);
}

#[test]
fn a_command_runs_in_the_project_root_with_a_constructed_environment() {
    let (dir, auth) = setup(vec![GateDecision::Approve]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::DoNotAskInsideSandbox, &auth, &run);

    let out = d
        .dispatch(ToolCall::RunCommand { program: "pwd".into(), args: vec![] }, false)
        .unwrap();

    match out {
        Outcome::Ran { exit_code, stdout, .. } => {
            assert_eq!(exit_code, 0);
            let seen = PathBuf::from(stdout.trim()).canonicalize().unwrap();
            assert_eq!(seen, dir.path().canonicalize().unwrap());
        }
        other => panic!("expected Ran, got {other:?}"),
    }
    assert_eq!(auth.asked()[0].0, GateReason::Command, "every command asks");
}

#[test]
fn patch_targets_are_taken_from_the_diff_headers() {
    let diff = "diff --git a/Sources/a.swift b/Sources/a.swift\n\
                --- a/Sources/a.swift\n\
                +++ b/Sources/a.swift\n\
                @@ -1 +1 @@\n\
                -old\n\
                +new\n\
                --- /dev/null\n\
                +++ b/Sources/b.swift\n";
    let t = patch_targets(diff);
    assert_eq!(t, vec![PathBuf::from("Sources/a.swift"), PathBuf::from("Sources/b.swift")]);
}

#[test]
fn a_patch_reaching_outside_the_root_is_refused_before_git_sees_it() {
    let (dir, auth) = setup(vec![]);
    let run = dir.path().join("run");
    let mut d = dispatcher(dir.path(), PermissionMode::DoNotAskInsideSandbox, &auth, &run);

    let diff = "--- a/x\n+++ b/../../escape.swift\n@@ -1 +1 @@\n-a\n+b\n";
    let out = d.dispatch(ToolCall::ApplyPatch { diff: diff.into() }, false).unwrap();
    assert!(matches!(out, Outcome::Refused(_)), "patch escaping the root must be refused");
}
