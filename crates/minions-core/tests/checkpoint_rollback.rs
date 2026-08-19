//! The checkpoint must be invisible to the user's git, and the rollback must be
//! incapable of touching work the run never touched.

use minions_core::checkpoint::*;
use std::path::{Path, PathBuf};
use std::process::Command;

fn git(root: &Path, args: &[&str]) -> String {
    let out = Command::new("git").args(args).current_dir(root).output().unwrap();
    assert!(out.status.success(), "git {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

fn repo() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    git(root, &["init", "-q"]);
    git(root, &["config", "user.email", "t@example.com"]);
    git(root, &["config", "user.name", "t"]);
    std::fs::write(root.join("tracked.txt"), "original\n").unwrap();
    std::fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
    git(root, &["add", "-A"]);
    git(root, &["-c", "commit.gpgsign=false", "commit", "-q", "-m", "base"]);
    dir
}

fn state(root: &Path) -> (String, String, String, String) {
    (
        git(root, &["rev-parse", "HEAD"]),
        git(root, &["rev-parse", "--abbrev-ref", "HEAD"]),
        git(root, &["status", "--porcelain"]),
        git(root, &["stash", "list"]),
    )
}

#[test]
fn a_checkpoint_leaves_the_users_git_exactly_as_it_was() {
    let dir = repo();
    let root = dir.path();
    std::fs::write(root.join("dirty.txt"), "uncommitted\n").unwrap();

    let before = state(root);
    let cp = snapshot(root, "run-1").unwrap();
    let after = state(root);

    assert_eq!(before, after, "the checkpoint disturbed HEAD, branch, worktree or stash");
    assert!(cp.reference.starts_with("refs/minions/"), "checkpoints must live in their own namespace");

    // ...and it is invisible to the ordinary view of the repository.
    let branches = git(root, &["branch", "--list"]);
    assert!(!branches.contains("minions"), "checkpoint appeared as a branch");
    let log = git(root, &["log", "--oneline"]);
    assert_eq!(log.lines().count(), 1, "checkpoint appeared in the log");
}

#[test]
fn rollback_restores_a_modified_file_and_deletes_a_created_one() {
    let dir = repo();
    let root = dir.path().canonicalize().unwrap();
    let cp = snapshot(&root, "run-2").unwrap();

    std::fs::write(root.join("tracked.txt"), "damaged\n").unwrap();
    std::fs::write(root.join("added.txt"), "new\n").unwrap();

    let mutated = vec![
        (root.join("tracked.txt"), true),
        (root.join("added.txt"), false),
    ];
    let done = rollback(&root, &cp, &mutated).unwrap();

    assert_eq!(std::fs::read_to_string(root.join("tracked.txt")).unwrap(), "original\n");
    assert!(!root.join("added.txt").exists(), "a file the run created must be removed");
    assert!(done.contains(&Restored::Content(root.join("tracked.txt"))));
    assert!(done.contains(&Restored::Removed(root.join("added.txt"))));
}

#[test]
fn rollback_cannot_touch_a_file_the_run_never_touched() {
    // The whole point of scoping by the journal: the user edits neighbouring
    // files while a run works, and those edits must survive.
    let dir = repo();
    let root = dir.path().canonicalize().unwrap();
    let cp = snapshot(&root, "run-3").unwrap();

    std::fs::write(root.join("tracked.txt"), "damaged by the agent\n").unwrap();
    std::fs::write(root.join("my-own-work.txt"), "written by the human mid-run\n").unwrap();

    // Only the agent's file is in the journal.
    let mutated = vec![(root.join("tracked.txt"), true)];
    rollback(&root, &cp, &mutated).unwrap();

    assert_eq!(std::fs::read_to_string(root.join("tracked.txt")).unwrap(), "original\n");
    assert_eq!(
        std::fs::read_to_string(root.join("my-own-work.txt")).unwrap(),
        "written by the human mid-run\n",
        "rollback destroyed work outside its scope"
    );
}

#[test]
fn a_gitignored_file_is_reported_as_unrecoverable_rather_than_silently_lost() {
    let dir = repo();
    let root = dir.path().canonicalize().unwrap();
    std::fs::write(root.join("ignored.txt"), "before\n").unwrap();

    let cp = snapshot(&root, "run-4").unwrap();
    std::fs::write(root.join("ignored.txt"), "after\n").unwrap();

    let mutated = vec![(root.join("ignored.txt"), true)];
    let done = rollback(&root, &cp, &mutated).unwrap();

    assert_eq!(done, vec![Restored::Unrecoverable(root.join("ignored.txt"))]);
    assert_eq!(
        std::fs::read_to_string(root.join("ignored.txt")).unwrap(),
        "after\n",
        "an unrecoverable file must be left alone, not deleted"
    );
}

#[test]
fn a_checkpoint_survives_aggressive_garbage_collection() {
    let dir = repo();
    let root = dir.path().canonicalize().unwrap();
    let cp = snapshot(&root, "run-5").unwrap();

    std::fs::write(root.join("tracked.txt"), "damaged\n").unwrap();
    git(&root, &["reflog", "expire", "--expire=now", "--all"]);
    git(&root, &["gc", "--prune=now", "--quiet"]);

    let done = rollback(&root, &cp, &[(root.join("tracked.txt"), true)]).unwrap();
    assert_eq!(done, vec![Restored::Content(root.join("tracked.txt"))]);
    assert_eq!(std::fs::read_to_string(root.join("tracked.txt")).unwrap(), "original\n");
}

#[test]
fn a_repository_with_no_commits_can_still_be_checkpointed() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    git(&root, &["init", "-q"]);
    git(&root, &["config", "user.email", "t@example.com"]);
    git(&root, &["config", "user.name", "t"]);
    std::fs::write(root.join("a.txt"), "one\n").unwrap();

    let cp = snapshot(&root, "run-6").expect("a fresh repository must be checkpointable");
    std::fs::write(root.join("a.txt"), "two\n").unwrap();
    rollback(&root, &cp, &[(root.join("a.txt"), true)]).unwrap();
    assert_eq!(std::fs::read_to_string(root.join("a.txt")).unwrap(), "one\n");
}

#[test]
fn a_folder_that_is_not_a_repository_refuses_up_front() {
    let dir = tempfile::tempdir().unwrap();
    let err = snapshot(dir.path(), "run-7").unwrap_err().to_string();
    assert!(err.contains("not a git repository"), "unhelpful message: {err}");
    assert!(err.contains("rollback"), "the message must name what is lost: {err}");
}

#[test]
fn discarding_a_checkpoint_removes_its_ref() {
    let dir = repo();
    let root = dir.path().canonicalize().unwrap();
    let cp = snapshot(&root, "run-8").unwrap();
    assert!(!git(&root, &["for-each-ref", "refs/minions/"]).is_empty());
    discard(&root, &cp).unwrap();
    assert!(git(&root, &["for-each-ref", "refs/minions/"]).is_empty());
}

#[test]
fn a_path_outside_the_root_is_reported_not_guessed_at() {
    let dir = repo();
    let root = dir.path().canonicalize().unwrap();
    let cp = snapshot(&root, "run-9").unwrap();
    let outside = PathBuf::from("/tmp/definitely-not-in-the-project.txt");
    let done = rollback(&root, &cp, &[(outside.clone(), true)]).unwrap();
    assert_eq!(done, vec![Restored::Unrecoverable(outside)]);
}
