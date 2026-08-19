//! Snapshot and journal-scoped restore.
//!
//! The snapshot captures the whole working tree as a git object without
//! touching anything the user owns: not HEAD, not the branch, not the index,
//! not the stash, not a file. It is anchored under `refs/minions/` so garbage
//! collection cannot reclaim it and `git branch` never shows it.
//!
//! Restore takes its *content* from the snapshot and its *path list* from the
//! run journal. The user edits neighbouring files while a run works; a
//! whole-tree revert would destroy that, and a journal-scoped one cannot.

use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    pub commit: String,
    pub reference: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Restored {
    Content(PathBuf),
    Removed(PathBuf),
    /// Present in the journal but absent from the snapshot, because
    /// `.gitignore` excluded it. Nothing can be restored; the user was warned
    /// at the gate, and it is reported again here.
    Unrecoverable(PathBuf),
}

fn git(root: &Path, args: &[&str]) -> Result<std::process::Output> {
    Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .with_context(|| format!("running git {}", args.join(" ")))
}

fn git_ok(root: &Path, args: &[&str]) -> Result<String> {
    let out = git(root, args)?;
    if !out.status.success() {
        bail!("git {} failed: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

pub fn is_repository(root: &Path) -> bool {
    git(root, &["rev-parse", "--is-inside-work-tree"])
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Captures the working tree. Nothing the user owns is modified.
pub fn snapshot(root: &Path, run_id: &str) -> Result<Checkpoint> {
    if !is_repository(root) {
        bail!("not a git repository, so no checkpoint can be taken and no rollback will be possible");
    }

    let index = std::env::temp_dir().join(format!("minions-index-{run_id}"));
    let _ = std::fs::remove_file(&index);

    let staged = Command::new("git")
        .args(["add", "-A"])
        .current_dir(root)
        .env("GIT_INDEX_FILE", &index)
        .output()
        .context("staging into a scratch index")?;
    if !staged.status.success() {
        bail!("staging failed: {}", String::from_utf8_lossy(&staged.stderr).trim());
    }

    let tree = {
        let out = Command::new("git")
            .args(["write-tree"])
            .current_dir(root)
            .env("GIT_INDEX_FILE", &index)
            .output()
            .context("writing tree")?;
        if !out.status.success() {
            bail!("write-tree failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    let _ = std::fs::remove_file(&index);

    let message = format!("minions checkpoint {run_id}");
    let head = git_ok(root, &["rev-parse", "HEAD"]).ok();
    let commit = match &head {
        Some(h) => git_ok(root, &["commit-tree", &tree, "-p", h, "-m", &message])?,
        // A repository with no commits yet has no HEAD to parent onto.
        None => git_ok(root, &["commit-tree", &tree, "-m", &message])?,
    };

    let reference = format!("refs/minions/checkpoints/{run_id}");
    git_ok(root, &["update-ref", &reference, &commit])?;

    Ok(Checkpoint { commit, reference })
}

fn in_snapshot(root: &Path, commit: &str, rel: &Path) -> bool {
    let spec = format!("{commit}:{}", rel.to_string_lossy());
    git(root, &["cat-file", "-e", &spec]).map(|o| o.status.success()).unwrap_or(false)
}

fn relative<'a>(root: &Path, path: &'a Path) -> Option<PathBuf> {
    path.strip_prefix(root).ok().map(|p| p.to_path_buf())
}

/// Restores exactly the paths the journal names, and nothing else.
pub fn rollback(root: &Path, cp: &Checkpoint, mutated: &[(PathBuf, bool)]) -> Result<Vec<Restored>> {
    let mut done = Vec::new();

    for (path, existed_before) in mutated {
        let Some(rel) = relative(root, path) else {
            // Outside the root: a gate let it through deliberately, and the
            // snapshot never covered it.
            done.push(Restored::Unrecoverable(path.clone()));
            continue;
        };

        if in_snapshot(root, &cp.commit, &rel) {
            let spec = format!("{}:{}", cp.commit, rel.to_string_lossy());
            let out = git(root, &["cat-file", "blob", &spec])?;
            if !out.status.success() {
                return Err(anyhow!("reading {} from the checkpoint failed", rel.display()));
            }
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, &out.stdout)?;
            done.push(Restored::Content(path.clone()));
        } else if *existed_before {
            // It existed when the run started but is absent from the snapshot,
            // which means .gitignore excluded it. Nothing to restore from.
            done.push(Restored::Unrecoverable(path.clone()));
        } else {
            // The run created it. Rollback means it should not exist.
            if path.exists() {
                std::fs::remove_file(path)
                    .with_context(|| format!("removing {}", path.display()))?;
            }
            done.push(Restored::Removed(path.clone()));
        }
    }

    Ok(done)
}

/// Drops a checkpoint's ref. Called when its run is deleted from history.
pub fn discard(root: &Path, cp: &Checkpoint) -> Result<()> {
    git_ok(root, &["update-ref", "-d", &cp.reference])?;
    Ok(())
}
