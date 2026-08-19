//! The action journal: every dispatched call, its verdict and its outcome.
//!
//! This is not diagnostics. It is the path list rollback restores from, and the
//! record that answers "what did the agent do to my project" (SPEC FR-57).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    Read,
    Wrote,
    Deleted,
    Ran,
    Searched,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub at: String,
    pub node: String,
    pub attempt: u32,
    pub effect: Effect,
    pub verdict: String,
    /// Paths the call actually touched. Rollback reads exactly this.
    #[serde(default)]
    pub paths: Vec<PathBuf>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub note: Option<String>,
    /// True when the path existed before this run first touched it. Rollback
    /// restores those and deletes the rest.
    #[serde(default)]
    pub existed_before: Vec<bool>,
}

pub struct Journal {
    path: PathBuf,
}

impl Journal {
    pub fn create(run_dir: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(run_dir)?;
        Ok(Self { path: run_dir.join("journal.jsonl") })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn append(&self, entry: Entry) -> std::io::Result<()> {
        let line = serde_json::to_string(&entry).expect("entry serialises");
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        writeln!(f, "{line}")?;
        Ok(())
    }

    pub fn load(path: &Path) -> std::io::Result<Vec<Entry>> {
        let text = std::fs::read_to_string(path)?;
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect())
    }

    /// How many files a given node actually wrote. The record, not the claim.
    ///
    /// An agent that reports work it did not do is the worst failure a product
    /// like this can have: a crash is visible, a fabricated success is not. The
    /// journal always knew; it simply was not asked.
    pub fn writes_by(entries: &[Entry], node: &str) -> usize {
        entries
            .iter()
            .filter(|e| e.node == node && e.effect == Effect::Wrote)
            .filter(|e| e.verdict != "Rejected" && e.verdict != "Forbidden")
            .count()
    }

    /// Paths this run wrote or deleted, in first-touch order, each with whether
    /// it existed before the run touched it. This is the whole input to rollback
    /// scoping: a path absent here is never read, written or considered.
    pub fn mutated_paths(entries: &[Entry]) -> Vec<(PathBuf, bool)> {
        let mut seen: Vec<(PathBuf, bool)> = Vec::new();
        for e in entries {
            if !matches!(e.effect, Effect::Wrote | Effect::Deleted) {
                continue;
            }
            for (i, p) in e.paths.iter().enumerate() {
                if seen.iter().any(|(q, _)| q == p) {
                    continue;
                }
                let existed = e.existed_before.get(i).copied().unwrap_or(false);
                seen.push((p.clone(), existed));
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(effect: Effect, paths: &[&str], existed: &[bool]) -> Entry {
        Entry {
            at: "2026-08-16T00:00:00Z".into(),
            node: "coder".into(),
            attempt: 1,
            effect,
            verdict: "Allowed".into(),
            paths: paths.iter().map(PathBuf::from).collect(),
            command: None,
            exit_code: None,
            note: None,
            existed_before: existed.to_vec(),
        }
    }

    #[test]
    fn round_trips_through_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut j = Journal::create(dir.path()).unwrap();
        j.append(entry(Effect::Wrote, &["a.rs"], &[true])).unwrap();
        j.append(entry(Effect::Read, &["b.rs"], &[])).unwrap();
        let loaded = Journal::load(j.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].effect, Effect::Wrote);
    }

    #[test]
    fn mutated_paths_ignores_reads() {
        let es = vec![
            entry(Effect::Read, &["read-only.rs"], &[]),
            entry(Effect::Wrote, &["changed.rs"], &[true]),
            entry(Effect::Searched, &[], &[]),
        ];
        let m = Journal::mutated_paths(&es);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, PathBuf::from("changed.rs"));
    }

    #[test]
    fn first_touch_wins_so_rollback_knows_what_to_delete() {
        // The file did not exist when the run first wrote it; a later write must
        // not flip that to "existed", or rollback would restore instead of delete.
        let es = vec![
            entry(Effect::Wrote, &["new.rs"], &[false]),
            entry(Effect::Wrote, &["new.rs"], &[true]),
        ];
        let m = Journal::mutated_paths(&es);
        assert_eq!(m.len(), 1);
        assert!(!m[0].1, "first touch must win");
    }

    #[test]
    fn writes_by_counts_only_what_that_node_really_did() {
        let mut coder = entry(Effect::Wrote, &["a.py"], &[false]);
        coder.node = "coder".into();
        let mut refused = entry(Effect::Wrote, &["b.py"], &[false]);
        refused.node = "coder".into();
        refused.verdict = "Rejected".into();
        let mut other = entry(Effect::Wrote, &["c.py"], &[false]);
        other.node = "tester".into();
        let mut read = entry(Effect::Read, &["d.py"], &[]);
        read.node = "coder".into();

        let es = vec![coder, refused, other, read];
        assert_eq!(Journal::writes_by(&es, "coder"), 1, "a refused write must not count as work done");
        assert_eq!(Journal::writes_by(&es, "tester"), 1);
        assert_eq!(Journal::writes_by(&es, "nobody"), 0);
    }

    #[test]
    fn deletions_are_in_scope_for_rollback() {
        let es = vec![entry(Effect::Deleted, &["gone.rs"], &[true])];
        let m = Journal::mutated_paths(&es);
        assert_eq!(m, vec![(PathBuf::from("gone.rs"), true)]);
    }
}
