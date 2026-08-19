//! The only thing that acts.
//!
//! Every effect the product has on the machine passes through `dispatch`. It
//! classifies, asks a human where the classification says to, performs the
//! effect, and journals what happened — in that order, always.

use crate::journal::{Effect, Entry, Journal};
use crate::node::now_utc;
use crate::sandbox::{classify, ForbidReason, GateReason, PermissionMode, Request, ScopedConsent, ToolCall, Verdict};
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    Approve,
    /// Approve and widen, for this run and this prefix only.
    ApproveWithConsent { prefix: PathBuf },
    Edit { call: ToolCall },
    Reject { note: String },
}

/// Who answers a gate. A trait so the engine can be driven by the UI in the
/// product and by a script in tests.
pub trait GateAuthority: Send + Sync {
    fn ask(&self, call: &ToolCall, reason: GateReason, consentable: bool) -> GateDecision;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Read(String),
    Wrote,
    Deleted,
    Ran { exit_code: i32, stdout: String, stderr: String },
    /// The agent asked for something it may not have. This is returned to the
    /// agent as a refusal rather than failing the run.
    Refused(String),
}

pub struct Dispatcher<'a> {
    root: PathBuf,
    mode: PermissionMode,
    consents: Vec<ScopedConsent>,
    source_roots: Vec<PathBuf>,
    authority: &'a dyn GateAuthority,
    journal: Journal,
    node: String,
    attempt: u32,
    command_timeout: Duration,
}

impl<'a> Dispatcher<'a> {
    pub fn new(
        root: &Path,
        mode: PermissionMode,
        source_roots: Vec<PathBuf>,
        authority: &'a dyn GateAuthority,
        journal: Journal,
    ) -> Result<Self> {
        Ok(Self {
            root: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
            mode,
            consents: Vec::new(),
            source_roots,
            authority,
            journal,
            node: "unknown".into(),
            attempt: 1,
            command_timeout: Duration::from_secs(600),
        })
    }

    pub fn for_node(&mut self, node: &str, attempt: u32) {
        self.node = node.to_string();
        self.attempt = attempt;
    }

    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    pub fn consents(&self) -> &[ScopedConsent] {
        &self.consents
    }

    fn record(
        &self,
        effect: Effect,
        verdict: &str,
        paths: Vec<PathBuf>,
        existed_before: Vec<bool>,
        command: Option<String>,
        exit_code: Option<i32>,
        note: Option<String>,
    ) -> Result<()> {
        self.journal.append(Entry {
            at: now_utc(),
            node: self.node.clone(),
            attempt: self.attempt,
            effect,
            verdict: verdict.to_string(),
            paths,
            command,
            exit_code,
            note,
            existed_before,
        })?;
        Ok(())
    }

    /// Classify, ask if required, act, journal. A rejection is returned to the
    /// agent as a refusal so it can proceed differently, not as a run failure.
    pub fn dispatch(&mut self, call: ToolCall, node_gate: bool) -> Result<Outcome> {
        let verdict = {
            let req = Request {
                call: &call,
                root: &self.root,
                mode: self.mode,
                consents: &self.consents,
                node_gate,
                source_roots: &self.source_roots,
            };
            classify(&req)
        };

        match verdict {
            Verdict::Forbidden { reason } => {
                let why = match reason {
                    ForbidReason::ReadOutsideRoot => "reading outside the working folder is not permitted",
                    ForbidReason::UnresolvablePath => "that path cannot be resolved",
                };
                self.record(effect_of(&call), "Forbidden", paths_of(&call), vec![], command_of(&call), None, Some(why.into()))?;
                Ok(Outcome::Refused(why.to_string()))
            }
            Verdict::Allowed => self.perform(call, "Allowed"),
            Verdict::Gated { reason, consentable } => {
                match self.authority.ask(&call, reason, consentable) {
                    GateDecision::Reject { note } => {
                        self.record(effect_of(&call), "Rejected", paths_of(&call), vec![], command_of(&call), None, Some(note.clone()))?;
                        Ok(Outcome::Refused(note))
                    }
                    GateDecision::Approve => self.perform(call, "Approved"),
                    GateDecision::Edit { call: edited } => self.perform(edited, "Approved after edit"),
                    GateDecision::ApproveWithConsent { prefix } => {
                        if !consentable {
                            // The floor is not consentable. Honour the approval
                            // once; refuse to widen.
                            self.perform(call, "Approved (consent refused: floor)")
                        } else {
                            self.consents.push(ScopedConsent { prefix });
                            self.perform(call, "Approved with consent")
                        }
                    }
                }
            }
        }
    }

    fn perform(&mut self, call: ToolCall, verdict: &str) -> Result<Outcome> {
        match call {
            ToolCall::Search { .. } => {
                self.record(Effect::Searched, verdict, vec![], vec![], None, None, None)?;
                Err(anyhow!("no index is built for this project yet"))
            }

            ToolCall::ReadFile { ref path } => {
                let p = self.resolved(path)?;
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| anyhow!("reading {}: {e}", p.display()))?;
                self.record(Effect::Read, verdict, vec![p], vec![], None, None, None)?;
                Ok(Outcome::Read(text))
            }

            ToolCall::WriteFile { ref path, ref content } => {
                let p = self.resolved(path)?;
                let existed = p.exists();
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&p, content)?;
                self.record(Effect::Wrote, verdict, vec![p], vec![existed], None, None, None)?;
                Ok(Outcome::Wrote)
            }

            ToolCall::ReplaceInFile { ref path, ref find, ref replace } => {
                let p = self.resolved(path)?;
                let before = std::fs::read_to_string(&p)
                    .map_err(|e| anyhow!("reading {}: {e}", p.display()))?;
                let hits = before.matches(find.as_str()).count();
                // Exactly once, or not at all. A fragment appearing twice means
                // the model has not said which one it meant, and guessing would
                // change code it never looked at.
                if hits == 0 {
                    return Ok(Outcome::Refused(format!(
                        "that fragment does not appear in {}. Read the file and quote an existing fragment exactly, including indentation.",
                        path.display()
                    )));
                }
                if hits > 1 {
                    return Ok(Outcome::Refused(format!(
                        "that fragment appears {hits} times in {}. Quote a longer fragment so there is only one match.",
                        path.display()
                    )));
                }
                let after = before.replacen(find.as_str(), replace, 1);
                std::fs::write(&p, &after)?;
                self.record(Effect::Wrote, verdict, vec![p], vec![true], None, None, None)?;
                Ok(Outcome::Wrote)
            }

            ToolCall::AppendToFile { ref path, ref content } => {
                let p = self.resolved(path)?;
                let existed = p.exists();
                let mut before = std::fs::read_to_string(&p).unwrap_or_default();
                if !before.is_empty() && !before.ends_with('\n') {
                    before.push('\n');
                }
                before.push_str(content);
                if !before.ends_with('\n') {
                    before.push('\n');
                }
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&p, before)?;
                self.record(Effect::Wrote, verdict, vec![p], vec![existed], None, None, None)?;
                Ok(Outcome::Wrote)
            }

            ToolCall::DeleteFile { ref path } => {
                let p = self.resolved(path)?;
                let existed = p.exists();
                if existed {
                    std::fs::remove_file(&p)?;
                }
                self.record(Effect::Deleted, verdict, vec![p], vec![existed], None, None, None)?;
                Ok(Outcome::Deleted)
            }

            ToolCall::ApplyPatch { ref diff } => {
                let targets = patch_targets(diff);
                if targets.is_empty() {
                    return Err(anyhow!("the patch names no target files"));
                }
                // Each target is judged individually through the same function.
                for t in &targets {
                    let probe = ToolCall::WriteFile { path: t.clone(), content: String::new() };
                    let req = Request {
                        call: &probe,
                        root: &self.root,
                        mode: self.mode,
                        consents: &self.consents,
                        node_gate: false,
                        source_roots: &self.source_roots,
                    };
                    if let Verdict::Forbidden { .. } = classify(&req) {
                        return Ok(Outcome::Refused(format!("the patch touches {}, which is outside the working folder", t.display())));
                    }
                }
                let existed: Vec<bool> = targets.iter().map(|t| self.root.join(t).exists()).collect();
                // --recount tells git to recompute the hunk line counts itself.
                // Generated diffs get those counts wrong constantly — it is the
                // single most common way a model's patch fails to apply — and
                // the counts are redundant with the hunk body anyway.
                // -C1 loosens context matching enough to survive a stale line
                // without accepting a patch that lands somewhere else.
                let out = self.git(&["apply", "--recount", "-C1", "--whitespace=nowarn", "-"], Some(diff))?;
                if !out.status.success() {
                    let err = String::from_utf8_lossy(&out.stderr).to_string();
                    self.record(Effect::Wrote, verdict, vec![], vec![], None, out.status.code(), Some(err.clone()))?;
                    return Err(anyhow!("git apply failed: {err}"));
                }
                let paths: Vec<PathBuf> = targets.iter().map(|t| self.root.join(t)).collect();
                self.record(Effect::Wrote, verdict, paths, existed, None, out.status.code(), None)?;
                Ok(Outcome::Wrote)
            }

            ToolCall::RunCommand { ref program, ref args } => {
                let out = Command::new(program)
                    .args(args)
                    .current_dir(&self.root)
                    .env_clear()
                    .env("PATH", std::env::var("PATH").unwrap_or_default())
                    .env("HOME", std::env::var("HOME").unwrap_or_default())
                    .output()
                    .map_err(|e| anyhow!("running {program}: {e}"))?;
                let code = out.status.code().unwrap_or(-1);
                let text = format!("{program} {}", args.join(" "));
                self.record(Effect::Ran, verdict, vec![], vec![], Some(text), Some(code), None)?;
                Ok(Outcome::Ran {
                    exit_code: code,
                    stdout: String::from_utf8_lossy(&out.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&out.stderr).to_string(),
                })
            }
        }
    }

    fn resolved(&self, path: &Path) -> Result<PathBuf> {
        crate::sandbox::resolve(path, &self.root).ok_or_else(|| anyhow!("cannot resolve {}", path.display()))
    }

    fn git(&self, args: &[&str], stdin: Option<&str>) -> Result<std::process::Output> {
        use std::io::Write;
        use std::process::Stdio;
        let mut child = Command::new("git")
            .args(args)
            .current_dir(&self.root)
            .stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        if let (Some(text), Some(mut si)) = (stdin, child.stdin.take()) {
            si.write_all(text.as_bytes())?;
        }
        Ok(child.wait_with_output()?)
    }

    pub fn set_command_timeout(&mut self, d: Duration) {
        self.command_timeout = d;
    }
}

fn effect_of(call: &ToolCall) -> Effect {
    match call {
        ToolCall::ReadFile { .. } => Effect::Read,
        ToolCall::WriteFile { .. }
        | ToolCall::ReplaceInFile { .. }
        | ToolCall::AppendToFile { .. }
        | ToolCall::ApplyPatch { .. } => Effect::Wrote,
        ToolCall::DeleteFile { .. } => Effect::Deleted,
        ToolCall::RunCommand { .. } => Effect::Ran,
        ToolCall::Search { .. } => Effect::Searched,
    }
}

fn paths_of(call: &ToolCall) -> Vec<PathBuf> {
    match call {
        ToolCall::ReadFile { path }
        | ToolCall::WriteFile { path, .. }
        | ToolCall::ReplaceInFile { path, .. }
        | ToolCall::AppendToFile { path, .. }
        | ToolCall::DeleteFile { path } => vec![path.clone()],
        _ => vec![],
    }
}

fn command_of(call: &ToolCall) -> Option<String> {
    match call {
        ToolCall::RunCommand { program, args } => Some(format!("{program} {}", args.join(" "))),
        _ => None,
    }
}

/// Target paths of a unified diff, taken from the `+++ b/...` headers.
pub fn patch_targets(diff: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for line in diff.lines() {
        let Some(rest) = line.strip_prefix("+++ ") else { continue };
        let rest = rest.trim();
        if rest == "/dev/null" {
            continue;
        }
        let cleaned = rest.strip_prefix("b/").unwrap_or(rest);
        let cleaned = cleaned.split('\t').next().unwrap_or(cleaned);
        let p = PathBuf::from(cleaned);
        if !out.contains(&p) {
            out.push(p);
        }
    }
    out
}
