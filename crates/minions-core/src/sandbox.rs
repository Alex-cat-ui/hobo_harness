//! All authority in one pure function.
//!
//! The model holds no capabilities: it returns `ToolCall` values and the core
//! decides. That turns "sandboxing an agent" into classifying a value, which is
//! total, deterministic, and provable by property and mutation testing.
//!
//! Two rules carry the weight:
//!   * paths are resolved before they are compared — the classic escape is a
//!     `..` or a symlink compared as written;
//!   * the floor is expressed as an early return, so no later branch can widen
//!     it however the mode or consents are set.

use std::path::{Component, Path, PathBuf};

/// Everything a model is able to ask for. Closed by construction: anything
/// outside this set is a parse failure, not an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCall {
    ReadFile { path: PathBuf },
    WriteFile { path: PathBuf, content: String },
    /// Replace one fragment of a file. For models that cannot hold a whole
    /// file in one argument — which is most of them below a certain size.
    ReplaceInFile { path: PathBuf, find: String, replace: String },
    /// Add to the end of a file. The natural shape of "add a function".
    AppendToFile { path: PathBuf, content: String },
    DeleteFile { path: PathBuf },
    ApplyPatch { diff: String },
    RunCommand { program: String, args: Vec<String> },
    Search { query: String, limit: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    AskForEverything,
    AskBeforeTouchingSource,
    DoNotAskInsideSandbox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateReason {
    WriteOutsideRoot,
    WriteToSource,
    WriteInsideRoot,
    Delete,
    Command,
    RemoteRepository,
    PackageInstall,
    Network,
    PrivilegeEscalation,
    NodeGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForbidReason {
    ReadOutsideRoot,
    UnresolvablePath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Allowed,
    /// `consentable` is false for floor categories: scoped consent can never
    /// grant them, however narrow the path.
    Gated { reason: GateReason, consentable: bool },
    Forbidden { reason: ForbidReason },
}

impl Verdict {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Verdict::Allowed)
    }
    pub fn is_gated(&self) -> bool {
        matches!(self, Verdict::Gated { .. })
    }
}

/// A grant, bounded to one run and one path prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopedConsent {
    pub prefix: PathBuf,
}

#[derive(Debug, Clone)]
pub struct Request<'a> {
    pub call: &'a ToolCall,
    /// Canonical project root. Everything is judged against this.
    pub root: &'a Path,
    pub mode: PermissionMode,
    pub consents: &'a [ScopedConsent],
    /// The workflow author put an explicit gate on this node.
    pub node_gate: bool,
    /// Paths the project treats as source, relative to the root.
    pub source_roots: &'a [PathBuf],
}

/// Programs that can never be granted by scoped consent, whatever the mode.
fn floor_category(program: &str, args: &[String]) -> Option<GateReason> {
    let base = Path::new(program)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(program)
        .to_ascii_lowercase();

    match base.as_str() {
        "sudo" | "su" | "doas" => return Some(GateReason::PrivilegeEscalation),
        "curl" | "wget" | "nc" | "ncat" | "netcat" | "ssh" | "scp" | "sftp" | "rsync" | "telnet" => {
            return Some(GateReason::Network)
        }
        "brew" | "apt" | "apt-get" | "yum" | "dnf" | "pacman" | "gem" | "pipx" => {
            return Some(GateReason::PackageInstall)
        }
        _ => {}
    }

    let a: Vec<String> = args.iter().map(|s| s.to_ascii_lowercase()).collect();

    if base == "git" {
        const REMOTE: [&str; 7] = ["push", "pull", "fetch", "clone", "remote", "submodule", "archive"];
        if a.iter().any(|x| REMOTE.contains(&x.as_str())) {
            return Some(GateReason::RemoteRepository);
        }
    }

    const INSTALLERS: [&str; 6] = ["npm", "pnpm", "yarn", "pip", "pip3", "cargo"];
    if INSTALLERS.contains(&base.as_str()) {
        const SUB: [&str; 5] = ["install", "add", "publish", "uninstall", "remove"];
        if a.iter().any(|x| SUB.contains(&x.as_str())) {
            return Some(GateReason::PackageInstall);
        }
    }

    None
}

/// Lexical normalisation followed by symlink resolution of the deepest existing
/// ancestor. `canonicalize` alone is not enough: a file being created does not
/// exist yet, and comparing the path as written is how sandboxes are escaped.
pub fn resolve(path: &Path, root: &Path) -> Option<PathBuf> {
    let joined = if path.is_absolute() { path.to_path_buf() } else { root.join(path) };

    let mut lexical = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !lexical.pop() {
                    return None;
                }
            }
            other => lexical.push(other.as_os_str()),
        }
    }

    // Resolve symlinks on the part that exists, keep the rest as written.
    let mut existing = lexical.clone();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        let Some(name) = existing.file_name().map(|s| s.to_os_string()) else {
            return Some(lexical);
        };
        tail.push(name);
        if !existing.pop() {
            return Some(lexical);
        }
    }
    let mut out = existing.canonicalize().ok()?;
    for name in tail.into_iter().rev() {
        out.push(name);
    }
    Some(out)
}

fn inside(path: &Path, root: &Path) -> bool {
    path.starts_with(root)
}

fn under_consent(path: &Path, consents: &[ScopedConsent]) -> bool {
    consents.iter().any(|c| path.starts_with(&c.prefix))
}

fn is_source(path: &Path, root: &Path, source_roots: &[PathBuf]) -> bool {
    if source_roots.is_empty() {
        return true;
    }
    source_roots.iter().any(|s| {
        let full = if s.is_absolute() { s.clone() } else { root.join(s) };
        path.starts_with(full)
    })
}

/// The whole authority decision.
pub fn classify(req: &Request<'_>) -> Verdict {
    use GateReason::*;
    use PermissionMode::*;
    use Verdict::*;

    match req.call {
        // Reading and searching inside the root is the one thing that never asks.
        ToolCall::Search { .. } => Allowed,

        ToolCall::ReadFile { path } => {
            let Some(p) = resolve(path, req.root) else {
                return Forbidden { reason: ForbidReason::UnresolvablePath };
            };
            if !inside(&p, req.root) {
                // Not gated: a gate implies a legitimate use, and none exists.
                return Forbidden { reason: ForbidReason::ReadOutsideRoot };
            }
            Allowed
        }

        // ---- floor: nothing below may widen these ----
        ToolCall::DeleteFile { path } => {
            if resolve(path, req.root).is_none() {
                return Forbidden { reason: ForbidReason::UnresolvablePath };
            }
            Gated { reason: Delete, consentable: false }
        }

        ToolCall::RunCommand { program, args } => {
            if let Some(reason) = floor_category(program, args) {
                return Gated { reason, consentable: false };
            }
            // Any other command is still gated; consent may grant it for the run.
            Gated { reason: Command, consentable: true }
        }

        ToolCall::WriteFile { path, .. }
        | ToolCall::ReplaceInFile { path, .. }
        | ToolCall::AppendToFile { path, .. } => {
            let Some(p) = resolve(path, req.root) else {
                return Forbidden { reason: ForbidReason::UnresolvablePath };
            };

            if !inside(&p, req.root) {
                return Gated { reason: WriteOutsideRoot, consentable: false };
            }

            // An explicit gate is a specific instruction; a mode is a blanket
            // default. Specific beats general.
            if req.node_gate {
                return Gated { reason: NodeGate, consentable: false };
            }

            if under_consent(&p, req.consents) {
                return Allowed;
            }

            match req.mode {
                AskForEverything => Gated { reason: WriteInsideRoot, consentable: true },
                AskBeforeTouchingSource => {
                    if is_source(&p, req.root, req.source_roots) {
                        Gated { reason: WriteToSource, consentable: true }
                    } else {
                        Allowed
                    }
                }
                DoNotAskInsideSandbox => Allowed,
            }
        }

        // A patch is judged per target file by the dispatcher, which classifies
        // each one through this same function. The patch itself always gates.
        ToolCall::ApplyPatch { .. } => Gated { reason: WriteInsideRoot, consentable: true },
    }
}
