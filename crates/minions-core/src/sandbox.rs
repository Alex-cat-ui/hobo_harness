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
    /// A git subcommand that rewrites the working tree or moves a ref — one of
    /// them destroys the point the run is rolled back to.
    RepositoryRewrite,
    /// The shell writes a file itself, where the journal cannot see it and the
    /// rollback cannot find it.
    ShellRedirect,
    /// The text does not name what will run: `eval`, or a command substitution
    /// standing where a program name should be.
    OpaqueCommand,
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
///
/// A model never names a program. `chat::to_tool_call` turns every
/// `run_command` into `bash -lc <script>`, so what this function used to see
/// was always `bash`, and the floor was dead for everything an agent ran
/// (finding 25). The shell is therefore transparent here: the script is read
/// and every command in it is judged by this same function.
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
        // `DeleteFile` is on the floor in every mode, and a command was the way
        // round it: the same effect, one word of shell away.
        "rm" | "rmdir" | "unlink" | "shred" => return Some(GateReason::Delete),
        // Runs something this text does not name, so nothing below can judge it.
        "eval" | "source" | "." => return Some(GateReason::OpaqueCommand),
        _ => {}
    }

    let a: Vec<String> = args.iter().map(|s| s.to_ascii_lowercase()).collect();

    if base == "git" {
        // The subcommand is the word that decides, and it is the first word
        // that is not a flag. Reading any argument instead made a commit
        // message saying "push" gate as a push.
        if let Some(sub) = git_subcommand(&a) {
            const REMOTE: [&str; 7] = ["push", "pull", "fetch", "clone", "remote", "submodule", "archive"];
            if REMOTE.contains(&sub) {
                return Some(GateReason::RemoteRepository);
            }

            // Rewrites the working tree or moves a ref. `mrollback` returns the
            // project to a checkpoint held in a git ref, so one of these
            // destroys the way back — silently, and under `--yes` without being
            // asked.
            const REWRITE: [&str; 11] = [
                "reset", "checkout", "restore", "switch", "clean", "stash", "update-ref", "rebase",
                "filter-branch", "reflog", "gc",
            ];
            if REWRITE.contains(&sub) {
                return Some(GateReason::RepositoryRewrite);
            }

            // Listing branches and tags is reading; deleting or forcing one is
            // not.
            const DESTRUCTIVE: [&str; 5] = ["-d", "-D", "--delete", "-f", "--force"];
            if matches!(sub, "branch" | "tag")
                && args.iter().any(|x| DESTRUCTIVE.contains(&x.as_str()))
            {
                return Some(GateReason::RepositoryRewrite);
            }
        }
    }

    const INSTALLERS: [&str; 6] = ["npm", "pnpm", "yarn", "pip", "pip3", "cargo"];
    if INSTALLERS.contains(&base.as_str()) {
        const SUB: [&str; 5] = ["install", "add", "publish", "uninstall", "remove"];
        if a.iter().any(|x| SUB.contains(&x.as_str())) {
            return Some(GateReason::PackageInstall);
        }
    }

    // A shell inside a shell needs no depth guard: the payload is one word of
    // the arguments it was given, so every step works on strictly less text
    // than the one before it and the walk cannot go on for ever.
    if let Some(script) = shell_payload(&base, args) {
        return scan_script(script);
    }

    None
}

/// The subcommand of a `git` invocation: the first argument that is not a flag.
/// The few flags that take a value before the subcommand carry it away with
/// them, so `git -C other status` is a status and not an `other`. Arguments
/// arrive lowercased, which is why `-C` is matched as `-c`.
fn git_subcommand(args: &[String]) -> Option<&str> {
    const TAKES_A_VALUE: [&str; 4] = ["-c", "--git-dir", "--work-tree", "--namespace"];
    let mut rest = args.iter();
    while let Some(a) = rest.next() {
        if !a.starts_with('-') {
            return Some(a.as_str());
        }
        if TAKES_A_VALUE.contains(&a.as_str()) {
            rest.next();
        }
    }
    None
}

/// The script a shell invocation would run, if this is one.
fn shell_payload<'a>(base: &str, args: &'a [String]) -> Option<&'a str> {
    if !matches!(base, "sh" | "bash" | "zsh" | "dash" | "ksh" | "fish") {
        return None;
    }
    let mut rest = args.iter();
    while let Some(a) = rest.next() {
        // `-c`, and the `-lc` / `-ic` the product itself sends: a single-dash
        // flag carrying `c` means the next argument is the script.
        if a.starts_with('-') && !a.starts_with("--") && a.contains('c') {
            return rest.next().map(|s| s.as_str());
        }
    }
    None
}

/// Words that stand in front of the real command without changing what it is.
const TRANSPARENT: [&str; 8] = ["env", "command", "nohup", "time", "nice", "stdbuf", "xargs", "exec"];

/// Every command in a script, judged by the same floor as a bare program.
///
/// Quoting is deliberately not honoured. A separator inside quotes only makes
/// one more segment to classify, which errs towards asking a human, while
/// honouring quotes would let `echo "; sudo rm"` hide the word from the scan.
///
/// This is a filter over what a model writes, not a boundary. A program name
/// assembled at runtime cannot be read here, and a command is not confined by
/// the operating system at all — `bash -lc` runs with the rights of whoever
/// started the product. Hence the two shapes that cannot be read are on the
/// floor themselves: `eval`, and a substitution standing where a program name
/// belongs.
fn scan_script(script: &str) -> Option<GateReason> {
    const SEPARATORS: [char; 9] = [';', '\n', '|', '&', '(', ')', '`', '{', '}'];

    for segment in script.split(SEPARATORS) {
        let quotes = |w: &str| w.trim_matches(|c| c == '\'' || c == '"').to_string();
        let mut words = segment
            .split_whitespace()
            .map(quotes)
            .skip_while(|w| w.contains('=') || TRANSPARENT.contains(&w.to_ascii_lowercase().as_str()));

        let Some(program) = words.next() else { continue };
        let rest: Vec<String> = words.collect();
        if let Some(reason) = floor_category(&program, &rest) {
            return Some(reason);
        }
    }

    if writes_a_file(script) {
        return Some(GateReason::ShellRedirect);
    }
    if script.contains("$(") || script.contains('`') {
        return Some(GateReason::OpaqueCommand);
    }
    None
}

/// A `>` that sends output into a file, as opposed to `2>&1`, which only moves
/// a descriptor and is how a model asks to see both streams.
fn writes_a_file(script: &str) -> bool {
    let chars: Vec<char> = script.chars().collect();
    chars
        .iter()
        .enumerate()
        .any(|(i, c)| *c == '>' && chars.get(i + 1) != Some(&'&'))
}

/// The path the kernel will act on, for a path that may not exist yet.
///
/// `canonicalize` alone cannot be used — a file being created does not exist —
/// and resolving only the deepest existing ancestor is not enough either:
/// `Path::exists` follows a link, so a link whose target is missing looks
/// absent, was carried through as an ordinary name, and the write then
/// followed it out of the root (finding 24). Two rules follow, and both are
/// the kernel's own:
///
///   * every component is inspected with `symlink_metadata`, and a link is
///     replaced by its target whether or not that target exists;
///   * `..` is applied to what the walk has resolved so far, never to the path
///     as written — otherwise `link/..` cancels a link pointing elsewhere and
///     lands back inside the root (finding 32).
pub fn resolve(path: &Path, root: &Path) -> Option<PathBuf> {
    // The kernel gives up on a chain of links rather than following it for
    // ever; so does this, which turns a loop into `UnresolvablePath`. Spent
    // downwards rather than compared against a limit: a counter and a
    // comparison give a mutation that shifts the boundary by one and no test
    // can tell, while running out of a budget has one meaning.
    const MAX_LINK_HOPS: usize = 40;

    enum Step {
        Root(std::ffi::OsString),
        Up,
        Name(std::ffi::OsString),
    }

    fn push_reversed(path: &Path, pending: &mut Vec<Step>) {
        let steps: Vec<Step> = path
            .components()
            .filter_map(|c| match c {
                Component::CurDir => None,
                Component::ParentDir => Some(Step::Up),
                Component::RootDir | Component::Prefix(_) => Some(Step::Root(c.as_os_str().to_os_string())),
                Component::Normal(n) => Some(Step::Name(n.to_os_string())),
            })
            .collect();
        // Pushed in reverse so the walk pops them left to right, and so a
        // link's target is walked before whatever followed the link.
        pending.extend(steps.into_iter().rev());
    }

    let joined = if path.is_absolute() { path.to_path_buf() } else { root.join(path) };
    let mut pending: Vec<Step> = Vec::new();
    push_reversed(&joined, &mut pending);

    let mut out = PathBuf::new();
    let mut budget = MAX_LINK_HOPS;

    while let Some(step) = pending.pop() {
        match step {
            // An absolute link target restarts the walk, as it does for the
            // kernel: pushing a rooted path replaces what came before it.
            Step::Root(sep) => out.push(sep),
            Step::Up => {
                if !out.pop() {
                    return None;
                }
            }
            Step::Name(name) => {
                let candidate = out.join(&name);
                match std::fs::symlink_metadata(&candidate) {
                    Ok(md) if md.file_type().is_symlink() => {
                        budget = budget.checked_sub(1)?;
                        push_reversed(&std::fs::read_link(&candidate).ok()?, &mut pending);
                    }
                    // Either it exists and is not a link, or it does not exist
                    // yet: the name stands as written.
                    _ => out = candidate,
                }
            }
        }
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
