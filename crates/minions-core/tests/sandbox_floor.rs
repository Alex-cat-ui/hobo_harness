//! The floor is a universal claim — "for any path, in any mode" — so it is
//! checked by generating cases rather than by listing a few.

use minions_core::sandbox::*;
use proptest::prelude::*;
use std::path::{Path, PathBuf};

fn modes() -> Vec<PermissionMode> {
    vec![
        PermissionMode::AskForEverything,
        PermissionMode::AskBeforeTouchingSource,
        PermissionMode::DoNotAskInsideSandbox,
    ]
}

fn req<'a>(
    call: &'a ToolCall,
    root: &'a Path,
    mode: PermissionMode,
    consents: &'a [ScopedConsent],
    node_gate: bool,
) -> Request<'a> {
    Request { call, root, mode, consents, node_gate, source_roots: &[] }
}

/// Path segments that cannot themselves climb out, so the generated path is
/// genuinely outside only because of the prefix we give it.
fn segment() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_]{1,12}"
}

proptest! {
    /// No mode, and no consent however broad, lets a write outside the root
    /// through without a gate.
    #[test]
    fn write_outside_root_always_gates(
        segs in prop::collection::vec(segment(), 1..5),
        mode_idx in 0usize..3,
        consent_everything in any::<bool>(),
    ) {
        let root = PathBuf::from("/tmp/minions-root");
        let outside: PathBuf = std::iter::once("/tmp/elsewhere".to_string())
            .chain(segs)
            .collect::<Vec<_>>()
            .join("/")
            .into();

        // A consent so broad it covers the filesystem must still not help.
        let consents = if consent_everything {
            vec![ScopedConsent { prefix: PathBuf::from("/") }]
        } else {
            vec![]
        };

        let call = ToolCall::WriteFile { path: outside, content: String::new() };
        let v = classify(&req(&call, &root, modes()[mode_idx], &consents, false));
        prop_assert!(v.is_gated(), "write outside root was not gated: {v:?}");
        prop_assert!(!v.is_allowed());
    }

    /// Traversal out of the root is resolved before comparison, so it lands on
    /// the same verdict as writing to the place it points at.
    #[test]
    fn traversal_cannot_reach_outside_unnoticed(
        depth in 1usize..4,
        mode_idx in 0usize..3,
    ) {
        // Deep enough that the generated climb never reaches above `/`, so the
        // path always resolves and the verdict can be pinned exactly. Climbing
        // past the filesystem root is a separate case, covered below.
        let root = PathBuf::from("/tmp/mroot/a/b/c");
        let mut p = root.clone();
        for _ in 0..depth {
            p.push("..");
        }
        p.push("escaped.txt");

        let call = ToolCall::WriteFile { path: p, content: String::new() };
        let v = classify(&req(&call, &root, modes()[mode_idx], &[], false));
        // Pinning the exact verdict matters: "not allowed" would also accept
        // Forbidden { UnresolvablePath }, which would hide a broken resolver.
        prop_assert_eq!(
            v,
            Verdict::Gated { reason: GateReason::WriteOutsideRoot, consentable: false },
            "traversal must resolve and then gate, not fail to resolve"
        );
    }

    /// Deletion is a floor category everywhere, and consent cannot grant it.
    #[test]
    fn delete_always_gates_and_is_never_consentable(
        segs in prop::collection::vec(segment(), 1..4),
        mode_idx in 0usize..3,
    ) {
        let root = PathBuf::from("/tmp/minions-root");
        let inside = root.join(segs.join("/"));
        let consents = vec![ScopedConsent { prefix: root.clone() }];
        let call = ToolCall::DeleteFile { path: inside };
        match classify(&req(&call, &root, modes()[mode_idx], &consents, false)) {
            Verdict::Gated { consentable, .. } => prop_assert!(!consentable),
            other => prop_assert!(false, "delete was not gated: {other:?}"),
        }
    }

    /// Reading outside the root is refused outright rather than gated: a gate
    /// implies a legitimate use, and the product has none.
    #[test]
    fn read_outside_root_is_forbidden_not_gated(
        segs in prop::collection::vec(segment(), 1..4),
        mode_idx in 0usize..3,
    ) {
        let root = PathBuf::from("/tmp/minions-root");
        let outside = PathBuf::from("/etc").join(segs.join("/"));
        let call = ToolCall::ReadFile { path: outside };
        let v = classify(&req(&call, &root, modes()[mode_idx], &[], false));
        prop_assert!(matches!(v, Verdict::Forbidden { .. }), "expected Forbidden, got {v:?}");
    }

    /// An explicit node gate is a specific instruction and outranks any mode.
    #[test]
    fn node_gate_beats_every_mode(
        segs in prop::collection::vec(segment(), 1..4),
        mode_idx in 0usize..3,
    ) {
        let root = PathBuf::from("/tmp/minions-root");
        let inside = root.join(segs.join("/"));
        let consents = vec![ScopedConsent { prefix: root.clone() }];
        let call = ToolCall::WriteFile { path: inside, content: String::new() };
        let v = classify(&req(&call, &root, modes()[mode_idx], &consents, true));
        prop_assert!(v.is_gated(), "node gate was swallowed by the mode: {v:?}");
    }

    /// Every command gates. Floor commands additionally cannot be consented.
    #[test]
    fn commands_always_gate(
        program in "[a-z]{2,10}",
        arg in "[a-z]{2,10}",
        mode_idx in 0usize..3,
    ) {
        let root = PathBuf::from("/tmp/minions-root");
        let call = ToolCall::RunCommand { program, args: vec![arg] };
        let v = classify(&req(&call, &root, modes()[mode_idx], &[], false));
        prop_assert!(v.is_gated(), "a command was allowed without a gate: {v:?}");
    }
}

#[test]
fn floor_commands_are_never_consentable() {
    let root = PathBuf::from("/tmp/minions-root");
    let cases: Vec<(&str, Vec<&str>)> = vec![
        ("sudo", vec!["rm"]),
        ("curl", vec!["https://example.com"]),
        ("git", vec!["push", "origin", "main"]),
        ("brew", vec!["install", "cowsay"]),
        ("npm", vec!["install", "left-pad"]),
        ("cargo", vec!["publish"]),
        ("/usr/bin/ssh", vec!["host"]),
    ];
    for (program, args) in cases {
        let call = ToolCall::RunCommand {
            program: program.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
        };
        for mode in [
            PermissionMode::AskForEverything,
            PermissionMode::AskBeforeTouchingSource,
            PermissionMode::DoNotAskInsideSandbox,
        ] {
            let consents = [ScopedConsent { prefix: PathBuf::from("/") }];
            let r = Request { call: &call, root: &root, mode, consents: &consents, node_gate: false, source_roots: &[] };
            match classify(&r) {
                Verdict::Gated { consentable, reason } => {
                    assert!(!consentable, "{program} {args:?} was consentable under {mode:?} ({reason:?})");
                }
                other => panic!("{program} {args:?} was not gated: {other:?}"),
            }
        }
    }
}

#[test]
fn a_symlink_pointing_out_of_the_root_does_not_slip_through() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("project");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(&outside).unwrap();

    let link = root.join("escape");
    std::os::unix::fs::symlink(&outside, &link).unwrap();

    let root_c = root.canonicalize().unwrap();
    let target = link.join("stolen.txt");
    let call = ToolCall::WriteFile { path: target, content: String::new() };

    for mode in [
        PermissionMode::AskForEverything,
        PermissionMode::AskBeforeTouchingSource,
        PermissionMode::DoNotAskInsideSandbox,
    ] {
        let r = Request { call: &call, root: &root_c, mode, consents: &[], node_gate: false, source_roots: &[] };
        let v = classify(&r);
        assert!(!v.is_allowed(), "symlink escape allowed under {mode:?}: {v:?}");
    }
}

#[test]
fn ordinary_writes_inside_the_root_follow_the_mode() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let call = ToolCall::WriteFile { path: root.join("src/main.rs"), content: String::new() };

    let ask = Request { call: &call, root: &root, mode: PermissionMode::AskForEverything, consents: &[], node_gate: false, source_roots: &[] };
    assert!(classify(&ask).is_gated());

    let permissive = Request { call: &call, root: &root, mode: PermissionMode::DoNotAskInsideSandbox, consents: &[], node_gate: false, source_roots: &[] };
    assert!(classify(&permissive).is_allowed());

    let src = vec![PathBuf::from("src")];
    let mid = Request { call: &call, root: &root, mode: PermissionMode::AskBeforeTouchingSource, consents: &[], node_gate: false, source_roots: &src };
    assert!(classify(&mid).is_gated(), "source write should gate in the middle mode");

    let call_test = ToolCall::WriteFile { path: root.join("Tests/x.swift"), content: String::new() };
    let mid2 = Request { call: &call_test, root: &root, mode: PermissionMode::AskBeforeTouchingSource, consents: &[], node_gate: false, source_roots: &src };
    assert!(classify(&mid2).is_allowed(), "non-source write should pass in the middle mode");
}

// ---- gaps found by mutation testing, 2026-08-16 ----

#[test]
fn resolve_keeps_a_nonexistent_tail_and_still_lands_inside() {
    // Guards the `while !existing.exists()` walk: a file being created does not
    // exist yet, and must still resolve to a path inside the root.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    std::fs::create_dir_all(root.join("src")).unwrap();

    let target = root.join("src/does/not/exist/yet.rs");
    let resolved = resolve(&target, &root).expect("must resolve");
    assert!(resolved.starts_with(&root), "resolved outside the root: {resolved:?}");
    assert!(resolved.ends_with("src/does/not/exist/yet.rs"), "tail was lost: {resolved:?}");

    let call = ToolCall::WriteFile { path: target, content: String::new() };
    let r = Request { call: &call, root: &root, mode: PermissionMode::DoNotAskInsideSandbox, consents: &[], node_gate: false, source_roots: &[] };
    assert_eq!(classify(&r), Verdict::Allowed, "a new file inside the root must be writable");
}

#[test]
fn resolve_refuses_to_climb_above_the_filesystem_root() {
    // Guards the `if !lexical.pop()` branch: exhausting the path must fail,
    // and succeeding must not.
    assert_eq!(resolve(Path::new("/../../.."), Path::new("/tmp")), None);

    // On macOS /tmp is itself a symlink to /private/tmp, and resolving it is the
    // point of the function — so the expectation is built the same way.
    let base = Path::new("/tmp").canonicalize().unwrap();
    let up = resolve(Path::new("/tmp/a/b/../c"), Path::new("/tmp"));
    assert_eq!(up, Some(base.join("a/c")), "a resolvable climb must resolve through the symlink");
}

#[test]
fn scoped_consent_actually_grants_an_ordinary_write() {
    // Without this, a consent check that never matches would pass every test:
    // the suite only proved consent does not help where it must not.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().canonicalize().unwrap();
    let target = root.join("Tests/generated.swift");
    let call = ToolCall::WriteFile { path: target, content: String::new() };

    let without = Request { call: &call, root: &root, mode: PermissionMode::AskForEverything, consents: &[], node_gate: false, source_roots: &[] };
    assert!(classify(&without).is_gated(), "precondition: this write gates without consent");

    let consents = [ScopedConsent { prefix: root.join("Tests") }];
    let with = Request { call: &call, root: &root, mode: PermissionMode::AskForEverything, consents: &consents, node_gate: false, source_roots: &[] };
    assert_eq!(classify(&with), Verdict::Allowed, "consent on Tests/ must grant a write under it");

    // ...and only under it.
    let elsewhere = ToolCall::WriteFile { path: root.join("Sources/main.swift"), content: String::new() };
    let narrow = Request { call: &elsewhere, root: &root, mode: PermissionMode::AskForEverything, consents: &consents, node_gate: false, source_roots: &[] };
    assert!(classify(&narrow).is_gated(), "consent on Tests/ must not reach Sources/");
}

#[test]
fn is_gated_distinguishes_verdicts() {
    // A helper that always answered "yes" would satisfy every other test here.
    assert!(!Verdict::Allowed.is_gated());
    assert!(!Verdict::Forbidden { reason: ForbidReason::ReadOutsideRoot }.is_gated());
    assert!(Verdict::Gated { reason: GateReason::Delete, consentable: false }.is_gated());
    assert!(Verdict::Allowed.is_allowed());
    assert!(!Verdict::Gated { reason: GateReason::Delete, consentable: false }.is_allowed());
}
