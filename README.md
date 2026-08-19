# Hobo Harness

A local multi-agent harness over Ollama. A graph engine walks a workflow, hands
documents between agents, and lets them read, change and test code through a
single dispatcher: every action classified, gated where it must be, journalled,
and reversible in one step from a checkpoint taken before the first write.

Rust workspace, no cloud — models are whatever Ollama has locally.

## Status: early, and the sandbox does not yet hold

Known and unfixed as of 2026-08-19, both proven by execution, both being fixed:

- commands issued by an agent bypass the permission floor — every one of them is
  wrapped in `bash -lc`, and the floor only looks at the program name;
- a write can leave the project root through a dangling symlink.

**Do not run this against anything you care about, and do not use `--yes`
outside a throwaway directory.**

## Running

    ./tools/ollama.sh up
    cargo test --release
    ./target/release/mrun workflows/analysis.json <project>

`mrun --yes` approves ordinary work without asking — see the note above before
trusting it.
