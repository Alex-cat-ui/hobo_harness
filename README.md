# Hobo Harness

A local multi-agent harness over Ollama. A graph engine walks a workflow, hands
documents between agents, and lets them read, change and test code through a
single dispatcher: every action classified, gated where it must be, journalled,
and reversible in one step from a checkpoint taken before the first write.

Rust workspace, no cloud — models are whatever Ollama has locally.

## Status: early — the sandbox filters, it does not confine

Both holes listed here on 2026-08-19 are closed, and each was proven closed the
way it was found — by execution:

- the floor reads the shell script an agent asks to run, so deletion, network
  fetches, package installs, redirection into a file and repository rewrites
  reach it. They used to arrive as the word `bash` and pass as ordinary work;
- a path is resolved component by component, so a write cannot leave the project
  root through a symlink, dangling or otherwise.

What is still true, and worth knowing before pointing this at anything:

- it is a filter over what a model writes, not a boundary. The operating system
  does not confine the command — the shell runs with your rights, and a program
  name assembled at runtime cannot be read;
- `.minions/` — the run's journal and the checkpoint it is rolled back to —
  lives inside the project and is writable by the agent;
- a refused call is not journalled, so the record says what was done and not
  what was declined.

As of 2026-08-20 a development workflow produced a working change for the first
time: on a toy project, one function and one test, the suite green and nothing
lost. The reviewer that passed it reported a defect that does not reproduce and
missed one that does.

Later the same day it stopped being able to claim success over failing tests. The
tests are a node in the graph again; the harness measures them itself before and
after; a role that changes code cannot close its document while they are red; and
the report says where they moved — `Tests: was 3/4 passing, now 4/4 passing`. Of
three runs that evening, two ended in a refusal that named its reason and none
reported a success it had not earned. What the harness still cannot do is the
work: the coder writes a whole file, breaks it in the first line, and edits on
blind from there.

**Prefer a throwaway directory, and read the above before using `--yes`.**

## Running

    ./tools/ollama.sh up
    cargo test --release
    ./target/release/mrun workflows/analysis.json <project>

`mrun --yes` approves ordinary work without asking — see the note above before
trusting it.
