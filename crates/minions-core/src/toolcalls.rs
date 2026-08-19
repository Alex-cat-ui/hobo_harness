//! Line-oriented tool calls.
//!
//! Not JSON. A 14B mangles nested JSON often enough that the parser would spend
//! more attempts on punctuation than on work, and every mangled call costs a
//! model round trip. Line prefixes with an explicit block terminator survive
//! what JSON does not.
//!
//! ```text
//! TOOL read src/durations.py
//! TOOL run python3 -m unittest discover -s tests
//! TOOL write src/durations.py
//! <<<FILE
//! ...the whole file...
//! FILE>>>
//! TOOL done
//! ```

use crate::sandbox::ToolCall;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    Call(ToolCall),
    /// The agent says it has finished acting and the rest is its document.
    Done,
}

/// Extracts the calls in the order the model wrote them.
pub fn parse(reply: &str) -> Vec<Step> {
    let mut steps = Vec::new();
    let lines: Vec<&str> = reply.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();
        let Some(rest) = line.strip_prefix("TOOL ") else {
            i += 1;
            continue;
        };
        let rest = rest.trim();

        if rest.eq_ignore_ascii_case("done") {
            steps.push(Step::Done);
            i += 1;
            continue;
        }

        if let Some(path) = rest.strip_prefix("read ") {
            steps.push(Step::Call(ToolCall::ReadFile { path: PathBuf::from(path.trim()) }));
            i += 1;
            continue;
        }

        if let Some(cmd) = rest.strip_prefix("run ") {
            let cmd = cmd.trim();
            if !cmd.is_empty() {
                steps.push(Step::Call(ToolCall::RunCommand {
                    program: "bash".into(),
                    args: vec!["-lc".into(), cmd.to_string()],
                }));
            }
            i += 1;
            continue;
        }

        if let Some(path) = rest.strip_prefix("write ") {
            let path = path.trim().to_string();
            // The body runs to the terminator. Anything before <<<FILE is the
            // model talking to itself and is discarded.
            let mut body = String::new();
            let mut j = i + 1;
            while j < lines.len() && !lines[j].trim().starts_with("<<<FILE") {
                j += 1;
            }
            j += 1;
            while j < lines.len() && !lines[j].trim().starts_with("FILE>>>") {
                body.push_str(lines[j]);
                body.push('\n');
                j += 1;
            }
            if j < lines.len() {
                steps.push(Step::Call(ToolCall::WriteFile { path: PathBuf::from(path), content: body }));
                i = j + 1;
                continue;
            }
            // No terminator: an unfinished write is not performed, because
            // half a file is worse than none.
            i += 1;
            continue;
        }

        i += 1;
    }
    steps
}

/// What the agent is told it may do. Kept in one place so the syntax the parser
/// accepts and the syntax the model is shown cannot drift apart.
pub const TOOL_INSTRUCTIONS: &str = r#"
HOW TO ACT

You work in steps. In each reply, do ONE of these two things.

Either issue tool calls, one per line, and nothing else:

  TOOL read <path>
  TOOL run <shell command>
  TOOL write <path>
  <<<FILE
  the entire new contents of the file
  FILE>>>

Or, when the work is finished, reply with your document and no TOOL lines.

Rules that matter:
- `TOOL write` replaces the WHOLE file. Never write a fragment or a diff.
  Read the file first, then write it back complete with your change in it.
- After changing code, run the tests before you finish.
- One step at a time. Do not plan five steps ahead in one reply.
- Paths are relative to the project root.
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_writes_and_runs_are_parsed_in_order() {
        let reply = "Let me look first.\nTOOL read src/a.py\nTOOL run python3 -m unittest\n";
        let steps = parse(reply);
        assert_eq!(steps.len(), 2);
        assert!(matches!(&steps[0], Step::Call(ToolCall::ReadFile { path }) if path.ends_with("a.py")));
        assert!(matches!(&steps[1], Step::Call(ToolCall::RunCommand { .. })));
    }

    #[test]
    fn a_write_carries_the_whole_body() {
        let reply = "TOOL write src/a.py\n<<<FILE\nline one\nline two\nFILE>>>\n";
        match &parse(reply)[0] {
            Step::Call(ToolCall::WriteFile { path, content }) => {
                assert!(path.ends_with("a.py"));
                assert_eq!(content, "line one\nline two\n");
            }
            other => panic!("expected a write, got {other:?}"),
        }
    }

    #[test]
    fn an_unterminated_write_is_not_performed() {
        // Half a file is worse than none: it would pass the sandbox and leave
        // the project broken in a way the agent believes succeeded.
        let reply = "TOOL write src/a.py\n<<<FILE\nline one\n";
        assert!(parse(reply).is_empty(), "an unterminated write must be dropped");
    }

    #[test]
    fn chatter_around_the_calls_is_ignored() {
        let reply = "I will now read the file.\n\nTOOL read src/a.py\n\nThat should tell me what I need.";
        assert_eq!(parse(reply).len(), 1);
    }

    #[test]
    fn done_is_recognised() {
        assert_eq!(parse("TOOL done").len(), 1);
        assert!(matches!(parse("TOOL done")[0], Step::Done));
    }

    #[test]
    fn a_reply_with_no_calls_yields_nothing() {
        assert!(parse("---\nartifact: Patch\n---\n\nAll finished.").is_empty());
    }

    #[test]
    fn the_instructions_show_exactly_what_the_parser_accepts() {
        // The syntax the model is shown and the syntax the parser takes must
        // not drift apart.
        for marker in ["TOOL read", "TOOL run", "TOOL write", "<<<FILE", "FILE>>>"] {
            assert!(TOOL_INSTRUCTIONS.contains(marker), "the instructions omit {marker}");
        }
    }
}
