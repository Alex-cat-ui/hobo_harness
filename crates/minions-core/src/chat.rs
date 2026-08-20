//! The chat protocol Qwen 2.5 was actually trained on.
//!
//! Two things were wrong with sending one concatenated prompt to
//! `/api/generate`. The model expects a message structure — system separate
//! from task, tool results in their own role — and it was pre-trained with
//! function-calling templates, so an invented text syntax competes with what
//! it already knows.
//!
//! Native tool calls are the primary path. The text parser stays as a
//! fallback, because Qwen's own documentation states the protocol is not
//! guaranteed to be followed and advises countermeasures in production. When
//! the fallback fires it is recorded: how often that happens is a measure of
//! how well the pairing works, not a detail.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Sent in the shape the server speaks, not ours — see `wire` below.
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "wire::serialize_calls",
        deserialize_with = "wire::deserialize_calls"
    )]
    pub tool_calls: Vec<ToolCallRequest>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: Role::System, content: text.into(), tool_name: None, tool_calls: Vec::new() }
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: Role::User, content: text.into(), tool_name: None, tool_calls: Vec::new() }
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: text.into(), tool_name: None, tool_calls: Vec::new() }
    }
    /// An assistant turn that asked for tools. The calls belong *in* the turn:
    /// a transcript that drops them leaves the model unable to see what it
    /// asked for, and it re-asks. See `wire` for what was measured.
    pub fn assistant_with_calls(text: impl Into<String>, calls: Vec<ToolCallRequest>) -> Self {
        Self { role: Role::Assistant, content: text.into(), tool_name: None, tool_calls: calls }
    }
    /// A result going back to the model. The name matters: the model matches it
    /// against the call it made.
    pub fn tool_result(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: text.into(),
            tool_name: Some(name.into()),
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRequest {
    pub name: String,
    /// Left as a value rather than a typed struct: the model decides the shape,
    /// and a mismatch has to be reported to it rather than fail deserialisation.
    pub arguments: Value,
}

/// The wire shape of a tool call, which is not ours.
///
/// Ollama nests a call as `{"function": {"name", "arguments"}}` in both
/// directions; `ToolCallRequest` is flat because that is what the rest of the
/// code wants to hold. The difference is not cosmetic. Measured against Ollama
/// 0.15.6 and `qwen2.5:7b` on 2026-08-19, an assistant turn replayed with the
/// same call:
///
///   * nested — 73 prompt tokens, and asked which tool it had called the model
///     answers `read_file`;
///   * flat — 66 tokens, and the model answers `NONE`;
///   * no call at all — 54 tokens.
///
/// The flat form is therefore worse than silence: the template renders it as a
/// call to nothing, so the model is told it asked for something nameless.
mod wire {
    use super::ToolCallRequest;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use serde_json::Value;

    #[derive(Serialize, Deserialize)]
    struct Wrapped {
        function: Inner,
    }

    #[derive(Serialize, Deserialize)]
    struct Inner {
        name: String,
        #[serde(default)]
        arguments: Value,
    }

    pub fn serialize_calls<S: Serializer>(calls: &[ToolCallRequest], s: S) -> Result<S::Ok, S::Error> {
        let wrapped: Vec<Wrapped> = calls
            .iter()
            .map(|c| Wrapped { function: Inner { name: c.name.clone(), arguments: c.arguments.clone() } })
            .collect();
        wrapped.serialize(s)
    }

    pub fn deserialize_calls<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ToolCallRequest>, D::Error> {
        let wrapped = Vec::<Wrapped>::deserialize(d)?;
        Ok(wrapped
            .into_iter()
            .map(|w| ToolCallRequest { name: w.function.name, arguments: w.function.arguments })
            .collect())
    }
}

/// The tools an agent may call, as JSON Schema. One definition, used both to
/// tell the model what exists and to validate what comes back.
pub fn tool_schemas() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the project. Always read a file before changing it.",
                "parameters": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the project root" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Replace a file with new contents. Always send the WHOLE file, never a fragment and never a diff.",
                "parameters": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the project root" },
                        "content": { "type": "string", "description": "The entire new contents of the file" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": "Replace one fragment of a file. Use this when a whole file is too large to send. The fragment must appear exactly once.",
                "parameters": {
                    "type": "object",
                    "required": ["path", "find", "replace"],
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the project root" },
                        "find": { "type": "string", "description": "The existing text, quoted exactly, including indentation" },
                        "replace": { "type": "string", "description": "What it becomes" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "append_to_file",
                "description": "Add text to the end of a file. The natural way to add a new function without resending the file.",
                "parameters": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to the project root" },
                        "content": { "type": "string", "description": "The text to add at the end" }
                    }
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the project root, for example the test suite.",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": { "type": "string", "description": "The command line to run" }
                    }
                }
            }
        }),
    ]
}

/// Turns a model's call into the closed set the sandbox judges. Anything that
/// does not map is reported back to the model rather than guessed at.
pub fn to_tool_call(req: &ToolCallRequest) -> Result<crate::sandbox::ToolCall, String> {
    use crate::sandbox::ToolCall;
    use std::path::PathBuf;

    let arg = |k: &str| -> Option<String> {
        req.arguments.get(k).and_then(|v| v.as_str().map(|s| s.to_string()))
    };

    match req.name.as_str() {
        "read_file" => arg("path")
            .map(|p| ToolCall::ReadFile { path: PathBuf::from(p) })
            .ok_or_else(|| "read_file needs a `path` string".to_string()),
        "write_file" => {
            let path = arg("path").ok_or_else(|| "write_file needs a `path` string".to_string())?;
            let content = arg("content")
                .ok_or_else(|| "write_file needs a `content` string holding the whole file".to_string())?;
            Ok(ToolCall::WriteFile { path: PathBuf::from(path), content })
        }
        "replace_in_file" => {
            let path = arg("path").ok_or_else(|| "replace_in_file needs a `path` string".to_string())?;
            let find = arg("find")
                .ok_or_else(|| "replace_in_file needs a `find` string quoting the existing text".to_string())?;
            let replace = arg("replace")
                .ok_or_else(|| "replace_in_file needs a `replace` string".to_string())?;
            Ok(ToolCall::ReplaceInFile { path: PathBuf::from(path), find, replace })
        }
        "append_to_file" => {
            let path = arg("path").ok_or_else(|| "append_to_file needs a `path` string".to_string())?;
            let content = arg("content").ok_or_else(|| "append_to_file needs a `content` string".to_string())?;
            Ok(ToolCall::AppendToFile { path: PathBuf::from(path), content })
        }
        "run_command" => {
            let cmd = arg("command").ok_or_else(|| "run_command needs a `command` string".to_string())?;
            Ok(ToolCall::RunCommand { program: "bash".into(), args: vec!["-lc".into(), cmd] })
        }
        other => Err(format!(
            "`{other}` is not a tool that exists. The tools are read_file, write_file, replace_in_file, append_to_file and run_command."
        )),
    }
}

/// The command line an agent has to send to get this argv run. The inverse of
/// the wrapper `to_tool_call` puts on every `run_command`: the harness keeps its
/// own test command as an argv with a shell in front of it, and handing that to
/// a model as it stands would wrap it a second time.
pub fn command_line(argv: &[String]) -> String {
    const SHELLS: [&str; 6] = ["sh", "bash", "zsh", "dash", "ksh", "fish"];
    if let [program, flag, script] = argv {
        let base = std::path::Path::new(program)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(program)
            .to_ascii_lowercase();
        let carries_a_script = flag.starts_with('-') && !flag.starts_with("--") && flag.contains('c');
        if SHELLS.contains(&base.as_str()) && carries_a_script {
            return script.clone();
        }
    }
    argv.join(" ")
}

/// How a call reached us. Recorded because the mix is a measure of how well
/// the pairing works: `qwen2.5-coder:14b` declares the tools capability and
/// then writes its calls into the text instead of the channel, which is
/// exactly the unreliability Qwen's documentation warns about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallSource {
    Native,
    JsonInText,
    LineSyntax,
}

/// Recovers calls a model wrote as JSON prose instead of sending through the
/// tool channel. Observed on qwen2.5-coder:14b, which emits well-formed JSON
/// matching the schema it was given — the content is right, only the channel
/// is wrong, so throwing it away would waste a correct answer.
pub fn recover_from_text(text: &str) -> Vec<ToolCallRequest> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] != b'{' {
            i += 1;
            continue;
        }
        // Find the balanced object starting here, ignoring braces inside strings.
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        let mut end = None;
        for (j, ch) in text[i..].char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + j + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else { break };

        if let Ok(v) = serde_json::from_str::<Value>(&text[i..end]) {
            // Either {"name":..,"arguments":..} or the wrapped function form.
            let candidate = v.get("function").unwrap_or(&v);
            if let Some(name) = candidate.get("name").and_then(|n| n.as_str()) {
                let arguments = candidate.get("arguments").cloned().unwrap_or_else(|| json!({}));
                // `arguments` is sometimes a JSON string rather than an object.
                let arguments = match arguments.as_str() {
                    Some(inner) => serde_json::from_str(inner).unwrap_or(json!({})),
                    None => arguments,
                };
                out.push(ToolCallRequest { name: name.to_string(), arguments });
            }
        }
        i = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::ToolCall;

    fn req(name: &str, args: Value) -> ToolCallRequest {
        ToolCallRequest { name: name.into(), arguments: args }
    }

    #[test]
    fn every_schema_names_its_required_arguments() {
        for s in tool_schemas() {
            let f = &s["function"];
            let required = f["parameters"]["required"].as_array().expect("required list");
            assert!(!required.is_empty(), "{} declares no required argument", f["name"]);
            for r in required {
                let key = r.as_str().unwrap();
                assert!(
                    !f["parameters"]["properties"][key].is_null(),
                    "{} requires {key} but does not describe it",
                    f["name"]
                );
            }
        }
    }

    #[test]
    fn a_write_maps_to_the_sandbox_call() {
        let c = to_tool_call(&req("write_file", json!({"path": "src/a.py", "content": "x = 1\n"}))).unwrap();
        match c {
            ToolCall::WriteFile { path, content } => {
                assert!(path.ends_with("a.py"));
                assert_eq!(content, "x = 1\n");
            }
            other => panic!("expected a write, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_argument_is_reported_not_guessed() {
        let e = to_tool_call(&req("write_file", json!({"path": "src/a.py"}))).unwrap_err();
        assert!(e.contains("content"), "the message must name what is missing: {e}");
    }

    #[test]
    fn an_invented_tool_is_named_back_with_the_real_ones() {
        let e = to_tool_call(&req("delete_everything", json!({}))).unwrap_err();
        assert!(e.contains("delete_everything"));
        assert!(e.contains("read_file") && e.contains("write_file") && e.contains("run_command"));
    }

    #[test]
    fn a_command_becomes_a_shell_invocation_the_sandbox_can_judge() {
        // This test used to check only the shape, while its name claimed the
        // consequence — and the consequence was false: the sandbox saw the
        // program `bash` and never the script, so no command a model issued
        // reached the floor (finding 25). The claim is now made where it is
        // built, so the two cannot drift apart again.
        use crate::sandbox::{classify, GateReason, PermissionMode, Request, Verdict};

        let ordinary = to_tool_call(&req("run_command", json!({"command": "python3 -m unittest"}))).unwrap();
        match &ordinary {
            ToolCall::RunCommand { program, args } => {
                assert_eq!(program, "bash");
                assert_eq!(args, &vec!["-lc".to_string(), "python3 -m unittest".to_string()]);
            }
            other => panic!("expected a command, got {other:?}"),
        }

        let root = std::path::Path::new("/tmp/minions-root");
        let judge = |call: &ToolCall| {
            classify(&Request {
                call,
                root,
                mode: PermissionMode::DoNotAskInsideSandbox,
                consents: &[],
                node_gate: false,
                source_roots: &[],
            })
        };

        assert_eq!(
            judge(&ordinary),
            Verdict::Gated { reason: GateReason::Command, consentable: true },
            "running the tests must stay possible unattended"
        );

        let dangerous = to_tool_call(&req("run_command", json!({"command": "sudo rm -rf /"}))).unwrap();
        assert_eq!(
            judge(&dangerous),
            Verdict::Gated { reason: GateReason::PrivilegeEscalation, consentable: false },
            "the floor must read the script, not the word `bash`"
        );
    }

    #[test]
    fn a_call_written_as_json_prose_is_recovered() {
        // Exactly what qwen2.5-coder:14b produced when asked to read a file.
        let text = "{\n  \"name\": \"read_file\",\n  \"arguments\": {\n    \"path\": \"src/durations.py\"\n  }\n}";
        let calls = recover_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "src/durations.py");
    }

    #[test]
    fn the_wrapped_function_form_is_recovered_too() {
        let text = "Sure, I will do that.\n{\"type\":\"function\",\"function\":{\"name\":\"run_command\",\"arguments\":{\"command\":\"ls\"}}}";
        let calls = recover_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "run_command");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn arguments_given_as_a_json_string_are_unwrapped() {
        let text = "{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"a.py\\\"}\"}";
        let calls = recover_from_text(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["path"], "a.py");
    }

    #[test]
    fn braces_inside_strings_do_not_confuse_the_scan() {
        let text = "{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.py\",\"content\":\"d = {}\\nprint(d)\"}}";
        let calls = recover_from_text(text);
        assert_eq!(calls.len(), 1, "the scan stopped at a brace inside a string");
        assert!(calls[0].arguments["content"].as_str().unwrap().contains("d = {}"));
    }

    #[test]
    fn ordinary_prose_yields_no_calls() {
        assert!(recover_from_text("I have finished the work and the tests pass.").is_empty());
        assert!(recover_from_text("The result is {1, 2, 3} as expected.").is_empty());
    }

    #[test]
    fn a_call_is_sent_in_the_shape_the_server_speaks() {
        let m = Message::assistant_with_calls("", vec![req("read_file", json!({"path": "a.txt"}))]);
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(v["tool_calls"][0]["function"]["arguments"]["path"], "a.txt");
        assert!(
            v["tool_calls"][0].get("name").is_none(),
            "the flat shape renders as a call to nothing, which is worse than sending none"
        );
    }

    #[test]
    fn a_turn_with_calls_survives_a_round_trip() {
        let m = Message::assistant_with_calls("thinking", vec![req("run_command", json!({"command": "ls"}))]);
        let back: Message = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back.tool_calls, m.tool_calls, "what we send must be what we can read back");
        assert_eq!(back.content, "thinking");
    }

    #[test]
    fn a_turn_without_calls_carries_no_empty_field() {
        let v = serde_json::to_value(Message::assistant("just words")).unwrap();
        assert!(v.get("tool_calls").is_none());
    }

    #[test]
    fn a_shell_wrapped_command_is_shown_as_the_command_itself() {
        let argv = ["bash", "-lc", "python3 -m unittest discover -s tests"].map(String::from);
        assert_eq!(command_line(&argv), "python3 -m unittest discover -s tests");
        // And the way back: what run_command makes of that line is the argv we
        // started from, so the model is told a command it can actually send.
        let back = to_tool_call(&req("run_command", json!({"command": command_line(&argv)}))).unwrap();
        assert_eq!(back, ToolCall::RunCommand { program: "bash".into(), args: argv[1..].to_vec() });
    }

    #[test]
    fn a_command_that_is_not_a_shell_invocation_is_shown_whole() {
        let argv = ["cargo", "test", "--all"].map(String::from);
        assert_eq!(command_line(&argv), "cargo test --all");
    }

    #[test]
    fn a_tool_result_carries_the_name_the_model_used() {
        let m = Message::tool_result("read_file", "contents");
        assert_eq!(m.role, Role::Tool);
        assert_eq!(m.tool_name.as_deref(), Some("read_file"));
    }
}
