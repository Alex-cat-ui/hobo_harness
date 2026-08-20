//! Measuring what a model can do, once, so the engine can adapt to it.
//!
//! The probe never executes anything. It only looks at what the model *asks
//! for*, which is all the engine needs to choose a protocol and means a probe
//! can never touch a project.

use crate::backend::{ModelBackend, TokenSink};
use crate::chat::{self, Message};
use crate::document;
use crate::ollama::Options;
use crate::profile::{EditStyle, ModelProfile, ToolChannel};
use crate::tokens::Tokenizer;
use anyhow::Result;

struct Silent;
impl TokenSink for Silent {
    fn token(&mut self, _t: &str) {}
}

fn opts(window: u32) -> Options {
    Options { num_ctx: window, num_predict: 900, temperature: 0.0, repeat_penalty: 1.1, seed: Some(17) }
}

/// One exchange, returning what came back and by which route.
async fn ask(
    backend: &dyn ModelBackend,
    model: &str,
    system: &str,
    user: &str,
    with_tools: bool,
) -> Result<(String, Vec<chat::ToolCallRequest>, ToolChannel)> {
    let messages = vec![Message::system(system), Message::user(user)];
    let tools = if with_tools { chat::tool_schemas() } else { Vec::new() };
    let mut sink = Silent;
    let reply = backend.chat(model, &messages, tools, &opts(8192), "120s", &mut sink).await?;

    if !reply.tool_calls.is_empty() {
        return Ok((reply.text, reply.tool_calls, ToolChannel::Native));
    }
    let recovered = chat::recover_from_text(&reply.text);
    if !recovered.is_empty() {
        return Ok((reply.text, recovered, ToolChannel::JsonInText));
    }
    Ok((reply.text, Vec::new(), ToolChannel::TextOnly))
}

/// Six short exchanges. About a minute per model, once per binding.
pub async fn measure(
    backend: &dyn ModelBackend,
    model: &str,
    tok: &dyn Tokenizer,
    now: &str,
    mut on_step: impl FnMut(&str),
) -> Result<ModelProfile> {
    let _ = tok;
    let mut p = ModelProfile::unmeasured(model);
    p.measured_at = now.to_string();
    let mut longest = 0usize;

    // 1. Does it call a tool at all, and by which route?
    on_step("channel: asking for a single file read");
    let (text, calls, channel) = ask(
        backend,
        model,
        "You change files in a project. Use the tools you are given.",
        "Read the file src/main.py so you can see what is in it.",
        true,
    )
    .await?;
    longest = longest.max(text.len());
    p.channel = channel;
    let called_read = calls.iter().any(|c| c.name == "read_file");
    on_step(&format!("channel: {:?}, read_file requested: {called_read}", p.channel));

    // 2. Several calls in one turn.
    on_step("parallel: asking for two file reads at once");
    let (text, calls, _) = ask(
        backend,
        model,
        "You change files in a project. Use the tools you are given.",
        "Read both src/main.py and tests/test_main.py. Do it in one reply.",
        true,
    )
    .await?;
    longest = longest.max(text.len());
    p.parallel_calls = calls.len() >= 2;
    on_step(&format!("parallel: {} call(s) in one turn", calls.len()));

    // 3. Can it deliver a whole file in one argument? This is a capability, not
    //    a preference: a model that cannot is not a model that cannot edit.
    on_step("edit style: asking for a whole small file");
    let (text, calls, _) = ask(
        backend,
        model,
        "You change files in a project. Use the tools you are given. When you write a file, send its entire contents.",
        "Create src/greet.py containing a single function greet(name) that returns the string 'hello ' followed by the name. Write the whole file.",
        true,
    )
    .await?;
    longest = longest.max(text.len());
    let whole_file_ok = calls.iter().any(|c| {
        c.name == "write_file"
            && c.arguments
                .get("content")
                .and_then(|v| v.as_str())
                .map(|s| s.contains("def greet"))
                .unwrap_or(false)
    });
    p.edit_style = if whole_file_ok { EditStyle::WholeFile } else { EditStyle::Replacement };
    on_step(&format!("edit style: {:?}", p.edit_style));

    // 4. Does it hold the document format without being corrected?
    on_step("format: asking for a document with our header");
    let contract = crate::node::format_contract(document::Artifact::Requirements, "probe", "probe", 1, model, &[]);
    let (text, _, _) = ask(
        backend,
        model,
        &format!("You are an analyst. Turn a task into requirements.{contract}"),
        "TASK\nAdd a function that formats a number of seconds as a duration string.",
        false,
    )
    .await?;
    longest = longest.max(text.len());
    p.holds_format = document::parse(&crate::node::strip_outer_fence(&text), tok).is_ok();
    on_step(&format!("format held first time: {}", p.holds_format));

    p.max_output_chars = longest;
    on_step(&format!("longest reply seen: {longest} characters"));
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ReplayBackend;
    use crate::tokens::CharRatioTokenizer;

    fn doc() -> String {
        "---\nartifact: Requirements\nrun: r\nnode: n\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n  requirements: 1\n  unknowns: 0\ndigest: |\n  Short.\n---\n\nStatement: s.\n\nRequirements:\n1. one\n\nOut of scope:\n- nothing\n\nBody.\n".to_string()
    }

    #[tokio::test]
    async fn a_model_using_the_channel_is_profiled_as_native() {
        let backend = ReplayBackend::new([
            r#"NATIVE:{"name":"read_file","arguments":{"path":"src/main.py"}}"#.to_string(),
            r#"NATIVE:{"name":"read_file","arguments":{"path":"a"}} {"name":"read_file","arguments":{"path":"b"}}"#.to_string(),
            r#"NATIVE:{"name":"write_file","arguments":{"path":"src/greet.py","content":"def greet(name):\n    return 'hello ' + name\n"}}"#.to_string(),
            doc(),
        ]);
        let tok = CharRatioTokenizer::default();
        let p = measure(&backend, "stub", &tok, "now", |_| {}).await.unwrap();
        assert_eq!(p.channel, ToolChannel::Native);
        assert!(p.parallel_calls, "two calls in one turn were scripted");
        assert!(p.send_schemas(), "a proven channel may be sent schemas");
    }

    #[tokio::test]
    async fn a_model_answering_in_json_text_is_profiled_as_such() {
        let backend = ReplayBackend::new([
            r#"{"name":"read_file","arguments":{"path":"src/main.py"}}"#.to_string(),
            r#"{"name":"read_file","arguments":{"path":"src/main.py"}}"#.to_string(),
            r#"{"name":"write_file","arguments":{"path":"src/greet.py","content":"def greet(name):\n    return 'hello ' + name\n"}}"#.to_string(),
            doc(),
        ]);
        let tok = CharRatioTokenizer::default();
        let p = measure(&backend, "stub", &tok, "now", |_| {}).await.unwrap();

        assert_eq!(p.channel, ToolChannel::JsonInText);
        assert!(!p.parallel_calls, "one call per turn was scripted");
        assert_eq!(p.edit_style, EditStyle::WholeFile, "it delivered a whole file");
        assert!(p.holds_format);
        assert!(p.is_measured());
        assert!(p.max_output_chars > 0);
    }

    #[tokio::test]
    async fn a_model_that_never_calls_a_tool_gets_the_weakest_profile() {
        let backend = ReplayBackend::new([
            "I would read the file.".to_string(),
            "I would read both files.".to_string(),
            "Here is the code you asked for.".to_string(),
            "Requirements: it should work well.".to_string(),
        ]);
        let tok = CharRatioTokenizer::default();
        let p = measure(&backend, "stub", &tok, "now", |_| {}).await.unwrap();

        assert_eq!(p.channel, ToolChannel::TextOnly);
        assert_eq!(p.edit_style, EditStyle::Replacement, "an unproven whole-file ability must not be assumed");
        assert!(!p.holds_format);
        assert!(!p.send_schemas());
        assert_eq!(p.document_attempts(), 5, "a model that loses the format gets more attempts");
    }

    #[tokio::test]
    async fn a_truncated_write_does_not_count_as_whole_file_ability() {
        // The exact failure watched live: write_file arriving without usable
        // content. Counting it as a capability would keep handing the model a
        // tool it cannot use.
        let backend = ReplayBackend::new([
            r#"{"name":"read_file","arguments":{"path":"src/main.py"}}"#.to_string(),
            r#"{"name":"read_file","arguments":{"path":"a"}}"#.to_string(),
            r#"{"name":"write_file","arguments":{"path":"src/greet.py"}}"#.to_string(),
            doc(),
        ]);
        let tok = CharRatioTokenizer::default();
        let p = measure(&backend, "stub", &tok, "now", |_| {}).await.unwrap();
        assert_eq!(p.edit_style, EditStyle::Replacement);
    }
}
