//! The model boundary.
//!
//! This trait exists so the engine can run against recorded fixtures, not so
//! the product can court other providers. That distinction is deliberate:
//! designing for hypothetical providers would flatten the request type toward
//! a lowest common denominator and cost us the per-request window and
//! residency control the memory budget depends on.

use crate::chat::{Message, ToolCallRequest};
use crate::ollama::{ChatReply, Completion, Ollama, Options};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;

/// Where streamed tokens go. A trait rather than a closure because a
/// `FnMut(&str)` cannot cross an `async_trait` boundary without the borrow
/// escaping; the blanket impl below keeps closures ergonomic at call sites.
pub trait TokenSink: Send {
    fn token(&mut self, t: &str);
}

impl<F: FnMut(&str) + Send> TokenSink for F {
    fn token(&mut self, t: &str) {
        self(t)
    }
}

#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// The structured path: message roles and native tool calls, which is what
    /// the model was trained on.
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Vec<serde_json::Value>,
        opts: &Options,
        keep_alive: &str,
        on_token: &mut dyn TokenSink,
    ) -> Result<ChatReply>;

    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &Options,
        keep_alive: &str,
        on_token: &mut dyn TokenSink,
    ) -> Result<Completion>;
}

#[async_trait]
impl ModelBackend for Ollama {
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Vec<serde_json::Value>,
        opts: &Options,
        keep_alive: &str,
        on_token: &mut dyn TokenSink,
    ) -> Result<ChatReply> {
        Ollama::chat(self, model, messages, tools, opts, keep_alive, |t| on_token.token(t)).await
    }

    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &Options,
        keep_alive: &str,
        on_token: &mut dyn TokenSink,
    ) -> Result<Completion> {
        Ollama::generate(self, model, prompt, opts, keep_alive, |t| on_token.token(t)).await
    }
}

/// Replays scripted replies. Every logic test in the suite runs against this,
/// so a test run needs no model, no memory and no network, and never varies.
///
/// A reply prefixed with `NATIVE:` is delivered through the tool channel; any
/// other reply is delivered as plain text. The distinction has to be scriptable
/// because it is real: qwen2.5:14b uses the channel and qwen2.5-coder:14b
/// writes the same JSON into the text. A stub that recovered calls from text
/// itself would hide exactly the difference the engine has to detect.
pub struct ReplayBackend {
    replies: Mutex<VecDeque<String>>,
    seen: Mutex<Vec<String>>,
}

impl ReplayBackend {
    pub fn new(replies: impl IntoIterator<Item = String>) -> Self {
        Self { replies: Mutex::new(replies.into_iter().collect()), seen: Mutex::new(Vec::new()) }
    }

    /// Prompts the engine actually sent, for asserting on context assembly.
    pub fn prompts(&self) -> Vec<String> {
        self.seen.lock().unwrap().clone()
    }

    pub fn remaining(&self) -> usize {
        self.replies.lock().unwrap().len()
    }
}

#[async_trait]
impl ModelBackend for ReplayBackend {
    /// Replays the same scripted text through the chat shape, recovering any
    /// tool call written into it — so a test can script either kind of reply.
    async fn chat(
        &self,
        _model: &str,
        messages: &[Message],
        _tools: Vec<serde_json::Value>,
        _opts: &Options,
        _keep_alive: &str,
        on_token: &mut dyn TokenSink,
    ) -> Result<ChatReply> {
        let rendered = messages
            .iter()
            .map(|m| {
                // Calls are rendered too: a test that cannot see them cannot
                // prove the turn carried them.
                let calls: Vec<&str> = m.tool_calls.iter().map(|c| c.name.as_str()).collect();
                match calls.is_empty() {
                    true => format!("[{:?}] {}", m.role, m.content),
                    false => format!("[{:?}] {} <calls: {}>", m.role, m.content, calls.join(", ")),
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        self.seen.lock().unwrap().push(rendered);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("the replay backend ran out of scripted replies"))?;
        for chunk in reply.as_bytes().chunks(64) {
            on_token.token(&String::from_utf8_lossy(chunk));
        }
        match reply.strip_prefix("NATIVE:") {
            Some(body) => {
                let tool_calls = crate::chat::recover_from_text(body);
                Ok(ChatReply { text: String::new(), tool_calls, prompt_tokens: 0, eval_tokens: 0 })
            }
            None => Ok(ChatReply { text: reply, tool_calls: Vec::new(), prompt_tokens: 0, eval_tokens: 0 }),
        }
    }

    async fn generate(
        &self,
        _model: &str,
        prompt: &str,
        _opts: &Options,
        _keep_alive: &str,
        on_token: &mut dyn TokenSink,
    ) -> Result<Completion> {
        self.seen.lock().unwrap().push(prompt.to_string());
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("the replay backend ran out of scripted replies"))?;
        // Streamed in pieces so callers exercise the same path as a live model.
        for chunk in reply.as_bytes().chunks(64) {
            on_token.token(&String::from_utf8_lossy(chunk));
        }
        Ok(Completion { text: reply, prompt_tokens: 0, eval_tokens: 0 })
    }
}
