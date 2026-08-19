//! Ollama client.
//!
//! Generation parameters are passed per request. No `OLLAMA_*` environment
//! variable is ever set, so the user's own `ollama run` behaviour is untouched
//! (SPEC FR-4).

use crate::chat::{Message, ToolCallRequest};
use anyhow::{bail, Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;

pub const DEFAULT_BASE: &str = "http://127.0.0.1:11434";

#[derive(Debug, Clone, Serialize)]
pub struct Options {
    pub num_ctx: u32,
    pub num_predict: i32,
    pub temperature: f32,
    pub repeat_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

impl Default for Options {
    fn default() -> Self {
        // The Ollama default window of 4096 cannot hold an artifact plus a
        // system prompt plus an answer (SPEC §12), so it is never inherited.
        Self { num_ctx: 16384, num_predict: 512, temperature: 0.2, repeat_penalty: 1.1, seed: None }
    }
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: &'a Options,
    keep_alive: &'a str,
}

#[derive(Debug, Deserialize)]
struct GenerateChunk {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: u32,
    pub eval_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[serde(default)]
    pub size: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadedModel {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
}

#[derive(Debug, Deserialize)]
struct ListResponse {
    #[serde(default)]
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct PsResponse {
    #[serde(default)]
    models: Vec<LoadedModel>,
}

pub struct Ollama {
    base: String,
    http: reqwest::Client,
}

impl Ollama {
    pub fn new(base: impl Into<String>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .context("building http client")?;
        Ok(Self { base: base.into(), http })
    }

    pub fn local() -> Result<Self> {
        Self::new(DEFAULT_BASE)
    }

    pub async fn reachable(&self) -> bool {
        self.http
            .get(format!("{}/api/tags", self.base))
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    pub async fn list(&self) -> Result<Vec<ModelInfo>> {
        let r: ListResponse = self
            .http
            .get(format!("{}/api/tags", self.base))
            .send()
            .await
            .context("GET /api/tags")?
            .json()
            .await
            .context("decoding /api/tags")?;
        Ok(r.models)
    }

    /// Models currently resident, with the memory Ollama reports for each.
    pub async fn loaded(&self) -> Result<Vec<LoadedModel>> {
        let r: PsResponse = self
            .http
            .get(format!("{}/api/ps", self.base))
            .send()
            .await
            .context("GET /api/ps")?
            .json()
            .await
            .context("decoding /api/ps")?;
        Ok(r.models)
    }

    /// Ask Ollama to release a model immediately.
    pub async fn unload(&self, model: &str) -> Result<()> {
        let body = serde_json::json!({ "model": model, "prompt": "", "keep_alive": 0 });
        self.http
            .post(format!("{}/api/generate", self.base))
            .json(&body)
            .send()
            .await
            .context("unload request")?;
        Ok(())
    }

    /// Streaming generation. `on_token` is called for every chunk as it arrives,
    /// which is what the Bridge renders as a live token stream.
    pub async fn generate<F>(
        &self,
        model: &str,
        prompt: &str,
        opts: &Options,
        keep_alive: &str,
        mut on_token: F,
    ) -> Result<Completion>
    where
        F: FnMut(&str),
    {
        let req = GenerateRequest { model, prompt, stream: true, options: opts, keep_alive };
        let resp = self
            .http
            .post(format!("{}/api/generate", self.base))
            .json(&req)
            .send()
            .await
            .context("POST /api/generate")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("ollama returned {status}: {body}");
        }

        let mut out = Completion::default();
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.context("reading stream")?);
            // NDJSON: one complete JSON object per line.
            while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = &line[..line.len() - 1];
                if line.is_empty() {
                    continue;
                }
                let c: GenerateChunk =
                    serde_json::from_slice(line).context("decoding stream chunk")?;
                if !c.response.is_empty() {
                    on_token(&c.response);
                    out.text.push_str(&c.response);
                }
                if c.done {
                    out.prompt_tokens = c.prompt_eval_count.unwrap_or(0);
                    out.eval_tokens = c.eval_count.unwrap_or(0);
                }
            }
        }
        Ok(out)
    }
}

// ---- the chat endpoint, which is what Qwen 2.5 was trained on ----

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    options: &'a Options,
    keep_alive: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Value>,
}

#[derive(Debug, Deserialize, Default)]
struct ChatChunkMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<RawToolCall>,
}

#[derive(Debug, Deserialize)]
struct RawToolCall {
    function: RawFunction,
}

#[derive(Debug, Deserialize)]
struct RawFunction {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize, Default)]
struct ChatChunk {
    #[serde(default)]
    message: ChatChunkMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatReply {
    pub text: String,
    pub tool_calls: Vec<ToolCallRequest>,
    pub prompt_tokens: u32,
    pub eval_tokens: u32,
}

impl Ollama {
    /// Streaming chat with optional native tool calling.
    ///
    /// Tool calls arrive as structured data rather than as prose to be parsed,
    /// which removes an entire class of failure: the model cannot get the
    /// punctuation of a call wrong when it is not writing punctuation.
    pub async fn chat<F>(
        &self,
        model: &str,
        messages: &[Message],
        tools: Vec<Value>,
        opts: &Options,
        keep_alive: &str,
        mut on_token: F,
    ) -> Result<ChatReply>
    where
        F: FnMut(&str),
    {
        let req = ChatRequest { model, messages, stream: true, options: opts, keep_alive, tools };
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base))
            .json(&req)
            .send()
            .await
            .context("POST /api/chat")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            bail!("ollama returned {status}: {body}");
        }

        let mut out = ChatReply::default();
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();

        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk.context("reading stream")?);
            while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = &line[..line.len() - 1];
                if line.is_empty() {
                    continue;
                }
                let c: ChatChunk = serde_json::from_slice(line).context("decoding chat chunk")?;
                if !c.message.content.is_empty() {
                    on_token(&c.message.content);
                    out.text.push_str(&c.message.content);
                }
                for t in c.message.tool_calls {
                    out.tool_calls.push(ToolCallRequest { name: t.function.name, arguments: t.function.arguments });
                }
                if c.done {
                    out.prompt_tokens = c.prompt_eval_count.unwrap_or(0);
                    out.eval_tokens = c.eval_count.unwrap_or(0);
                }
            }
        }
        Ok(out)
    }
}
