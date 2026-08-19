//! Running one agent node: prompt, stream, parse, retry with guidance, write.
//!
//! The retry loop is the mechanism SPEC requires in place of silent acceptance:
//! a malformed document goes back to the model with an instruction, at most
//! three times, and then the node fails so a human can supply it.

use crate::document::{self, Artifact, Document};
use crate::ollama::{Ollama, Options};
use crate::tokens::Tokenizer;
use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

pub const MAX_ATTEMPTS: u32 = 3;

pub struct Role {
    pub name: &'static str,
    pub model: String,
    pub window: u32,
    pub temperature: f32,
    pub artifact: Artifact,
    pub system: &'static str,
}

/// The format contract, appended to every role prompt. Kept in one place so a
/// change to the document format cannot drift from what roles are told.
pub fn format_contract(artifact: Artifact, run: &str, node: &str, attempt: u32, model: &str, inputs: &[String]) -> String {
    let keys = artifact.required_results();
    let results_block = if keys.is_empty() {
        "  (none required)".to_string()
    } else {
        keys.iter().map(|k| format!("  {k}: <number or single word>")).collect::<Vec<_>>().join("\n")
    };
    format!(
        "\n\nOUTPUT FORMAT — this is not optional, and the reply is rejected without it.\n\
         Reply with exactly one document and nothing else. No preamble, no closing remarks,\n\
         no code fence around the whole reply.\n\n\
         ---\n\
         artifact: {artifact}\n\
         run: {run}\n\
         node: {node}\n\
         attempt: {attempt}\n\
         model: {model}\n\
         created: <ISO 8601 timestamp>\n\
         inputs: [{inputs}]\n\
         results:\n{results}\n\
         digest: |\n\
         \x20 <two or three sentences of substance, with key numbers, at most 400 tokens>\n\
         ---\n\n\
         <the document body, in the structure described above>\n",
        artifact = artifact.name(),
        run = run,
        node = node,
        attempt = attempt,
        model = model,
        inputs = inputs.join(", "),
        results = results_block,
    )
}

pub struct NodeOutcome {
    pub document: Document,
    pub attempts_used: u32,
    pub path: PathBuf,
}

/// Runs one agent node to a parsed, written document.
pub async fn run_agent_node(
    ollama: &Ollama,
    role: &Role,
    run_id: &str,
    task: &str,
    inputs: &[String],
    run_dir: &Path,
    file_stem: &str,
    tok: &dyn Tokenizer,
    mut on_token: impl FnMut(&str),
    mut on_event: impl FnMut(&str),
) -> Result<NodeOutcome> {
    let mut correction: Option<String> = None;

    for attempt in 1..=MAX_ATTEMPTS {
        let mut prompt = String::new();
        prompt.push_str(role.system);
        prompt.push_str(&format_contract(role.artifact, run_id, role.name, attempt, &role.model, inputs));
        prompt.push_str("\n\nTASK\n");
        prompt.push_str(task);
        if let Some(c) = &correction {
            prompt.push_str("\n\nCORRECTION\n");
            prompt.push_str(c);
        }

        let opts = Options {
            num_ctx: role.window,
            num_predict: 1400,
            temperature: role.temperature,
            repeat_penalty: 1.1,
            seed: None,
        };

        on_event(&format!("attempt {attempt} of {MAX_ATTEMPTS} — {} on {}", role.name, role.model));
        let completion = ollama.generate(&role.model, &prompt, &opts, "300s", &mut on_token).await?;

        // Models sometimes wrap the whole reply in a fence despite instruction.
        let raw = strip_outer_fence(&completion.text);

        match document::parse(&raw, tok) {
            Ok(mut doc) => {
                stamp_harness_fields(&mut doc, run_id, role, attempt, inputs);
                let name = if attempt == 1 {
                    format!("{file_stem}.md")
                } else {
                    format!("{file_stem}.a{attempt}.md")
                };
                let path = run_dir.join(&name);
                std::fs::create_dir_all(run_dir)?;
                std::fs::write(&path, document::render(&doc))?;
                on_event(&format!("parsed and written: {name}"));
                return Ok(NodeOutcome { document: doc, attempts_used: attempt, path });
            }
            Err(e) => {
                on_event(&format!("rejected: {e}"));
                correction = Some(e.guidance());
            }
        }
    }

    Err(anyhow!(
        "node {} failed to produce a valid document in {MAX_ATTEMPTS} attempts — manual supply is the next step",
        role.name
    ))
}

/// Overwrites every header field the harness already knows.
///
/// A model cannot know the wall clock, and asking it produced a timestamp
/// hallucinated from training data — 2023 for a run in 2026. The same argument
/// applies to the run id, node name, attempt number, model name and input list:
/// the harness owns all of them, so whatever the model wrote is discarded
/// rather than trusted. The model is asked for the block anyway, because a
/// uniform document shape keeps parsing and guidance simple, and because the
/// act of filling it keeps the model oriented in the format.
///
/// What genuinely comes from the model is the artifact type, the results block,
/// the digest and the body.
pub fn stamp_harness_fields(doc: &mut Document, run_id: &str, role: &Role, attempt: u32, inputs: &[String]) {
    doc.header.run = run_id.to_string();
    doc.header.node = role.name.to_string();
    doc.header.attempt = attempt;
    doc.header.model = role.model.clone();
    doc.header.created = now_utc();
    doc.header.inputs = inputs.to_vec();
    doc.header.authored_by_human = false;
}

/// UTC in ISO 8601, from the system clock.
pub fn now_utc() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Strips a fence that wraps the entire reply, which some models add despite
/// being told not to. A fence inside the body is left alone.
pub fn strip_outer_fence(text: &str) -> String {
    let t = text.trim();
    if !t.starts_with("```") {
        return t.to_string();
    }
    let Some(first_nl) = t.find('\n') else { return t.to_string() };
    let after = &t[first_nl + 1..];
    match after.rfind("```") {
        Some(end) => after[..end].trim_end().to_string(),
        None => after.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_a_fence_wrapping_the_whole_reply() {
        let t = "```markdown\n---\nartifact: Plan\n---\n\nbody\n```";
        assert!(strip_outer_fence(t).starts_with("---"));
        assert!(!strip_outer_fence(t).contains("```"));
    }

    #[test]
    fn leaves_unfenced_text_alone() {
        assert_eq!(strip_outer_fence("---\nartifact: Plan\n"), "---\nartifact: Plan");
    }

    #[test]
    fn harness_fields_override_whatever_the_model_wrote() {
        use crate::document;
        use crate::tokens::CharRatioTokenizer;
        let text = "---\nartifact: Requirements\nrun: made-up\nnode: impostor\nattempt: 9\nmodel: gpt-9\ncreated: 2023-10-07T19:24:41Z\ninputs: [nonsense.md]\nresults:\n  requirements: 2\n  unknowns: 0\ndigest: |\n  Something short.\n---\n\nBody.\n";
        let mut doc = document::parse(text, &CharRatioTokenizer::default()).unwrap();
        let role = Role { name: "analyst", model: "qwen2.5:14b".into(), window: 8192, temperature: 0.5, artifact: Artifact::Requirements, system: "" };
        stamp_harness_fields(&mut doc, "2026-run", &role, 1, &["00_task.md".to_string()]);
        assert_eq!(doc.header.run, "2026-run");
        assert_eq!(doc.header.node, "analyst");
        assert_eq!(doc.header.attempt, 1);
        assert_eq!(doc.header.model, "qwen2.5:14b");
        assert_eq!(doc.header.inputs, vec!["00_task.md"]);
        assert!(!doc.header.created.starts_with("2023"), "the model's invented timestamp survived");
        // what the model legitimately owns is untouched
        assert_eq!(doc.header.results.get("requirements").unwrap(), "2");
        assert_eq!(doc.body, "Body.");
    }

    #[test]
    fn contract_names_every_required_results_key() {
        let c = format_contract(Artifact::Findings, "r", "reviewer", 1, "m", &[]);
        for k in Artifact::Findings.required_results() {
            assert!(c.contains(k), "contract omits {k}");
        }
    }
}
