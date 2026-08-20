//! The clarification ladder.
//!
//! An agent that does not understand its task must not invent an answer, and
//! must not interrupt the human before it has tried. The ladder is: write the
//! unknowns down, try twice to resolve them from what is already available,
//! and only then ask — with the answers kept for every later run, so the same
//! question is not asked tomorrow.

use crate::document::Document;
use anyhow::Result;
use std::path::Path;

pub const SELF_RESOLUTION_PASSES: u32 = 2;

/// Who answers a clarification. A trait for the same reason the gate authority
/// is one: the product asks in the interface, tests answer from a script.
pub trait ClarificationAuthority: Send + Sync {
    fn ask(&self, node: &str, questions: &[String]) -> Vec<String>;
}

/// How many unknowns the document declares. `none`, an empty value and a
/// missing field all mean zero; anything unparseable is treated as one, on the
/// principle that an unreadable count is itself something to ask about.
pub fn unknown_count(doc: &Document) -> u32 {
    match doc.header.results.get("unknowns") {
        None => 0,
        Some(v) => {
            let v = v.trim();
            if v.is_empty() || v.eq_ignore_ascii_case("none") || v == "0" {
                0
            } else {
                v.parse().unwrap_or(1)
            }
        }
    }
}

/// Pulls the questions out of the body's Unknowns section.
///
/// Prose is parsed here, which the design avoids everywhere else — but the
/// count that decides whether to parse at all comes from `results`, so a
/// misread body can only ever cost a badly worded question, never a wrong
/// route.
pub fn extract_questions(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in body.lines() {
        let t = line.trim();
        let lower = t.to_ascii_lowercase();
        let heading = lower.trim_start_matches(['#', '-', '*', ' ']);
        if heading.starts_with("unknowns") {
            inside = true;
            // "Unknowns: none" and "Unknowns: what about X?" on one line
            if let Some((_, rest)) = t.split_once(':') {
                let rest = rest.trim();
                if !rest.is_empty() && !rest.eq_ignore_ascii_case("none") {
                    out.push(rest.to_string());
                }
            }
            continue;
        }
        if !inside {
            continue;
        }
        if t.is_empty() {
            continue;
        }
        // A new section ends the list.
        if t.starts_with('#') || (t.ends_with(':') && t.len() < 40) {
            break;
        }
        let cleaned = t.trim_start_matches(['-', '*', ' ']).trim();
        let cleaned = cleaned
            .split_once(". ")
            .map(|(head, rest)| if head.chars().all(|c| c.is_ascii_digit()) { rest } else { cleaned })
            .unwrap_or(cleaned);
        if !cleaned.is_empty() && !cleaned.eq_ignore_ascii_case("none") {
            out.push(cleaned.to_string());
        }
    }
    out
}

/// The instruction for a self-resolution pass.
pub fn self_resolution_prompt(questions: &[String]) -> String {
    let list = questions
        .iter()
        .enumerate()
        .map(|(i, q)| format!("{}. {q}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Your previous answer left these unknowns:\n{list}\n\n\
         Resolve as many as you can yourself, from the code and documents you were given. \
         Most of them are answerable by reading. Rewrite your document with the resolved ones \
         turned into requirements, and keep under Unknowns only what genuinely cannot be \
         answered without the person who set the task. Set results.unknowns to how many are left."
    )
}

/// The instruction carrying the human's answers.
pub fn answered_prompt(pairs: &[(String, String)]) -> String {
    let list = pairs
        .iter()
        .map(|(q, a)| format!("Q: {q}\nA: {a}"))
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "The person who set the task has answered your questions:\n\n{list}\n\n\
         Rewrite your document using these answers. Set results.unknowns to 0 unless something \
         genuinely remains."
    )
}

/// Accumulated answers, so tomorrow's run does not ask today's questions.
pub fn append_knowledge(project: &Path, pairs: &[(String, String)]) -> Result<()> {
    if pairs.is_empty() {
        return Ok(());
    }
    let dir = project.join(".minions");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("knowledge.md");
    let mut text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        "# What this project's owner has already told us\n\n\
         Answers given at clarification gates. Every run reads this, so a question asked once \
         is not asked again.\n"
            .to_string()
    });
    for (q, a) in pairs {
        if text.contains(q.as_str()) {
            continue;
        }
        text.push_str(&format!("\n- **{q}**\n  {a}\n"));
    }
    std::fs::write(path, text)?;
    Ok(())
}

pub fn read_knowledge(project: &Path) -> String {
    std::fs::read_to_string(project.join(".minions/knowledge.md")).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{self, Artifact};
    use crate::tokens::CharRatioTokenizer;

    fn doc_with(unknowns: &str, body: &str) -> Document {
        let text = format!(
            "---\nartifact: Requirements\nrun: r\nnode: analyst\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n  requirements: 3\n  unknowns: {unknowns}\ndigest: |\n  Something.\n---\n\nStatement: s.\n\nRequirements:\n1. one\n\nOut of scope:\n- nothing\n\n{body}\n"
        );
        document::parse(&text, &CharRatioTokenizer::default()).unwrap()
    }

    #[test]
    fn none_and_zero_and_missing_all_mean_no_questions() {
        assert_eq!(unknown_count(&doc_with("none", "b")), 0);
        assert_eq!(unknown_count(&doc_with("0", "b")), 0);
        let mut d = doc_with("0", "b");
        d.header.results.remove("unknowns");
        assert_eq!(unknown_count(&d), 0);
    }

    #[test]
    fn an_unreadable_count_is_treated_as_a_question() {
        // Better to ask once too often than to proceed on a number nobody read.
        assert_eq!(unknown_count(&doc_with("several", "b")), 1);
    }

    #[test]
    fn questions_are_taken_from_the_unknowns_section() {
        let body = "Statement: do the thing.\n\nUnknowns:\n1. Should negative input raise or clamp?\n2. Which locale?\n\nAreas touched:\n- src/a.py\n";
        let qs = extract_questions(body);
        assert_eq!(qs.len(), 2, "{qs:?}");
        assert!(qs[0].contains("negative"));
        assert!(qs[1].contains("locale"));
        assert!(!qs.iter().any(|q| q.contains("src/a.py")), "the next section leaked in");
    }

    #[test]
    fn an_inline_none_yields_nothing() {
        assert!(extract_questions("Unknowns: none\n").is_empty());
    }

    #[test]
    fn an_inline_question_is_taken() {
        let qs = extract_questions("Unknowns: what should happen on an empty string?\n");
        assert_eq!(qs.len(), 1);
        assert!(qs[0].contains("empty string"));
    }

    #[test]
    fn knowledge_accumulates_without_duplicating() {
        let dir = tempfile::tempdir().unwrap();
        let pairs = vec![("Which locale?".to_string(), "en_GB".to_string())];
        append_knowledge(dir.path(), &pairs).unwrap();
        append_knowledge(dir.path(), &pairs).unwrap();
        let text = read_knowledge(dir.path());
        assert_eq!(text.matches("Which locale?").count(), 1, "the same answer was stored twice");
        assert!(text.contains("en_GB"));
    }

    #[test]
    fn the_self_resolution_prompt_carries_every_question() {
        let qs = vec!["First?".to_string(), "Second?".to_string()];
        let p = self_resolution_prompt(&qs);
        assert!(p.contains("First?") && p.contains("Second?"));
        assert!(p.contains("from the code and documents"), "it must tell the model where to look");
    }

    #[test]
    fn artifact_stays_untouched() {
        // The ladder never rewrites the artifact type.
        assert_eq!(doc_with("2", "b").header.artifact, Artifact::Requirements);
    }
}
