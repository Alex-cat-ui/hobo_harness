//! Run documents: the bus agents hand work over on.
//!
//! Parsing is strict and hand-written rather than delegated to a general YAML
//! parser, because this text is produced by a model and a rejection must come
//! back as an instruction the model can act on — not "invalid yaml at line 3".

use crate::tokens::Tokenizer;
use std::collections::BTreeMap;
use std::fmt;

pub const DIGEST_TOKEN_LIMIT: usize = 400;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Artifact {
    Task,
    Requirements,
    Plan,
    Patch,
    Findings,
    SystemMap,
    RiskAssessment,
    TestSurface,
    TestPlan,
    TestSuite,
    TestReport,
    Triage,
    ForemanLog,
    Report,
}

impl Artifact {
    pub fn parse(s: &str) -> Option<Self> {
        use Artifact::*;
        Some(match s.trim() {
            "Task" => Task,
            "Requirements" => Requirements,
            "Plan" => Plan,
            "Patch" => Patch,
            "Findings" => Findings,
            "SystemMap" => SystemMap,
            "RiskAssessment" => RiskAssessment,
            "TestSurface" => TestSurface,
            "TestPlan" => TestPlan,
            "TestSuite" => TestSuite,
            "TestReport" => TestReport,
            "Triage" => Triage,
            "ForemanLog" => ForemanLog,
            "Report" => Report,
            _ => return None,
        })
    }

    pub fn name(&self) -> &'static str {
        use Artifact::*;
        match self {
            Task => "Task",
            Requirements => "Requirements",
            Plan => "Plan",
            Patch => "Patch",
            Findings => "Findings",
            SystemMap => "SystemMap",
            RiskAssessment => "RiskAssessment",
            TestSurface => "TestSurface",
            TestPlan => "TestPlan",
            TestSuite => "TestSuite",
            TestReport => "TestReport",
            Triage => "Triage",
            ForemanLog => "ForemanLog",
            Report => "Report",
        }
    }

    /// The `results` keys this artifact must carry. Branches are written
    /// against these, so the schema is fixed per type (SPEC appendix D).
    pub fn required_results(&self) -> &'static [&'static str] {
        use Artifact::*;
        match self {
            Task => &[],
            Requirements => &["requirements", "unknowns"],
            Plan => &["steps", "risks", "verdict"],
            Patch => &["files", "added", "removed", "steps_done", "steps_skipped"],
            Findings => &["breaking", "risky", "minor", "unmet_requirements"],
            SystemMap => &["parts", "entry_points", "blank_spots"],
            RiskAssessment => &["fragile_areas", "hidden_couplings"],
            TestSurface => &["public_units", "covered", "uncovered", "untestable"],
            TestPlan => &["tests", "mandatory"],
            TestSuite => &["tests_written", "files"],
            TestReport => &["total", "passed", "failed", "baseline_total", "baseline_passed"],
            Triage => &["failures", "test_at_fault", "code_at_fault"],
            ForemanLog => &["calls", "budget_used", "complete"],
            Report => &[],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub artifact: Artifact,
    pub run: String,
    pub node: String,
    pub attempt: u32,
    pub model: String,
    pub created: String,
    pub inputs: Vec<String>,
    pub results: BTreeMap<String, String>,
    pub digest: String,
    /// Set when a human supplied the document instead of a model (FR-41).
    pub authored_by_human: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub header: Header,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocError {
    MissingFrontMatter,
    UnterminatedFrontMatter,
    MissingField(&'static str),
    UnknownArtifact(String),
    BadAttempt(String),
    EmptyDigest,
    DigestTooLong { tokens: usize, limit: usize },
    MissingResultsKey { artifact: &'static str, key: &'static str },
    EmptyBody,
}

impl fmt::Display for DocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DocError::MissingFrontMatter => write!(f, "the document does not begin with a `---` header block"),
            DocError::UnterminatedFrontMatter => write!(f, "the header block is never closed with `---`"),
            DocError::MissingField(k) => write!(f, "the header is missing the required field `{k}`"),
            DocError::UnknownArtifact(a) => write!(f, "`{a}` is not a known artifact type"),
            DocError::BadAttempt(v) => write!(f, "`attempt` must be a positive whole number, found `{v}`"),
            DocError::EmptyDigest => write!(f, "`digest` is present but empty"),
            DocError::DigestTooLong { tokens, limit } => {
                write!(f, "the digest is {tokens} tokens, over the limit of {limit}")
            }
            DocError::MissingResultsKey { artifact, key } => {
                write!(f, "a `{artifact}` document must carry `results.{key}`")
            }
            DocError::EmptyBody => write!(f, "the document has a header but no body"),
        }
    }
}

impl DocError {
    /// What is sent back to the model on a retry. The spec requires format
    /// guidance rather than a bare rejection, and this is that guidance.
    pub fn guidance(&self) -> String {
        let fix = match self {
            DocError::MissingFrontMatter | DocError::UnterminatedFrontMatter => {
                "Begin the reply with a line containing only `---`, then the header fields, then a line containing only `---`, then the document body.".to_string()
            }
            DocError::MissingField(k) => format!("Add a `{k}:` line to the header."),
            DocError::UnknownArtifact(_) => "Set `artifact:` to the exact type you were asked to produce.".to_string(),
            DocError::BadAttempt(_) => "Set `attempt:` to a whole number starting at 1.".to_string(),
            DocError::EmptyDigest => "Write two or three sentences of substance under `digest:`.".to_string(),
            DocError::DigestTooLong { limit, .. } => {
                format!("Shorten the digest to at most {limit} tokens. Keep the key numbers and drop the elaboration; the body carries the detail.")
            }
            DocError::MissingResultsKey { key, .. } => {
                format!("Add `{key}:` under the `results:` block, with a numeric or single-word value.")
            }
            DocError::EmptyBody => "Write the document body after the closing `---`.".to_string(),
        };
        format!("The previous reply was rejected: {self}. {fix} Reply with the corrected document only.")
    }
}

fn parse_inline_list(v: &str) -> Vec<String> {
    let v = v.trim();
    let v = v.strip_prefix('[').unwrap_or(v);
    let v = v.strip_suffix(']').unwrap_or(v);
    v.split(',')
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn parse(text: &str, tok: &dyn Tokenizer) -> Result<Document, DocError> {
    let text = text.trim_start_matches('\u{feff}').trim_start();
    let rest = text.strip_prefix("---").ok_or(DocError::MissingFrontMatter)?;
    let rest = rest.strip_prefix('\n').unwrap_or(rest);

    let mut header_lines: Vec<&str> = Vec::new();
    let mut body = String::new();
    let mut closed = false;
    let mut it = rest.lines();
    for line in it.by_ref() {
        if line.trim_end() == "---" {
            closed = true;
            break;
        }
        header_lines.push(line);
    }
    if !closed {
        return Err(DocError::UnterminatedFrontMatter);
    }
    for line in it {
        body.push_str(line);
        body.push('\n');
    }

    let mut simple: BTreeMap<String, String> = BTreeMap::new();
    let mut results: BTreeMap<String, String> = BTreeMap::new();
    let mut digest = String::new();

    let mut mode = 0u8; // 0 top level, 1 inside results, 2 inside digest block
    for line in header_lines {
        let indented = line.starts_with(' ') || line.starts_with('\t');
        let trimmed = line.trim();

        if trimmed.is_empty() {
            if mode == 2 {
                digest.push('\n');
            }
            continue;
        }
        if !indented {
            mode = 0;
        }

        if mode == 2 && indented {
            if !digest.is_empty() {
                digest.push('\n');
            }
            digest.push_str(trimmed);
            continue;
        }
        if mode == 1 && indented {
            if let Some((k, v)) = trimmed.split_once(':') {
                results.insert(k.trim().to_string(), v.trim().to_string());
            }
            continue;
        }

        let Some((key, value)) = trimmed.split_once(':') else { continue };
        let key = key.trim().to_lowercase();
        let value = value.trim();

        match key.as_str() {
            "results" => {
                mode = 1;
                if !value.is_empty() {
                    for pair in value.trim_matches(|c| c == '{' || c == '}').split(',') {
                        if let Some((k, v)) = pair.split_once(':') {
                            results.insert(k.trim().to_string(), v.trim().to_string());
                        }
                    }
                }
            }
            "digest" => {
                mode = 2;
                let v = value.trim_start_matches('|').trim();
                if !v.is_empty() {
                    digest.push_str(v);
                }
            }
            _ => {
                simple.insert(key, value.to_string());
            }
        }
    }

    let get = |k: &'static str| -> Result<String, DocError> {
        simple.get(k).filter(|v| !v.is_empty()).cloned().ok_or(DocError::MissingField(k))
    };

    let artifact_raw = get("artifact")?;
    let artifact = Artifact::parse(&artifact_raw).ok_or(DocError::UnknownArtifact(artifact_raw))?;

    let attempt_raw = simple.get("attempt").cloned().unwrap_or_else(|| "1".to_string());
    let attempt: u32 = attempt_raw
        .trim()
        .parse()
        .map_err(|_| DocError::BadAttempt(attempt_raw.clone()))?;
    if attempt == 0 {
        return Err(DocError::BadAttempt(attempt_raw));
    }

    let digest = digest.trim().to_string();
    if digest.is_empty() {
        return Err(DocError::EmptyDigest);
    }
    let dt = tok.count(&digest);
    if dt > DIGEST_TOKEN_LIMIT {
        return Err(DocError::DigestTooLong { tokens: dt, limit: DIGEST_TOKEN_LIMIT });
    }

    for key in artifact.required_results() {
        if !results.contains_key(*key) {
            return Err(DocError::MissingResultsKey { artifact: artifact.name(), key });
        }
    }

    // The blank line separating header from body is punctuation, not content.
    // Keeping it would make render(parse(x)) differ from x on every round trip.
    let body = body.trim_start_matches('\n').trim_end().to_string();
    if body.trim().is_empty() {
        return Err(DocError::EmptyBody);
    }

    Ok(Document {
        header: Header {
            artifact,
            run: get("run").unwrap_or_default(),
            node: get("node").unwrap_or_default(),
            attempt,
            model: get("model").unwrap_or_default(),
            created: get("created").unwrap_or_default(),
            inputs: simple.get("inputs").map(|v| parse_inline_list(v)).unwrap_or_default(),
            results,
            digest,
            authored_by_human: simple
                .get("authored_by_human")
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        },
        body,
    })
}

pub fn render(doc: &Document) -> String {
    let h = &doc.header;
    let mut s = String::from("---\n");
    s.push_str(&format!("artifact: {}\n", h.artifact.name()));
    s.push_str(&format!("run: {}\n", h.run));
    s.push_str(&format!("node: {}\n", h.node));
    s.push_str(&format!("attempt: {}\n", h.attempt));
    s.push_str(&format!("model: {}\n", h.model));
    s.push_str(&format!("created: {}\n", h.created));
    s.push_str(&format!("inputs: [{}]\n", h.inputs.join(", ")));
    if h.authored_by_human {
        s.push_str("authored_by_human: true\n");
    }
    s.push_str("results:\n");
    for (k, v) in &h.results {
        s.push_str(&format!("  {k}: {v}\n"));
    }
    s.push_str("digest: |\n");
    for line in h.digest.lines() {
        s.push_str(&format!("  {line}\n"));
    }
    s.push_str("---\n\n");
    s.push_str(&doc.body);
    s.push('\n');
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::CharRatioTokenizer;

    fn tk() -> CharRatioTokenizer {
        CharRatioTokenizer::default()
    }

    const GOOD: &str = "---
artifact: Requirements
run: 2026-08-16T14-30-12
node: analyst
attempt: 1
model: qwen2.5:14b
created: 2026-08-16T14:31:05
inputs: [00_task.md]
results:
  requirements: 3
  unknowns: 0
digest: |
  Switching the tunnel profile must not drop the active session.
  Three requirements, no unknowns.
---

Statement: the profile must change without a restart.
";

    #[test]
    fn parses_a_well_formed_document() {
        let d = parse(GOOD, &tk()).unwrap();
        assert_eq!(d.header.artifact, Artifact::Requirements);
        assert_eq!(d.header.attempt, 1);
        assert_eq!(d.header.inputs, vec!["00_task.md"]);
        assert_eq!(d.header.results.get("requirements").unwrap(), "3");
        assert!(d.header.digest.contains("no unknowns"));
        assert!(d.body.starts_with("Statement:"));
    }

    #[test]
    fn round_trips() {
        let d = parse(GOOD, &tk()).unwrap();
        let again = parse(&render(&d), &tk()).unwrap();
        assert_eq!(d, again);
    }

    #[test]
    fn rejects_missing_front_matter() {
        let e = parse("just prose", &tk()).unwrap_err();
        assert_eq!(e, DocError::MissingFrontMatter);
        assert!(e.guidance().contains("---"));
    }

    #[test]
    fn rejects_missing_results_key_naming_it() {
        let text = GOOD.replace("  unknowns: 0\n", "");
        let e = parse(&text, &tk()).unwrap_err();
        assert_eq!(e, DocError::MissingResultsKey { artifact: "Requirements", key: "unknowns" });
        assert!(e.guidance().contains("unknowns"));
    }

    #[test]
    fn rejects_an_over_long_digest_rather_than_truncating() {
        let long = "word ".repeat(2000);
        let text = GOOD.replace(
            "  Switching the tunnel profile must not drop the active session.\n  Three requirements, no unknowns.\n",
            &format!("  {long}\n"),
        );
        match parse(&text, &tk()).unwrap_err() {
            DocError::DigestTooLong { tokens, limit } => {
                assert!(tokens > limit);
                assert_eq!(limit, DIGEST_TOKEN_LIMIT);
            }
            other => panic!("expected DigestTooLong, got {other:?}"),
        }
    }

    #[test]
    fn rejects_empty_digest() {
        let text = GOOD.replace(
            "digest: |\n  Switching the tunnel profile must not drop the active session.\n  Three requirements, no unknowns.\n",
            "digest: |\n",
        );
        assert_eq!(parse(&text, &tk()).unwrap_err(), DocError::EmptyDigest);
    }

    #[test]
    fn rejects_unknown_artifact() {
        let text = GOOD.replace("artifact: Requirements", "artifact: Wishlist");
        assert!(matches!(parse(&text, &tk()).unwrap_err(), DocError::UnknownArtifact(_)));
    }

    #[test]
    fn rejects_empty_body() {
        let text = GOOD.split("---\n\n").next().unwrap().to_string() + "---\n\n";
        assert_eq!(parse(&text, &tk()).unwrap_err(), DocError::EmptyBody);
    }

    #[test]
    fn every_guidance_tells_the_model_what_to_do() {
        let errors = [
            DocError::MissingFrontMatter,
            DocError::UnterminatedFrontMatter,
            DocError::MissingField("run"),
            DocError::UnknownArtifact("x".into()),
            DocError::BadAttempt("x".into()),
            DocError::EmptyDigest,
            DocError::DigestTooLong { tokens: 500, limit: 400 },
            DocError::MissingResultsKey { artifact: "Plan", key: "steps" },
            DocError::EmptyBody,
        ];
        for e in errors {
            let g = e.guidance();
            assert!(g.len() > 40, "guidance too thin for {e:?}");
            assert!(g.contains("rejected"), "guidance must say what happened: {g}");
        }
    }
}
