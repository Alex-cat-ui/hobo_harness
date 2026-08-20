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
            // A report says first whether it measured anything. The numbers
            // are required only when it did — writing zeros for a command that
            // ran no tests is how "nothing was measured" read as "all green"
            // (finding 26); see the extra check in `parse`.
            TestReport => &["conclusive"],
            Triage => &["failures", "test_at_fault", "code_at_fault"],
            ForemanLog => &["calls", "budget_used", "complete"],
            Report => &[],
        }
    }

    /// Sections the body must carry — for the artifacts whose role prompt names
    /// them as a list, and only those. The check is over prose, so it is worth
    /// exactly as much as what the prompt teaches: `RiskAssessment` is
    /// deliberately absent, because its role asks for prose and one of the four
    /// such documents produced so far is sound prose without a single heading.
    /// Rejecting it would be the harness punishing a model for obeying the
    /// prompt it was handed. `TestPlan` is absent for a simpler reason: no role
    /// produces one yet.
    ///
    /// Measured before it was written: the rule was run over all 30 documents
    /// of every run on disk, and the only rejection was that RiskAssessment.
    pub fn required_sections(&self) -> &'static [&'static str] {
        use Artifact::*;
        match self {
            Requirements => &["Statement", "Requirements", "Out of scope"],
            Plan => &["Approach", "Rejected alternatives", "Steps"],
            SystemMap => &["Purpose", "Entry points", "Main parts", "Data flows"],
            _ => &[],
        }
    }
}

/// The body with a wrapping code fence taken off, if it has one. Models fence
/// JSON out of habit, and a fence must not hide what it wraps.
fn unfenced(body: &str) -> &str {
    let t = body.trim();
    let Some(rest) = t.strip_prefix("```") else { return t };
    let rest = rest.split_once('\n').map(|(_, r)| r).unwrap_or("");
    rest.trim_end().strip_suffix("```").unwrap_or(rest).trim()
}

/// Whether the body opens a section with this name. Deliberately lenient: the
/// name may be a heading, a list item, bold, or a bare line, in any case. A
/// document is rejected for missing a section, never for how it dressed one —
/// a stricter rule would burn attempts on punctuation.
fn body_has_section(body: &str, name: &str) -> bool {
    let wanted = name.to_ascii_lowercase();
    body.lines().any(|line| {
        line.trim()
            .trim_start_matches(['#', '-', '*', '>', ' '])
            .to_ascii_lowercase()
            .starts_with(&wanted)
    })
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
    MissingSection { artifact: &'static str, name: &'static str },
    BodyRestatesHeader,
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
            DocError::MissingSection { artifact, name } => {
                write!(f, "the body of a `{artifact}` document must carry a `{name}` section")
            }
            DocError::BodyRestatesHeader => {
                write!(f, "the body is the header said again as JSON, not a document body")
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
            DocError::MissingSection { name, .. } => format!(
                "Write the missing part of the body: a line beginning with `{name}:` and then that section. \
                 The body is prose in the structure the role described, not the header said twice."
            ),
            DocError::BodyRestatesHeader => "The body after the closing `---` is prose in the structure your role \
                 described. Do not repeat the header there, and do not answer in JSON: the numbers already stand in \
                 `results:`, and what belongs below is the substance behind them."
                .to_string(),
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

    // A report that says it measured something must carry what it measured.
    // The pair is the contract: `conclusive: no` and no numbers, or
    // `conclusive: yes` and all three.
    if artifact == Artifact::TestReport && results.get("conclusive").map(|s| s.as_str()) == Some("yes") {
        for key in ["total", "passed", "failed"] {
            if !results.contains_key(key) {
                return Err(DocError::MissingResultsKey { artifact: artifact.name(), key });
            }
        }
    }

    // A body that restates the header is what a model produces when it has
    // learned the shape of an answer and not the substance of one. It passed
    // every check there was — the body was not empty and the header above it
    // was perfect — and went into the report as "one breaking issue found":
    // `05_findings.md` of the run 2026-08-16T18-41-51.
    let naked = unfenced(&body);
    if naked.starts_with('{') && naked.contains("\"digest\"") {
        return Err(DocError::BodyRestatesHeader);
    }

    // Until now the above was the whole check on the body. The next node reads
    // sections out of it, so a body that skips one is a document-shaped reply
    // rather than a document (IMPROVEMENTS 1.3).
    for name in artifact.required_sections() {
        if !body_has_section(&body, name) {
            return Err(DocError::MissingSection { artifact: artifact.name(), name });
        }
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

Requirements:
1. The active session survives the switch.
2. The switch is atomic.

Out of scope:
- Changing the tunnel protocol.
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

    /// The real body of `05_findings.md`, run 2026-08-16T18-41-51: the reviewer
    /// answered with its own header as JSON, the harness accepted it, and
    /// `results.breaking: 1` went into the report as a finding nobody had made.
    #[test]
    fn a_report_that_claims_a_measurement_must_carry_it() {
        let head = "---\nartifact: TestReport\nrun: r\nnode: tests\nattempt: 1\nmodel: harness\ncreated: c\ninputs: []\nresults:\n";
        let tail = "digest: |\n  Something ran.\n---\n\n$ python3 -m unittest\n\nRan 4 tests\n";

        // Nothing measured: `conclusive` alone is the whole contract.
        let nothing = format!("{head}  conclusive: no\n{tail}");
        assert!(parse(&nothing, &tk()).is_ok(), "a report of nothing owes no numbers");

        // Something measured, and the numbers left out: that is the shape a
        // branch would read as zero.
        let half = format!("{head}  conclusive: yes\n  total: 4\n  passed: 4\n{tail}");
        assert_eq!(
            parse(&half, &tk()).unwrap_err(),
            DocError::MissingResultsKey { artifact: "TestReport", key: "failed" }
        );

        let whole = format!("{head}  conclusive: yes\n  total: 4\n  passed: 3\n  failed: 1\n{tail}");
        assert!(parse(&whole, &tk()).is_ok());
    }

    #[test]
    fn rejects_a_body_that_is_the_header_said_again() {
        let echo = "{\n  \"artifact\": \"Findings\",\n  \"run\": \"2026-08-16T18-41-51_full-development\",\n  \"node\": \"reviewer\",\n  \"attempt\": 1,\n  \"model\": \"qwen2.5-coder:14b\",\n  \"results\": {\n    \"breaking\": 1\n  },\n  \"digest\": \"One breaking issue found in src/durations.py.\"\n}";
        let text = format!(
            "---\nartifact: Findings\nrun: r\nnode: reviewer\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n  breaking: 1\n  risky: 0\n  minor: 0\n  unmet_requirements: 0\ndigest: |\n  One breaking issue.\n---\n\n{echo}\n"
        );
        let e = parse(&text, &tk()).unwrap_err();
        assert_eq!(e, DocError::BodyRestatesHeader);
        assert!(e.guidance().contains("prose"), "{}", e.guidance());

        // Fenced, which is how the same model writes it half the time.
        let fenced = text.replace(echo, &format!("```json\n{echo}\n```"));
        assert_eq!(parse(&fenced, &tk()).unwrap_err(), DocError::BodyRestatesHeader);
    }

    #[test]
    fn a_body_that_merely_contains_json_is_not_an_echo() {
        // Quoting a payload is legitimate: the rule is about a body that *is*
        // the header, not about the character `{`.
        let body = "The failure fires on this input:\n\n```json\n{\"digest\": \"x\"}\n```\n\nThe parser then returns None.";
        let text = format!(
            "---\nartifact: Findings\nrun: r\nnode: reviewer\nattempt: 1\nmodel: m\ncreated: c\ninputs: []\nresults:\n  breaking: 1\n  risky: 0\n  minor: 0\n  unmet_requirements: 0\ndigest: |\n  One issue.\n---\n\n{body}\n"
        );
        assert!(parse(&text, &tk()).is_ok(), "a body quoting JSON was taken for an echo of the header");
    }

    #[test]
    fn rejects_a_body_missing_a_required_section_and_names_it() {
        let text = GOOD.replace("\nOut of scope:\n- Changing the tunnel protocol.\n", "\n");
        let e = parse(&text, &tk()).unwrap_err();
        assert_eq!(e, DocError::MissingSection { artifact: "Requirements", name: "Out of scope" });
        assert!(e.guidance().contains("Out of scope"), "{}", e.guidance());
    }

    #[test]
    fn a_section_is_recognised_however_the_model_dresses_it() {
        for dressed in ["## Out of scope", "- Out of scope:", "**Out of scope**", "OUT OF SCOPE:", "> Out of scope"] {
            let text = GOOD.replace("Out of scope:\n- Changing", &format!("{dressed}\n- Changing"));
            assert!(
                parse(&text, &tk()).is_ok(),
                "a section written as `{dressed}` was rejected, which spends an attempt on punctuation"
            );
        }
    }

    #[test]
    fn an_artifact_whose_role_asks_for_prose_demands_no_sections() {
        // Measured over every document on disk before the rule was written: the
        // only rejection was a RiskAssessment that is sound prose without a
        // heading, and its role prompt asks for exactly that.
        assert!(Artifact::RiskAssessment.required_sections().is_empty());
        assert!(Artifact::Findings.required_sections().is_empty());
        assert!(Artifact::Patch.required_sections().is_empty());
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
