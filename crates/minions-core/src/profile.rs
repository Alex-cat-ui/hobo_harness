//! What a model can actually do, measured rather than declared.
//!
//! Vendor metadata is not trustworthy on this point. All three Qwen models
//! installed here declare the `tools` capability; two use the channel and
//! `qwen2.5-coder:14b` writes its calls into the text instead. A product that
//! must run on any model cannot ask the model what it can do — it has to find
//! out, once, and then adapt.
//!
//! The protocol adapts to the model. A weaker model gets a simpler protocol,
//! never a broken one.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChannel {
    /// Calls arrive as structured data. Nothing to parse, nothing to get wrong.
    Native,
    /// Calls arrive as JSON written into the reply. Recoverable, and observed.
    JsonInText,
    /// Neither. Tools are described in the prompt and answers are parsed by
    /// line syntax, which is the last resort.
    TextOnly,
}

/// How a model is asked to change a file. Granularity is a capability: a model
/// that cannot emit a whole file in one argument is not a model that cannot
/// edit, it is a model that needs smaller edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditStyle {
    /// Send the entire new file.
    WholeFile,
    /// Send the old fragment and the new one; the harness splices them.
    Replacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    pub model: String,
    pub channel: ToolChannel,
    pub edit_style: EditStyle,
    /// Several calls in one turn, which halves the number of round trips.
    pub parallel_calls: bool,
    /// Produces our document header without needing a correction.
    pub holds_format: bool,
    /// Largest reply seen during probing, in characters. A model asked for more
    /// than it has ever produced will truncate silently.
    pub max_output_chars: usize,
    pub measured_at: String,
}

impl ModelProfile {
    /// What to assume before anything has been measured. Deliberately the most
    /// conservative combination that still works: every capability is proven,
    /// never presumed.
    pub fn unmeasured(model: &str) -> Self {
        Self {
            model: model.to_string(),
            channel: ToolChannel::TextOnly,
            edit_style: EditStyle::Replacement,
            parallel_calls: false,
            holds_format: false,
            max_output_chars: 2000,
            measured_at: String::new(),
        }
    }

    pub fn is_measured(&self) -> bool {
        !self.measured_at.is_empty()
    }

    /// How many attempts a node should get at producing its document.
    pub fn document_attempts(&self) -> u32 {
        if self.holds_format {
            3
        } else {
            5
        }
    }

    /// Whether the tool list can be sent as a schema at all.
    pub fn send_schemas(&self) -> bool {
        self.channel == ToolChannel::Native
    }
}

/// A profile per model, kept beside the settings and refreshed when a model is
/// bound to a slot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileStore {
    #[serde(default)]
    pub profiles: Vec<ModelProfile>,
}

impl ProfileStore {
    pub fn get(&self, model: &str) -> Option<&ModelProfile> {
        self.profiles.iter().find(|p| p.model == model)
    }

    /// The profile to work from: measured if we have one, conservative if not.
    pub fn effective(&self, model: &str) -> ModelProfile {
        self.get(model).cloned().unwrap_or_else(|| ModelProfile::unmeasured(model))
    }

    pub fn put(&mut self, profile: ModelProfile) {
        self.profiles.retain(|p| p.model != profile.model);
        self.profiles.push(profile);
    }

    pub fn load(path: &std::path::Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_model_is_assumed_to_be_the_weakest_one_that_works() {
        let p = ModelProfile::unmeasured("something-new");
        assert_eq!(p.channel, ToolChannel::TextOnly);
        assert_eq!(p.edit_style, EditStyle::Replacement);
        assert!(!p.parallel_calls);
        assert!(!p.holds_format);
        assert!(!p.is_measured());
        assert!(!p.send_schemas(), "schemas must not be sent to an unproven channel");
    }

    #[test]
    fn a_model_that_loses_the_format_gets_more_attempts() {
        let mut p = ModelProfile::unmeasured("m");
        assert_eq!(p.document_attempts(), 5);
        p.holds_format = true;
        assert_eq!(p.document_attempts(), 3);
    }

    #[test]
    fn the_effective_profile_falls_back_without_pretending_to_be_measured() {
        let store = ProfileStore::default();
        let p = store.effective("never-seen");
        assert!(!p.is_measured());
        assert_eq!(p.model, "never-seen");
    }

    #[test]
    fn putting_a_profile_replaces_the_previous_one() {
        let mut store = ProfileStore::default();
        let mut a = ModelProfile::unmeasured("m");
        a.measured_at = "yesterday".into();
        store.put(a);
        let mut b = ModelProfile::unmeasured("m");
        b.measured_at = "today".into();
        b.holds_format = true;
        store.put(b);
        assert_eq!(store.profiles.len(), 1, "a remeasurement must replace, not accumulate");
        assert!(store.get("m").unwrap().holds_format);
    }

    #[test]
    fn a_store_survives_a_round_trip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profiles.json");
        let mut store = ProfileStore::default();
        let mut p = ModelProfile::unmeasured("qwen2.5:14b");
        p.channel = ToolChannel::Native;
        p.edit_style = EditStyle::WholeFile;
        p.measured_at = "now".into();
        store.put(p);
        store.save(&path).unwrap();

        let back = ProfileStore::load(&path);
        assert_eq!(back.get("qwen2.5:14b").unwrap().channel, ToolChannel::Native);
    }

    #[test]
    fn a_missing_or_broken_file_yields_an_empty_store_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ProfileStore::load(&dir.path().join("absent.json")).profiles.is_empty());
        let broken = dir.path().join("broken.json");
        std::fs::write(&broken, "{ this is not json").unwrap();
        assert!(ProfileStore::load(&broken).profiles.is_empty());
    }
}
