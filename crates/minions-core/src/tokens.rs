//! Token counting. Never skipped: the design forbids "it will probably fit".

pub trait Tokenizer: Send + Sync {
    fn count(&self, text: &str) -> usize;
}

/// Conservative fallback used when a model reports no tokenizer (SDD §7).
/// Deliberately over-counts: a digest wrongly rejected costs one retry, a
/// digest wrongly accepted corrupts every downstream node that reads it.
pub struct CharRatioTokenizer {
    chars_per_token: f32,
}

impl Default for CharRatioTokenizer {
    fn default() -> Self {
        Self { chars_per_token: 3.2 }
    }
}

impl Tokenizer for CharRatioTokenizer {
    fn count(&self, text: &str) -> usize {
        (text.chars().count() as f32 / self.chars_per_token).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn over_counts_rather_than_under() {
        let t = CharRatioTokenizer::default();
        // English prose is roughly 4 chars/token; 3.2 keeps us on the safe side.
        let text = "the quick brown fox jumps over the lazy dog";
        assert!(t.count(text) >= text.split_whitespace().count());
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(CharRatioTokenizer::default().count(""), 0);
    }
}
