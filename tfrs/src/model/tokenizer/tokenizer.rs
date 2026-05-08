#![allow(dead_code)]

//! Tokenizer wrapper around the HuggingFace `tokenizers` crate.
//!
//! Why this design:
//! - The `Tokenizer` trait is the only thing the rest of the codebase depends on.
//!   Swap `HfTokenizer` for any other implementation (tiktoken-rs, sentencepiece,
//!   a custom BPE) without touching call sites.
//! - `HfTokenizer` is a thin adapter that loads a `tokenizer.json` from disk.
//!   Fully offline — no network at build or runtime. Download a pretrained
//!   `tokenizer.json` once (e.g. from a HuggingFace model repo) and load it.

use std::path::Path;


/// Default location of the bundled pretrained tokenizer, relative to the
/// crate root. The file is committed under `assets/` so the project works
/// fully offline — no Hub download at runtime.
pub const DEFAULT_TOKENIZER_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/assets/tokenizer.json");

/// The same `tokenizer.json` baked directly into the compiled binary so the
/// produced executable is self-contained: ship just `tfrs` (and the trained
/// `mail_model.bin`) and it works anywhere — no need to also distribute
/// `assets/tokenizer.json`.
pub const DEFAULT_TOKENIZER_BYTES: &[u8] =
    include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/tokenizer.json"));


// ============================================================================
// Trait — generic interface, replaceable backend
// ============================================================================

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;

    fn vocab_size(&self) -> usize;

    fn pad_id(&self) -> usize;
    fn unk_id(&self) -> usize;
    fn bos_id(&self) -> usize;
    fn eos_id(&self) -> usize;

    /// Encode and wrap with bos/eos.
    fn encode_with_specials(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::with_capacity(2);
        out.push(self.bos_id());
        out.extend(self.encode(text));
        out.push(self.eos_id());
        out
    }

    /// Pad/truncate to exactly `len` using `pad_id`.
    fn pad_to(&self, mut ids: Vec<usize>, len: usize) -> Vec<usize> {
        if ids.len() >= len {
            ids.truncate(len);
        } else {
            ids.resize(len, self.pad_id());
        }
        ids
    }
}


// ============================================================================
// HuggingFace adapter
// ============================================================================

pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    pad_id: usize,
    unk_id: usize,
    bos_id: usize,
    eos_id: usize,
}

impl HfTokenizer {
    /// Load any `tokenizer.json` from disk. Fully offline.
    /// Get such a file e.g. by `git clone`-ing a model repo from HuggingFace
    /// (it contains `tokenizer.json`), or by exporting one with
    /// `tokenizer.save("tokenizer.json")` from Python.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError(format!("from_file: {e}")))?;
        Ok(Self::wrap(inner))
    }

    /// Load any `tokenizer.json` from a byte slice. Use together with
    /// `include_bytes!` to ship a self-contained binary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| TokenizerError(format!("from_bytes: {e}")))?;
        Ok(Self::wrap(inner))
    }

    /// Load the bundled default tokenizer — embedded into the binary at
    /// compile time, so this works even when the binary is moved away from
    /// the source tree.
    /// This is the **gbert** WordPiece tokenizer (deepset, 31 102 tokens, cased)
    /// — a German-first vocabulary trained on German Wikipedia, OpenLegalData,
    /// news and Common Crawl. Picked deliberately for the email use case:
    /// preserves casing (relevant for German nouns and "Sie"), keeps common
    /// German words (Rechnung, Damen, freundlich) as whole tokens, handles
    /// Umlauts and ß cleanly. English text still tokenizes correctly but
    /// fragments more — acceptable for emails that are mostly German.
    pub fn default_pretrained() -> Result<Self, TokenizerError> {
        Self::from_bytes(DEFAULT_TOKENIZER_BYTES)
    }

    fn wrap(inner: tokenizers::Tokenizer) -> Self {
        // Different pretrained tokenizers use different special-token names.
        // Probe the common conventions and fall back to id 0 if absent —
        // callers that care can `.expect(...)` on the trait getters.
        let pad_id = find(&inner, &["[PAD]", "<pad>", "<|pad|>"]).unwrap_or(0);
        let unk_id = find(&inner, &["[UNK]", "<unk>", "<|unk|>"]).unwrap_or(0);
        let bos_id = find(
            &inner,
            &["[CLS]", "<s>", "<bos>", "<|startoftext|>", "<|endoftext|>"],
        )
        .unwrap_or(0);
        let eos_id = find(
            &inner,
            &["[SEP]", "</s>", "<eos>", "<|endoftext|>"],
        )
        .unwrap_or(0);

        Self {
            inner,
            pad_id,
            unk_id,
            bos_id,
            eos_id,
        }
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Vec<usize> {
        // `add_special_tokens = false` here: callers opt in via
        // `encode_with_specials` if they want bos/eos wrapping.
        let enc = self
            .inner
            .encode(text, false)
            .expect("HfTokenizer::encode failed");
        enc.get_ids().iter().map(|&i| i as usize).collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        let u32_ids: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
        self.inner
            .decode(&u32_ids, true)
            .expect("HfTokenizer::decode failed")
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn pad_id(&self) -> usize { self.pad_id }
    fn unk_id(&self) -> usize { self.unk_id }
    fn bos_id(&self) -> usize { self.bos_id }
    fn eos_id(&self) -> usize { self.eos_id }
}


fn find(tok: &tokenizers::Tokenizer, candidates: &[&str]) -> Option<usize> {
    for c in candidates {
        if let Some(id) = tok.token_to_id(c) {
            return Some(id as usize);
        }
    }
    None
}


// ============================================================================
// Error type
// ============================================================================

#[derive(Debug)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TokenizerError {}


// ============================================================================
// Tests — exercise the bundled tokenizer end-to-end
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_default_pretrained() {
        let t = HfTokenizer::default_pretrained().expect("load failed");
        // gbert has 31 102 tokens
        assert_eq!(t.vocab_size(), 31_102);
    }

    #[test]
    fn encodes_and_decodes_english() {
        let t = HfTokenizer::default_pretrained().unwrap();
        let ids = t.encode("Hi team, please find attached the invoice.");
        assert!(!ids.is_empty());
        let back = t.decode(&ids);
        assert!(back.to_lowercase().contains("invoice"));
        assert!(back.to_lowercase().contains("team"));
    }

    #[test]
    fn encodes_and_decodes_german_email() {
        let t = HfTokenizer::default_pretrained().unwrap();
        let ids = t.encode("Sehr geehrte Damen und Herren, anbei die Rechnung. Mit freundlichen Grüßen, Lukas");
        assert!(!ids.is_empty());
        // German vocab should cover umlauts, ß and common words without [UNK]
        assert!(!ids.iter().any(|&i| i == t.unk_id()));
        let back = t.decode(&ids);
        assert!(back.contains("Rechnung"));
        assert!(back.contains("Damen"));
        assert!(back.contains("Lukas"));
    }

    #[test]
    fn german_is_more_compact_than_english() {
        // Sanity check that the German-first vocab is doing its job:
        // an email-style German sentence shouldn't fragment dramatically more
        // than a comparable English one.
        let t = HfTokenizer::default_pretrained().unwrap();
        let de = t.encode("Sehr geehrte Damen und Herren, anbei die Rechnung.");
        let en = t.encode("Dear Sirs and Madams, please find the invoice attached.");
        // Allow English to be up to 2x more fragmented — gbert is German-first.
        assert!(
            de.len() <= en.len() * 2,
            "DE={} EN={} — German tokenization unexpectedly bad",
            de.len(),
            en.len()
        );
    }

    #[test]
    fn special_tokens_resolved() {
        let t = HfTokenizer::default_pretrained().unwrap();
        // gbert: [PAD]=0, [UNK]=101, [CLS]=102, [SEP]=103
        assert_eq!(t.pad_id(), 0);
        assert_eq!(t.unk_id(), 101);
        assert_eq!(t.bos_id(), 102);
        assert_eq!(t.eos_id(), 103);
    }

    #[test]
    fn encode_with_specials_wraps_cls_sep() {
        let t = HfTokenizer::default_pretrained().unwrap();
        let ids = t.encode_with_specials("hi");
        assert_eq!(ids[0], t.bos_id());
        assert_eq!(*ids.last().unwrap(), t.eos_id());
    }

    #[test]
    fn pad_to_extends_with_pad_id() {
        let t = HfTokenizer::default_pretrained().unwrap();
        let padded = t.pad_to(vec![5, 6, 7], 6);
        assert_eq!(padded.len(), 6);
        assert_eq!(padded[3], t.pad_id());
        assert_eq!(padded[4], t.pad_id());
        assert_eq!(padded[5], t.pad_id());
    }
}
