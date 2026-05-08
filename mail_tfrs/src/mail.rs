//! Loader for the email-classification dataset shipped as `dataset_*.ron`.
//!
//! The exporter writes a `Dataset(...)` value containing a flat list of
//! `Mail(...)` records with multi-label "1,0,..." strings over N categories.
//! We deserialize, optionally strip HTML, and emit `(token_ids, label_vec)`
//! samples suitable for the `transformer` crate's classifier.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use transformer::Sample;
use transformer::model::tokenizer::{HfTokenizer, Tokenizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Dataset {
    pub version: u32,
    pub exported_at: String,
    pub categories: Vec<String>,
    pub mail_count: usize,
    pub mails: Vec<Mail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Mail {
    pub id: String,
    pub user: String,
    pub folder: String,
    pub subject: String,
    pub sender_name: String,
    pub sender_email: String,
    #[serde(default)]
    pub to: Vec<String>,
    #[serde(default)]
    pub cc: Vec<String>,
    #[serde(default)]
    pub bcc: Vec<String>,
    pub received: String,
    #[serde(default)]
    pub body_plain: String,
    #[serde(default)]
    pub body_html: String,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    #[serde(default)]
    pub conversation_id: String,
    pub labels: String,
    pub status: Status,
    pub touched_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Attachment {
    pub name: String,
    pub mime: String,
    pub size: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Status {
    Tagged,
    Untagged,
    Pending,
    Skipped,
}

pub struct MailDataset {
    pub categories: Vec<String>,
    pub samples: Vec<Sample>,
    pub seq_len: usize,
}

impl MailDataset {
    pub fn num_classes(&self) -> usize {
        self.categories.len()
    }
}

pub fn parse_labels(s: &str, expected: usize) -> Vec<f32> {
    let parts: Vec<f32> = s
        .split(',')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| if p == "1" { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(
        parts.len(),
        expected,
        "label vector length {} != categories {} (raw={s:?})",
        parts.len(),
        expected
    );
    parts
}

/// Strip HTML tags and decode a few common entities. Good enough for cleaning
/// noisy email body_html into bag-of-tokens-ready text.
pub fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if in_tag {
            if c == '>' {
                in_tag = false;
            }
            i += 1;
            continue;
        }
        if c == '<' {
            let lower = html[i..].to_lowercase();
            if lower.starts_with("<script") {
                in_script = true;
            } else if lower.starts_with("<style") {
                in_style = true;
            } else if in_script && lower.starts_with("</script") {
                in_script = false;
            } else if in_style && lower.starts_with("</style") {
                in_style = false;
            }
            in_tag = true;
            i += 1;
            continue;
        }
        if !in_script && !in_style {
            out.push(c);
        }
        i += 1;
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#43;", "+")
        .replace("&#39;", "'");

    let mut prev_ws = false;
    let mut cleaned = String::with_capacity(decoded.len());
    for ch in decoded.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                cleaned.push(' ');
                prev_ws = true;
            }
        } else {
            cleaned.push(ch);
            prev_ws = false;
        }
    }
    cleaned.trim().to_string()
}

/// Canonical text format the model sees: subject + sender + body. Identical to
/// what the trainer feeds in, so prediction stays in-distribution.
pub fn mail_to_text(m: &Mail) -> String {
    let body = if !m.body_plain.trim().is_empty() {
        m.body_plain.clone()
    } else {
        strip_html(&m.body_html)
    };
    format!(
        "{} | {} <{}> | {}",
        m.subject.trim(),
        m.sender_name.trim(),
        m.sender_email.trim(),
        body
    )
}

pub fn find_dataset_files(dir: &Path, prefix: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let e = entry?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(prefix) && name.ends_with(".ron") {
            out.push(e.path());
        }
    }
    out.sort();
    Ok(out)
}

pub fn load_dataset_file<P: AsRef<Path>>(path: P) -> Result<Dataset, String> {
    let text = fs::read_to_string(&path).map_err(|e| format!("read {:?}: {}", path.as_ref(), e))?;
    ron::from_str::<Dataset>(&text).map_err(|e| format!("parse {:?}: {}", path.as_ref(), e))
}

/// Build a `MailDataset` from one or more `.ron` files, tokenizing each mail
/// and padding/truncating to `seq_len`. All files must share the same
/// `categories` list (order-sensitive).
pub fn build_dataset<P: AsRef<Path>>(
    paths: &[P],
    tok: &HfTokenizer,
    seq_len: usize,
) -> Result<MailDataset, String> {
    let mut categories: Option<Vec<String>> = None;
    let mut samples = Vec::new();

    for p in paths {
        let ds = load_dataset_file(p)?;
        match &categories {
            None => categories = Some(ds.categories.clone()),
            Some(c) if c != &ds.categories => {
                return Err(format!(
                    "category list mismatch in {:?}: {:?} vs {:?}",
                    p.as_ref(),
                    c,
                    ds.categories
                ));
            }
            _ => {}
        }
        let n_classes = ds.categories.len();
        for m in ds.mails {
            let text = mail_to_text(&m);
            let mut ids = tok.encode(&text);
            ids.insert(0, tok.bos_id());
            let ids = tok.pad_to(ids, seq_len);
            let labels = parse_labels(&m.labels, n_classes);
            samples.push(Sample { ids, labels });
        }
    }

    let categories = categories.ok_or_else(|| "no dataset files loaded".to_string())?;
    Ok(MailDataset { categories, samples, seq_len })
}
