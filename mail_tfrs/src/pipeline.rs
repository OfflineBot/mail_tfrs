//! High-level mail training/prediction pipeline. Wraps the generic classifier
//! from the `transformer` crate with mail-specific tokenization and the
//! on-disk dataset format.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use transformer::model::tokenizer::{HfTokenizer, Tokenizer};
use transformer::model::transformer::{Transformer, TransformerHeader};
use transformer::utils::Optimizer;
use transformer::{
    ClassifierArch, ProgressEvent, StopCriteria, build_encoder_classifier,
    forward_predict, save_classifier, split_train_test, train_loop,
};

use crate::mail::{Mail, MailDataset, build_dataset};

/// Inputs for a single email prediction.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct MailInput {
    #[serde(default)] pub subject: String,
    #[serde(default)] pub sender_name: String,
    #[serde(default)] pub sender_email: String,
    #[serde(default)] pub body: String,
}

impl MailInput {
    pub fn from_body(body: impl Into<String>) -> Self {
        Self { body: body.into(), ..Default::default() }
    }
    /// Mirrors `mail_to_text` so prediction sees the same shape as training.
    pub fn to_text(&self) -> String {
        format!(
            "{} | {} <{}> | {}",
            self.subject.trim(),
            self.sender_name.trim(),
            self.sender_email.trim(),
            self.body
        )
    }
}

impl From<&Mail> for MailInput {
    fn from(m: &Mail) -> Self {
        Self {
            subject: m.subject.clone(),
            sender_name: m.sender_name.clone(),
            sender_email: m.sender_email.clone(),
            body: if !m.body_plain.trim().is_empty() {
                m.body_plain.clone()
            } else {
                crate::mail::strip_html(&m.body_html)
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PredictedLabel {
    pub name: String,
    pub probability: f32,
    pub predicted: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainSummary {
    pub steps: usize,
    pub epochs: usize,
    pub stopped_for: String,
    pub eval_loss: f32,
    pub eval_label_acc: f32,
    pub eval_exact_acc: f32,
    pub categories: Vec<String>,
    pub model_path: PathBuf,
    pub num_train: usize,
    pub num_test: usize,
}

/// Tokenize a `MailInput` into a `seq_len`-long id sequence the same way
/// the trainer does.
pub fn tokenize_mail(tok: &HfTokenizer, input: &MailInput, seq_len: usize) -> Vec<usize> {
    let text = input.to_text();
    let mut ids = tok.encode(&text);
    ids.insert(0, tok.bos_id());
    tok.pad_to(ids, seq_len)
}

/// Loaded model + everything you need to predict with it.
pub struct MailModel {
    pub model: Transformer,
    pub tokenizer: HfTokenizer,
    pub arch: ClassifierArch,
    pub categories: Vec<String>,
}

impl MailModel {
    pub fn load<P: AsRef<Path>>(path: P, arch: ClassifierArch) -> Result<Self, String> {
        let tok = HfTokenizer::default_pretrained()
            .map_err(|e| format!("load tokenizer: {e}"))?;
        let header: TransformerHeader = Transformer::read_header(&path)
            .map_err(|e| format!("read header {}: {e}", path.as_ref().display()))?;
        if header.d_model != arch.d_model {
            return Err(format!("d_model mismatch: file={} cfg={}", header.d_model, arch.d_model));
        }
        if header.n_enc_layers != arch.n_enc_layers {
            return Err(format!("n_enc_layers mismatch: file={} cfg={}", header.n_enc_layers, arch.n_enc_layers));
        }
        if header.src_vocab != tok.vocab_size() {
            return Err(format!("vocab mismatch: file={} tok={}", header.src_vocab, tok.vocab_size()));
        }

        let opt = Optimizer::adam_default(1e-3);
        let mut model = build_encoder_classifier(arch, tok.vocab_size(), header.num_classes, opt);
        let categories = model.load_from_file(&path)
            .map_err(|e| format!("load {}: {e}", path.as_ref().display()))?;

        Ok(Self { model, tokenizer: tok, arch, categories })
    }

    pub fn predict(&mut self, input: &MailInput, threshold: f32) -> Vec<PredictedLabel> {
        let ids = tokenize_mail(&self.tokenizer, input, self.arch.seq_len);
        let probs = forward_predict(&mut self.model, &ids);
        self.categories.iter().enumerate().map(|(i, name)| PredictedLabel {
            name: name.clone(),
            probability: probs[i],
            predicted: probs[i] >= threshold,
        }).collect()
    }
}

/// Single-call prediction: load the model and run one input through it.
pub fn predict_mail<P: AsRef<Path>>(
    model_path: P,
    arch: ClassifierArch,
    input: &MailInput,
    threshold: f32,
) -> Result<Vec<PredictedLabel>, String> {
    MailModel::load(model_path, arch).map(|mut m| m.predict(input, threshold))
}

/// Top-level training entry point. Loads `paths`, splits 90/10, trains under
/// `stop`, saves to `save_path`, returns a summary. `progress` is invoked for
/// streaming logs (CLI prints, server SSE, etc.).
pub fn train_mails<P: AsRef<Path>>(
    paths: &[PathBuf],
    save_path: &Path,
    arch: ClassifierArch,
    stop: StopCriteria,
    lr: f32,
    seed: u64,
    log_every: usize,
    continue_from: Option<P>,
    progress: Option<&mut dyn FnMut(ProgressEvent)>,
) -> Result<TrainSummary, String> {
    let tok = HfTokenizer::default_pretrained()
        .map_err(|e| format!("load tokenizer: {e}"))?;

    let ds: MailDataset = build_dataset(paths, &tok, arch.seq_len)?;
    if ds.samples.is_empty() {
        return Err("dataset is empty".into());
    }

    let (train, test) = split_train_test(&ds.samples, seed, 10);
    let num_train = train.len();
    let num_test  = test.len();
    let num_classes = ds.num_classes();

    let opt = Optimizer::adam_default(lr);
    let mut model = build_encoder_classifier(arch, tok.vocab_size(), num_classes, opt);

    if let Some(cp) = continue_from {
        let _ = model.load_from_file(cp.as_ref())
            .map_err(|e| format!("load checkpoint {}: {e}", cp.as_ref().display()))?;
    }

    let result = train_loop(
        &mut model, &opt, &train, &test, stop, num_classes, log_every, seed, progress,
    );

    save_classifier(&model, save_path, &ds.categories)
        .map_err(|e| format!("save {}: {e}", save_path.display()))?;

    Ok(TrainSummary {
        steps: result.steps,
        epochs: result.epochs,
        stopped_for: result.stopped_for.to_string(),
        eval_loss: result.final_eval.loss,
        eval_label_acc: result.final_eval.label_acc,
        eval_exact_acc: result.final_eval.exact_acc,
        categories: ds.categories,
        model_path: save_path.to_path_buf(),
        num_train,
        num_test,
    })
}

