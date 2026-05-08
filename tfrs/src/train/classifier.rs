//! Generic multi-label classifier on top of `Transformer`.
//!
//! Domain-agnostic: the trainer takes already-tokenized `Sample`s (ids + label
//! vector) and knows nothing about where the text came from. Wrap this from a
//! downstream crate (e.g. an email classifier) by tokenizing your inputs and
//! producing `Sample`s.

use std::path::Path;

use ndarray::{Array2, Axis};

use crate::{
    model::{
        encoder::EncoderConfig,
        transformer::Transformer,
    },
    utils::{Activation, Loss, Optimizer, Trainable},
};

/// One training/eval example: token ids and a 0/1 label vector of length
/// `num_classes`.
#[derive(Debug, Clone)]
pub struct Sample {
    pub ids: Vec<usize>,
    pub labels: Vec<f32>,
}

/// Architecture parameters shared between training and prediction. Must match
/// across save/load — used to construct an empty `Transformer` whose shapes
/// line up with the on-disk weights.
#[derive(Debug, Clone, Copy)]
pub struct ClassifierArch {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_enc_layers: usize,
    pub seq_len: usize,
}

impl ClassifierArch {
    pub fn small() -> Self {
        Self { d_model: 64, n_heads: 4, d_ff: 128, n_enc_layers: 2, seq_len: 96 }
    }
}

/// How many gradient updates to do.
#[derive(Clone, Copy, Debug)]
pub enum TrainBudget {
    Steps(usize),
    Epochs(usize),
}

/// Stopping criteria. Any combination may be set; training stops as soon as
/// **any** active criterion fires (logical OR).
///
/// Eval-based criteria (`target_loss`, `target_label_acc`, `target_exact_acc`)
/// are checked at every epoch boundary and additionally every
/// `eval_every_steps` steps if that field is set.
#[derive(Clone, Copy, Debug, Default)]
pub struct StopCriteria {
    pub max_steps: Option<usize>,
    pub max_epochs: Option<usize>,
    pub target_loss: Option<f32>,
    pub target_label_acc: Option<f32>,
    pub target_exact_acc: Option<f32>,
    pub eval_every_steps: Option<usize>,
}

impl StopCriteria {
    pub fn from_budget(b: TrainBudget) -> Self {
        let mut s = Self::default();
        match b {
            TrainBudget::Steps(n)  => s.max_steps  = Some(n),
            TrainBudget::Epochs(n) => s.max_epochs = Some(n),
        }
        s
    }
    pub fn any_set(&self) -> bool {
        self.max_steps.is_some()
            || self.max_epochs.is_some()
            || self.target_loss.is_some()
            || self.target_label_acc.is_some()
            || self.target_exact_acc.is_some()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EvalStats {
    pub loss: f32,
    pub label_acc: f32,
    pub exact_acc: f32,
}

#[derive(Debug, Clone)]
pub struct TrainResult {
    pub steps: usize,
    pub epochs: usize,
    pub stopped_for: &'static str,
    pub final_eval: EvalStats,
}

/// Construct an encoder-only classifier with the given architecture, ready to
/// train or to load a checkpoint into.
pub fn build_encoder_classifier(
    arch: ClassifierArch,
    vocab: usize,
    num_classes: usize,
    opt: Optimizer,
) -> Transformer {
    // tgt_vocab = 2 — placeholder, decoder-side embedding never runs in encoder-only mode.
    let mut model = Transformer::new_empty(arch.d_model, vocab, 2, num_classes);
    let enc_cfg = EncoderConfig::new(
        arch.n_heads, 1e-5, arch.d_ff, Activation::ReLU, Loss::MSE, opt,
    );
    model.set_encoder_configs((0..arch.n_enc_layers).map(|_| enc_cfg).collect());
    model
}

/// Run the encoder over `ids`, mean-pool over the sequence, sigmoid → per-class
/// probabilities. Length of returned vec equals `num_classes`.
pub fn forward_predict(model: &mut Transformer, ids: &[usize]) -> Vec<f32> {
    let logits = model.forward(Some(ids), None);
    let mean = logits.mean_axis(Axis(0)).unwrap();
    mean.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

/// Single forward+backward+update on one sample. Returns the per-sample
/// (sum-over-classes) binary cross-entropy loss.
pub fn train_step(
    model: &mut Transformer,
    opt: &Optimizer,
    s: &Sample,
    num_classes: usize,
) -> f32 {
    model.clear_grads();
    let logits = model.forward(Some(&s.ids), None);
    let seq    = s.ids.len();

    let mean = logits.mean_axis(Axis(0)).unwrap();
    let mut probs = vec![0f32; num_classes];
    let mut loss  = 0f32;
    for c in 0..num_classes {
        let p = (1.0 / (1.0 + (-mean[c]).exp())).clamp(1e-7, 1.0 - 1e-7);
        probs[c] = p;
        let y = s.labels[c];
        loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
    }

    let mut delta = Array2::<f32>::zeros((seq, num_classes));
    for r in 0..seq {
        for c in 0..num_classes {
            delta[[r, c]] = (probs[c] - s.labels[c]) / seq as f32;
        }
    }
    model.backward(delta);
    model.update(opt);
    loss
}

/// Evaluate on a held-out set: average BCE loss, per-label accuracy, and
/// exact-match accuracy (all labels right on the same sample).
pub fn evaluate(model: &mut Transformer, samples: &[Sample], num_classes: usize) -> EvalStats {
    if samples.is_empty() {
        return EvalStats::default();
    }
    let mut total_loss = 0f32;
    let mut correct_labels = 0usize;
    let mut total_labels = 0usize;
    let mut exact_matches = 0usize;

    for s in samples {
        let probs = forward_predict(model, &s.ids);
        let mut all_match = true;
        for c in 0..num_classes {
            let p = probs[c].clamp(1e-7, 1.0 - 1e-7);
            let y = s.labels[c];
            total_loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
            let pred = if probs[c] > 0.5 { 1.0 } else { 0.0 };
            if pred == y { correct_labels += 1; } else { all_match = false; }
            total_labels += 1;
        }
        if all_match { exact_matches += 1; }
    }
    EvalStats {
        loss:      total_loss / samples.len() as f32,
        label_acc: correct_labels as f32 / total_labels as f32,
        exact_acc: exact_matches as f32 / samples.len() as f32,
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Fisher–Yates shuffle of `0..n` using a deterministic xorshift seeded from `seed`.
pub fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    for i in (1..n).rev() {
        let j = (xorshift64(&mut state) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Hook called after every evaluation so callers can stream progress out
/// (e.g. an HTTP server pushing log lines to a client). `step` is `None` for
/// epoch-boundary evals, `Some(step)` for mid-epoch evals.
pub type ProgressFn<'a> = &'a mut dyn FnMut(ProgressEvent);

#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Step  { step: usize, epoch: usize, cursor: usize, of: usize, avg_loss: f32 },
    Eval  { step: usize, epoch: usize, mid_epoch: bool, stats: EvalStats },
    Stop  { reason: &'static str, steps: usize, epochs: usize },
}

/// Full training loop. `train` and `test` are owned slices of samples already
/// tokenized to the same `seq_len`. The loop:
/// - shuffles `train` deterministically per epoch
/// - calls `train_step` per sample, logs `running_loss` every `log_every`
/// - evaluates on `test` at every epoch boundary and (optionally) mid-epoch
/// - stops on the first criterion in `stop` that fires
///
/// Pass `progress` to receive structured events (CLI prints, server streaming).
pub fn train_loop(
    model: &mut Transformer,
    opt: &Optimizer,
    train: &[&Sample],
    test: &[Sample],
    stop: StopCriteria,
    num_classes: usize,
    log_every: usize,
    seed: u64,
    mut progress: Option<ProgressFn<'_>>,
) -> TrainResult {
    let stop = if stop.any_set() {
        stop
    } else {
        StopCriteria { max_epochs: Some(1), ..Default::default() }
    };

    let mut step_state = seed;
    let mut order: Vec<usize> = (0..train.len()).collect();
    for i in (1..order.len()).rev() {
        let j = (xorshift64(&mut step_state) as usize) % (i + 1);
        order.swap(i, j);
    }

    let mut running = 0f32;
    let mut running_n = 0usize;
    let mut cursor = 0usize;
    let mut epoch  = 0usize;
    let mut step   = 0usize;
    let mut last_eval: Option<EvalStats> = None;
    let stopped_for: &'static str;

    let check_eval_targets = |s: EvalStats| -> Option<&'static str> {
        if let Some(t) = stop.target_loss      { if s.loss      <= t { return Some("target_loss"); } }
        if let Some(t) = stop.target_label_acc { if s.label_acc >= t { return Some("target_label_acc"); } }
        if let Some(t) = stop.target_exact_acc { if s.exact_acc >= t { return Some("target_exact_acc"); } }
        None
    };

    loop {
        if let Some(m) = stop.max_steps {
            if step >= m { stopped_for = "max_steps"; break; }
        }

        if cursor >= order.len() {
            let stats = evaluate(model, test, num_classes);
            last_eval = Some(stats);
            if let Some(p) = progress.as_deref_mut() {
                p(ProgressEvent::Eval { step, epoch, mid_epoch: false, stats });
            }
            if let Some(reason) = check_eval_targets(stats) { stopped_for = reason; break; }
            epoch += 1;
            if let Some(m) = stop.max_epochs {
                if epoch >= m { stopped_for = "max_epochs"; break; }
            }
            cursor = 0;
            for i in (1..order.len()).rev() {
                let j = (xorshift64(&mut step_state) as usize) % (i + 1);
                order.swap(i, j);
            }
            running = 0.0;
            running_n = 0;
        }

        let i = order[cursor];
        cursor += 1;
        let l = train_step(model, opt, train[i], num_classes);
        running   += l;
        running_n += 1;
        step += 1;

        if log_every > 0 && step % log_every == 0 {
            if let Some(p) = progress.as_deref_mut() {
                p(ProgressEvent::Step {
                    step, epoch, cursor, of: order.len(),
                    avg_loss: running / running_n as f32,
                });
            }
        }

        if let Some(every) = stop.eval_every_steps {
            if every > 0 && step % every == 0 {
                let stats = evaluate(model, test, num_classes);
                last_eval = Some(stats);
                if let Some(p) = progress.as_deref_mut() {
                    p(ProgressEvent::Eval { step, epoch, mid_epoch: true, stats });
                }
                if let Some(reason) = check_eval_targets(stats) { stopped_for = reason; break; }
            }
        }
    }

    let final_eval = last_eval.unwrap_or_else(|| evaluate(model, test, num_classes));
    if let Some(p) = progress.as_deref_mut() {
        p(ProgressEvent::Stop { reason: stopped_for, steps: step, epochs: epoch });
    }
    TrainResult { steps: step, epochs: epoch, stopped_for, final_eval }
}

/// Convenience: split a flat sample list into `train` (refs) and `test` (owned)
/// using a deterministic shuffle and a 1/`test_frac_inv` test split.
pub fn split_train_test(samples: &[Sample], seed: u64, test_frac_inv: usize)
    -> (Vec<&Sample>, Vec<Sample>)
{
    let order = shuffle_indices(samples.len(), seed);
    let n_test = (samples.len() / test_frac_inv.max(1)).max(1);
    let n_train = samples.len() - n_test;
    let train: Vec<&Sample> = order[..n_train].iter().map(|&i| &samples[i]).collect();
    let test:  Vec<Sample>  = order[n_train..].iter().map(|&i| samples[i].clone()).collect();
    (train, test)
}

/// Save the model + category names to disk, creating parent directories.
pub fn save_classifier<P: AsRef<Path>>(
    model: &Transformer,
    path: P,
    categories: &[String],
) -> std::io::Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent).ok();
    }
    model.save_to_file(path, categories)
}
