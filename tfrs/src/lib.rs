//! `transformer` — generic encoder/decoder transformer with a multi-label
//! classifier trainer on top.
//!
//! This crate is domain-agnostic: it knows about token ids, label vectors, and
//! gradient updates. Anything task-specific (e.g. an email loader, an HTTP
//! API) lives in a downstream crate that depends on this one.
//!
//! Typical use from a downstream crate:
//!
//! ```no_run
//! use transformer::{
//!     ClassifierArch, Sample, StopCriteria,
//!     build_encoder_classifier, train_loop, split_train_test, save_classifier,
//! };
//! use transformer::utils::Optimizer;
//!
//! # let samples: Vec<Sample> = vec![];
//! # let categories: Vec<String> = vec![];
//! # let vocab = 32_000;
//! let arch = ClassifierArch::small();
//! let opt  = Optimizer::adam_default(1e-3);
//! let mut model = build_encoder_classifier(arch, vocab, categories.len(), opt);
//!
//! let (train, test) = split_train_test(&samples, 0xC0FFEE, 10);
//! let stop = StopCriteria { max_epochs: Some(3), ..Default::default() };
//! let _ = train_loop(&mut model, &opt, &train, &test, stop, categories.len(), 50, 0xC0FFEE, None);
//!
//! save_classifier(&model, "model.bin", &categories).unwrap();
//! ```

pub mod model;
pub mod train;
pub mod utils;

pub use train::{
    ClassifierArch, EvalStats, ProgressEvent, ProgressFn, Sample, StopCriteria, TrainBudget,
    TrainResult, build_encoder_classifier, evaluate, forward_predict, save_classifier,
    shuffle_indices, split_train_test, train_loop, train_step,
};
