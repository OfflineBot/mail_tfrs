//! Email-classification glue on top of the generic `transformer` crate.
//!
//! Provides:
//! - `MailInput` and `mail_to_text` — canonical text format the model sees
//! - `mail::*` — RON dataset loader (`dataset_*.ron`)
//! - `tokenize_mail` — turn a `MailInput` into a fixed-length id sequence
//! - `train_mails` / `predict_mail` — thin wrappers around the lib's classifier
//!
//! The `cli` and `server` binaries in this crate are end-user entrypoints.

pub mod mail;
pub mod pipeline;

pub use mail::{
    Attachment, Dataset, Mail, MailDataset, Status, build_dataset, find_dataset_files,
    load_dataset_file, mail_to_text, parse_labels, strip_html,
};
pub use pipeline::{
    MailInput, MailModel, PredictedLabel, TrainSummary, predict_mail, tokenize_mail, train_mails,
};
