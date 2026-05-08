//! Smoke-test binary for the `transformer` crate.
//!
//! Runs the synthetic copy-task so you can sanity-check the model end-to-end
//! without any dataset. For real workloads use a downstream crate (e.g.
//! `mail_tfrs`) that wires the lib up to your data.

use std::process::ExitCode;

use transformer::train::{TrainConfig, overfit_one_batch, train_copy_task};

fn print_usage() {
    eprintln!(
        r#"tfrs — smoke-test binary for the transformer library

USAGE:
    tfrs copy        run the synthetic copy task (default)
    tfrs overfit     overfit on a single fixed batch (backprop sanity)
    tfrs --help

For real training/serving, build on top of the library — see the
`mail_tfrs` crate for an example.
"#
    );
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let sub = argv.first().map(|s| s.as_str()).unwrap_or("copy");
    match sub {
        "copy"               => { train_copy_task(TrainConfig::small_copy());     ExitCode::SUCCESS }
        "overfit"            => { overfit_one_batch(TrainConfig::small_copy());   ExitCode::SUCCESS }
        "-h" | "--help"      => { print_usage();                                  ExitCode::SUCCESS }
        other                => { eprintln!("unknown subcommand: {other:?}");
                                   print_usage();                                 ExitCode::from(2) }
    }
}
