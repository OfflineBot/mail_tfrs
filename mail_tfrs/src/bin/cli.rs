//! `mail_tfrs` CLI — wraps the library for local training & prediction.

use std::path::PathBuf;
use std::process::ExitCode;

use mail_tfrs::{
    MailInput, find_dataset_files, predict_mail, train_mails,
};
use transformer::{ClassifierArch, ProgressEvent, StopCriteria};

const DEFAULT_MODEL: &str = "bin/mail_model.bin";

fn print_usage() {
    eprintln!(
        r#"mail_tfrs — multi-label email classifier (built on `transformer`)

USAGE:
    mail_tfrs train     [OPTIONS]                     train from scratch
    mail_tfrs continue  [OPTIONS]                     continue training a checkpoint
    mail_tfrs eval      [OPTIONS]                     evaluate on the test split
    mail_tfrs predict   [PREDICT OPTIONS]             predict labels for one email
    mail_tfrs --predict "<text>"                      shortcut for `predict --text "<text>"`

COMMON OPTIONS:
    --model PATH                checkpoint file (default: {DEFAULT_MODEL})
    --seq-len N                 token window (default: 512)
    --d-model N                 model width  (default: 64)
    --n-heads N                 attention heads (default: 4)
    --d-ff N                    FFN width  (default: 128)
    --n-enc-layers N            encoder layers (default: 2)

TRAIN / CONTINUE OPTIONS:
    --files PATH [PATH ...]     dataset file(s); default: scan ~/Downloads/dataset*.ron
    --lr F                      learning rate (default: 1e-3)
    --seed N                    PRNG seed
    --log-every N               training log cadence (default: 50)

STOPPING CRITERIA (combine freely; first to fire stops training):
    --max-steps N
    --max-epochs N
    --target-loss F
    --target-label-acc F
    --target-exact-acc F
    --eval-every N

PREDICT OPTIONS:
    --text "..."                full email text
    --subject "..."             subject     (optional)
    --from "..."                sender name (optional)
    --sender-email "..."        sender addr (optional)
    --body "..."                body text   (required if --text not given)
    --threshold F               cutoff (default: 0.5)
"#,
        DEFAULT_MODEL = DEFAULT_MODEL
    );
}

struct Args {
    positional: Vec<String>,
    flags: std::collections::BTreeMap<String, Vec<String>>,
}

impl Args {
    fn parse(argv: &[String]) -> Self {
        let mut positional = Vec::new();
        let mut flags: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            if let Some(name) = a.strip_prefix("--") {
                let key = name.to_string();
                let mut values = Vec::new();
                let mut j = i + 1;
                while j < argv.len() && !argv[j].starts_with("--") {
                    values.push(argv[j].clone());
                    j += 1;
                }
                flags.entry(key).or_default().extend(values);
                i = j;
            } else {
                positional.push(a.clone());
                i += 1;
            }
        }
        Self { positional, flags }
    }
    fn first(&self, key: &str) -> Option<&str> {
        self.flags.get(key).and_then(|v| v.first()).map(|s| s.as_str())
    }
    fn values(&self, key: &str) -> Option<&[String]> {
        self.flags.get(key).map(|v| v.as_slice())
    }
    fn parse_opt<T: std::str::FromStr>(&self, key: &str) -> Result<Option<T>, String> {
        match self.first(key) {
            None => Ok(None),
            Some(s) => s.parse::<T>().map(Some).map_err(|_| format!("--{key}: cannot parse {s:?}")),
        }
    }
    fn has(&self, key: &str) -> bool { self.flags.contains_key(key) }
}

fn arch(args: &Args) -> Result<ClassifierArch, String> {
    Ok(ClassifierArch {
        d_model:      args.parse_opt::<usize>("d-model")?.unwrap_or(64),
        n_heads:      args.parse_opt::<usize>("n-heads")?.unwrap_or(4),
        d_ff:         args.parse_opt::<usize>("d-ff")?.unwrap_or(128),
        n_enc_layers: args.parse_opt::<usize>("n-enc-layers")?.unwrap_or(2),
        seq_len:      args.parse_opt::<usize>("seq-len")?.unwrap_or(512),
    })
}

fn dataset_paths(args: &Args) -> Result<Vec<PathBuf>, String> {
    if let Some(v) = args.values("files") {
        return Ok(v.iter().map(PathBuf::from).collect());
    }
    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let downloads = PathBuf::from(home).join("Downloads");
    let found = find_dataset_files(&downloads, "dataset")
        .map_err(|e| format!("scan {}: {e}", downloads.display()))?;
    if found.is_empty() {
        return Err(format!(
            "no dataset files found — pass --files PATH [PATH ...] (or place dataset*.ron under {})",
            downloads.display()
        ));
    }
    Ok(found)
}

fn print_progress(ev: ProgressEvent) {
    match ev {
        ProgressEvent::Step { step, epoch, cursor, of, avg_loss } => {
            println!("step {step:>5}  epoch {epoch} ({cursor:>4}/{of})  avg_loss = {avg_loss:.4}");
        }
        ProgressEvent::Eval { step, epoch, mid_epoch, stats } => {
            let tag = if mid_epoch { format!("eval @ step {step}") } else { format!("epoch {epoch} done") };
            println!(
                "{tag} — test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
                stats.loss, stats.label_acc * 100.0, stats.exact_acc * 100.0
            );
        }
        ProgressEvent::Stop { reason, steps, epochs } => {
            println!("training done ({reason}) — steps = {steps}  epochs = {epochs}");
        }
    }
}

fn run_train(args: Args, cont: bool) -> Result<(), String> {
    let arch = arch(&args)?;
    let model_path = args.first("model").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let files = dataset_paths(&args)?;

    let stop = StopCriteria {
        max_steps:        args.parse_opt::<usize>("max-steps")?,
        max_epochs:       args.parse_opt::<usize>("max-epochs")?,
        target_loss:      args.parse_opt::<f32>("target-loss")?,
        target_label_acc: args.parse_opt::<f32>("target-label-acc")?,
        target_exact_acc: args.parse_opt::<f32>("target-exact-acc")?,
        eval_every_steps: args.parse_opt::<usize>("eval-every")?,
    };

    let lr        = args.parse_opt::<f32>("lr")?.unwrap_or(1e-3);
    let seed      = args.parse_opt::<u64>("seed")?.unwrap_or(0xC0FFEE);
    let log_every = args.parse_opt::<usize>("log-every")?.unwrap_or(50);

    println!("dataset files: {} found", files.len());
    for f in &files { println!("  - {}", f.display()); }

    let mut cb = print_progress;
    let cont_path: Option<PathBuf> = if cont { Some(model_path.clone()) } else { None };
    let summary = train_mails(
        &files, &model_path, arch, stop, lr, seed, log_every,
        cont_path,
        Some(&mut cb),
    )?;
    println!(
        "saved model to {} — categories: {:?}",
        summary.model_path.display(), summary.categories
    );
    Ok(())
}

fn run_eval(args: Args) -> Result<(), String> {
    // Eval reuses train_mails with EvalOnly-style stop (max_steps=0 won't work
    // because we need at least one eval). Easiest: load and predict on each.
    let arch = arch(&args)?;
    let model_path = args.first("model").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    let files = dataset_paths(&args)?;

    let tok = transformer::model::tokenizer::HfTokenizer::default_pretrained()
        .map_err(|e| format!("load tokenizer: {e}"))?;
    let ds = mail_tfrs::build_dataset(&files, &tok, arch.seq_len)?;
    let (_, test) = transformer::split_train_test(&ds.samples, 0xC0FFEE, 10);

    let mut mm = mail_tfrs::MailModel::load(&model_path, arch)?;
    let stats = transformer::evaluate(&mut mm.model, &test, ds.num_classes());
    println!(
        "eval — loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%  (n_test = {})",
        stats.loss, stats.label_acc * 100.0, stats.exact_acc * 100.0, test.len()
    );
    Ok(())
}

fn build_predict_input(args: &Args) -> Result<MailInput, String> {
    if let Some(text) = args.first("text") {
        return Ok(MailInput::from_body(text));
    }
    let subject      = args.first("subject").unwrap_or("").to_string();
    let sender_name  = args.first("from").unwrap_or("").to_string();
    let sender_email = args.first("sender-email").unwrap_or("").to_string();
    let body         = args.first("body").unwrap_or("").to_string();
    if subject.is_empty() && sender_name.is_empty() && sender_email.is_empty() && body.is_empty() {
        return Err("predict: provide --text, --body, or any of --subject/--from/--sender-email".into());
    }
    Ok(MailInput { subject, sender_name, sender_email, body })
}

fn run_predict(args: Args) -> Result<(), String> {
    let arch = arch(&args)?;
    let model_path = args.first("model").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    if !model_path.exists() {
        return Err(format!("model not found: {}", model_path.display()));
    }
    let threshold = args.parse_opt::<f32>("threshold")?.unwrap_or(0.5);
    let input = build_predict_input(&args)?;

    println!("model: {}", model_path.display());
    let preds = predict_mail(&model_path, arch, &input, threshold)?;
    let max_name = preds.iter().map(|p| p.name.len()).max().unwrap_or(0);
    for p in &preds {
        let mark = if p.predicted { "*" } else { " " };
        println!(
            "  {mark} {:<width$}  {:>6.2}%  {}",
            p.name, p.probability * 100.0,
            if p.predicted { "yes" } else { "no" },
            width = max_name
        );
    }
    let positives: Vec<&str> = preds.iter().filter(|p| p.predicted).map(|p| p.name.as_str()).collect();
    println!("labels: {:?}", positives);
    Ok(())
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }
    let args = Args::parse(&argv);

    if args.positional.is_empty() && args.has("predict") {
        let text = args.first("predict").unwrap_or("").to_string();
        let mut a2 = args;
        a2.flags.insert("text".to_string(), vec![text]);
        a2.positional.push("predict".to_string());
        return run_predict(a2);
    }

    match args.positional.first().map(|s| s.as_str()).unwrap_or("") {
        "train"               => run_train(args, false),
        "continue" | "resume" => run_train(args, true),
        "eval"                => run_eval(args),
        "predict"             => run_predict(args),
        ""                    => { print_usage(); Err("missing subcommand".into()) }
        other                 => { print_usage(); Err(format!("unknown subcommand: {other:?}")) }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::from(2) }
    }
}
