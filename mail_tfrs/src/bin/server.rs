//! HTTP server for `mail_tfrs`. Endpoints:
//!
//!   GET  /health                  → "ok"
//!   GET  /info                    → loaded model metadata (or 503 if no model)
//!   POST /predict                 → JSON MailInput → JSON labels
//!   POST /train                   → JSON TrainRequest → JSON TrainSummary (sync)
//!
//! Train requests are processed serially, guarded by a global mutex, so you
//! cannot accidentally kick off two training runs in parallel. Predictions
//! reuse the cached `MailModel` between calls.
//!
//! Configuration via env vars:
//!   MAIL_TFRS_BIND      bind address              (default 127.0.0.1:8080)
//!   MAIL_TFRS_MODEL     model checkpoint path     (default ./bin/mail_model.bin)
//!   MAIL_TFRS_DATASETS  default dataset directory (default ~/Downloads)
//!
//! Architecture is fixed at startup from MAIL_TFRS_D_MODEL, MAIL_TFRS_N_HEADS,
//! MAIL_TFRS_D_FF, MAIL_TFRS_N_ENC_LAYERS, MAIL_TFRS_SEQ_LEN (all optional;
//! defaults match the CLI).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use mail_tfrs::{MailInput, MailModel, find_dataset_files, train_mails};
use transformer::{ClassifierArch, ProgressEvent, StopCriteria};

// ============================ env-driven config =============================

struct AppConfig {
    bind: String,
    model_path: PathBuf,
    dataset_dir: PathBuf,
    arch: ClassifierArch,
    threshold: f32,
}

impl AppConfig {
    fn from_env() -> Self {
        fn parse<T: std::str::FromStr>(key: &str, default: T) -> T {
            std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Self {
            bind: std::env::var("MAIL_TFRS_BIND").unwrap_or_else(|_| "127.0.0.1:8080".into()),
            model_path: std::env::var("MAIL_TFRS_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("bin/mail_model.bin")),
            dataset_dir: std::env::var("MAIL_TFRS_DATASETS")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(home).join("Downloads")),
            arch: ClassifierArch {
                d_model:      parse("MAIL_TFRS_D_MODEL", 64),
                n_heads:      parse("MAIL_TFRS_N_HEADS", 4),
                d_ff:         parse("MAIL_TFRS_D_FF", 128),
                n_enc_layers: parse("MAIL_TFRS_N_ENC_LAYERS", 2),
                seq_len:      parse("MAIL_TFRS_SEQ_LEN", 512),
            },
            threshold: parse("MAIL_TFRS_THRESHOLD", 0.5f32),
        }
    }
}

// ============================ request bodies ================================

#[derive(Debug, Deserialize)]
struct PredictRequest {
    #[serde(flatten)]
    input: MailInput,
    #[serde(default)]
    threshold: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct TrainRequest {
    /// Explicit dataset paths. If empty, scans the configured dataset dir for
    /// `dataset*.ron`.
    #[serde(default)]
    files: Vec<PathBuf>,
    #[serde(default)]
    fresh: bool,
    #[serde(default)] max_steps: Option<usize>,
    #[serde(default)] max_epochs: Option<usize>,
    #[serde(default)] target_loss: Option<f32>,
    #[serde(default)] target_label_acc: Option<f32>,
    #[serde(default)] target_exact_acc: Option<f32>,
    #[serde(default)] eval_every_steps: Option<usize>,
    #[serde(default)] lr: Option<f32>,
    #[serde(default)] seed: Option<u64>,
    #[serde(default)] log_every: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ErrorBody { error: String }

#[derive(Debug, Serialize)]
struct InfoBody {
    model_path: String,
    arch: ArchOut,
    categories: Vec<String>,
    threshold: f32,
}

#[derive(Debug, Serialize)]
struct ArchOut {
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    n_enc_layers: usize,
    seq_len: usize,
}

impl From<ClassifierArch> for ArchOut {
    fn from(a: ClassifierArch) -> Self {
        Self {
            d_model: a.d_model, n_heads: a.n_heads, d_ff: a.d_ff,
            n_enc_layers: a.n_enc_layers, seq_len: a.seq_len,
        }
    }
}

// ============================ helpers =======================================

fn json_response<T: Serialize>(status: u16, body: &T) -> Response<std::io::Cursor<Vec<u8>>> {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|_| b"{}".to_vec());
    Response::from_data(bytes)
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
}

fn err(status: u16, msg: impl Into<String>) -> Response<std::io::Cursor<Vec<u8>>> {
    json_response(status, &ErrorBody { error: msg.into() })
}

fn read_body(req: &mut Request) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    req.as_reader().read_to_end(&mut buf).map_err(|e| format!("read body: {e}"))?;
    Ok(buf)
}

fn resolve_files(req_files: Vec<PathBuf>, dataset_dir: &std::path::Path) -> Result<Vec<PathBuf>, String> {
    if !req_files.is_empty() {
        for p in &req_files {
            if !p.exists() {
                return Err(format!("dataset file not found: {}", p.display()));
            }
        }
        return Ok(req_files);
    }
    let found = find_dataset_files(dataset_dir, "dataset")
        .map_err(|e| format!("scan {}: {e}", dataset_dir.display()))?;
    if found.is_empty() {
        return Err(format!(
            "no dataset files in {}; pass `files` in the request body",
            dataset_dir.display()
        ));
    }
    Ok(found)
}

// ============================ handlers ======================================

fn handle_health() -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string("ok\n")
        .with_status_code(StatusCode(200))
        .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/plain"[..]).unwrap())
}

fn handle_info(cfg: &AppConfig, model: &Mutex<Option<MailModel>>) -> Response<std::io::Cursor<Vec<u8>>> {
    let guard = model.lock().unwrap();
    match guard.as_ref() {
        Some(m) => json_response(200, &InfoBody {
            model_path: cfg.model_path.display().to_string(),
            arch: m.arch.into(),
            categories: m.categories.clone(),
            threshold: cfg.threshold,
        }),
        None => err(503, format!("no model loaded (expected at {})", cfg.model_path.display())),
    }
}

fn handle_predict(
    req: &mut Request,
    cfg: &AppConfig,
    model: &Mutex<Option<MailModel>>,
) -> Response<std::io::Cursor<Vec<u8>>> {
    let body = match read_body(req) { Ok(b) => b, Err(e) => return err(400, e) };
    let parsed: PredictRequest = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => return err(400, format!("invalid JSON: {e}")),
    };
    let threshold = parsed.threshold.unwrap_or(cfg.threshold);

    let mut guard = model.lock().unwrap();
    let mm = match guard.as_mut() {
        Some(m) => m,
        None => return err(503, format!("no model loaded (expected at {})", cfg.model_path.display())),
    };
    let preds = mm.predict(&parsed.input, threshold);
    json_response(200, &preds)
}

fn handle_train(
    req: &mut Request,
    cfg: &AppConfig,
    model: &Mutex<Option<MailModel>>,
    train_lock: &Mutex<()>,
) -> Response<std::io::Cursor<Vec<u8>>> {
    let _guard = match train_lock.try_lock() {
        Ok(g) => g,
        Err(_) => return err(409, "another training run is in progress"),
    };

    let body = match read_body(req) { Ok(b) => b, Err(e) => return err(400, e) };
    let parsed: TrainRequest = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => return err(400, format!("invalid JSON: {e}")),
    };

    let files = match resolve_files(parsed.files.clone(), &cfg.dataset_dir) {
        Ok(f) => f,
        Err(e) => return err(400, e),
    };

    let stop = StopCriteria {
        max_steps:        parsed.max_steps,
        max_epochs:       parsed.max_epochs,
        target_loss:      parsed.target_loss,
        target_label_acc: parsed.target_label_acc,
        target_exact_acc: parsed.target_exact_acc,
        eval_every_steps: parsed.eval_every_steps,
    };
    let lr        = parsed.lr.unwrap_or(1e-3);
    let seed      = parsed.seed.unwrap_or(0xC0FFEE);
    let log_every = parsed.log_every.unwrap_or(50);

    // drop any cached model so we don't hold a stale copy of the old weights
    // while training writes a new file
    {
        let mut guard = model.lock().unwrap();
        *guard = None;
    }

    let cont_path: Option<PathBuf> = if parsed.fresh { None } else if cfg.model_path.exists() {
        Some(cfg.model_path.clone())
    } else {
        None
    };

    let mut cb = move |ev: ProgressEvent| {
        match ev {
            ProgressEvent::Step { step, epoch, cursor, of, avg_loss } => {
                eprintln!("step {step:>5}  epoch {epoch} ({cursor:>4}/{of})  avg_loss = {avg_loss:.4}");
            }
            ProgressEvent::Eval { step, epoch, mid_epoch, stats } => {
                let tag = if mid_epoch { format!("eval @ step {step}") } else { format!("epoch {epoch} done") };
                eprintln!(
                    "{tag} — test loss = {:.4}  label_acc = {:.2}%  exact_acc = {:.2}%",
                    stats.loss, stats.label_acc * 100.0, stats.exact_acc * 100.0
                );
            }
            ProgressEvent::Stop { reason, steps, epochs } => {
                eprintln!("training done ({reason}) — steps = {steps}  epochs = {epochs}");
            }
        }
    };

    let result = train_mails(
        &files, &cfg.model_path, cfg.arch, stop, lr, seed, log_every,
        cont_path,
        Some(&mut cb),
    );
    let summary = match result {
        Ok(s) => s,
        Err(e) => return err(500, format!("training failed: {e}")),
    };

    // reload the freshly-trained model into the cache
    match MailModel::load(&cfg.model_path, cfg.arch) {
        Ok(m) => { *model.lock().unwrap() = Some(m); }
        Err(e) => eprintln!("warn: could not reload model after training: {e}"),
    }
    json_response(200, &summary)
}

// ============================ main ==========================================

fn try_load_initial_model(cfg: &AppConfig) -> Option<MailModel> {
    if !cfg.model_path.exists() { return None; }
    match MailModel::load(&cfg.model_path, cfg.arch) {
        Ok(m) => {
            eprintln!("loaded model from {}", cfg.model_path.display());
            Some(m)
        }
        Err(e) => {
            eprintln!("warn: failed to load {}: {e}", cfg.model_path.display());
            None
        }
    }
}

fn run() -> Result<(), String> {
    let cfg = AppConfig::from_env();
    eprintln!("mail_tfrs server");
    eprintln!("  bind:      {}", cfg.bind);
    eprintln!("  model:     {}", cfg.model_path.display());
    eprintln!("  datasets:  {}", cfg.dataset_dir.display());
    eprintln!("  arch:      d_model={} n_heads={} d_ff={} n_enc_layers={} seq_len={}",
              cfg.arch.d_model, cfg.arch.n_heads, cfg.arch.d_ff,
              cfg.arch.n_enc_layers, cfg.arch.seq_len);

    let server = Server::http(&cfg.bind)
        .map_err(|e| format!("bind {}: {e}", cfg.bind))?;
    eprintln!("listening on http://{}", cfg.bind);

    let model: Mutex<Option<MailModel>> = Mutex::new(try_load_initial_model(&cfg));
    let train_lock: Mutex<()> = Mutex::new(());

    for mut req in server.incoming_requests() {
        let method = req.method().clone();
        let url = req.url().to_string();
        // match on the path only (drop query string)
        let path = url.split('?').next().unwrap_or("").to_string();

        let resp = match (&method, path.as_str()) {
            (Method::Get,  "/health")  => handle_health(),
            (Method::Get,  "/info")    => handle_info(&cfg, &model),
            (Method::Post, "/predict") => handle_predict(&mut req, &cfg, &model),
            (Method::Post, "/train")   => handle_train(&mut req, &cfg, &model, &train_lock),
            (Method::Get,  "/")        => handle_health(),
            _ => err(404, format!("no route for {method} {path}")),
        };

        if let Err(e) = req.respond(resp) {
            eprintln!("warn: respond error: {e}");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("error: {e}"); ExitCode::from(2) }
    }
}
