use crate::{
    model::{
        decoder::DecoderConfig,
        encoder::EncoderConfig,
        transformer::Transformer,
    },
    train::data::CopyTask,
    utils::{
        Activation, Loss, Optimizer, Trainable,
        cross_entropy_grad, cross_entropy_loss,
    },
};

pub struct TrainConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_enc_layers: usize,
    pub n_dec_layers: usize,
    pub vocab: usize,
    pub seq_len: usize,
    pub steps: usize,
    pub lr: f32,
    pub log_every: usize,
    pub seed: u64,
    pub use_adam: bool,
}

impl TrainConfig {
    pub fn small_copy() -> Self {
        Self {
            d_model: 16,
            n_heads: 2,
            d_ff: 32,
            n_enc_layers: 1,
            n_dec_layers: 1,
            vocab: 8,
            seq_len: 6,
            steps: 1000,
            lr: 1e-3,
            log_every: 25,
            seed: 0xDEAD_BEEF,
            use_adam: false,
        }
    }

    fn make_optimizer(&self) -> Optimizer {
        if self.use_adam { Optimizer::adam_default(self.lr) } else { Optimizer::SGD(self.lr) }
    }
}

pub fn train_copy_task(cfg: TrainConfig) {
    let opt = cfg.make_optimizer();

    let mut model = Transformer::new_empty(cfg.d_model, cfg.vocab, cfg.vocab, cfg.vocab);

    let enc_cfg = EncoderConfig::new(
        cfg.n_heads, 1e-5, cfg.d_ff, Activation::ReLU, Loss::MSE, opt,
    );
    let dec_cfg = DecoderConfig::new(
        cfg.n_heads, 1e-5, cfg.d_ff, Activation::ReLU, Loss::MSE, opt,
    );
    model.set_encoder_configs((0..cfg.n_enc_layers).map(|_| enc_cfg).collect());
    model.set_decoder_configs((0..cfg.n_dec_layers).map(|_| dec_cfg).collect());

    let mut task = CopyTask::new(cfg.vocab, cfg.seq_len, cfg.seed);

    for step in 0..cfg.steps {
        let (src, tgt_in, labels) = task.sample();

        model.clear_grads();
        let logits = model.forward(Some(&src), Some(&tgt_in));
        let loss   = cross_entropy_loss(&labels, &logits);
        let delta  = cross_entropy_grad(&labels, &logits);
        model.backward(delta);
        model.update(&opt);

        if step % cfg.log_every == 0 || step + 1 == cfg.steps {
            println!("step {step:>5}  loss = {loss:.4}");
        }
    }
}

/// Diagnostic: train on a single fixed example for many steps.
/// If loss does not approach 0, there is a bug in backprop (or optimizer is too weak for the LR).
pub fn overfit_one_batch(cfg: TrainConfig) {
    let opt = cfg.make_optimizer();

    let mut model = Transformer::new_empty(cfg.d_model, cfg.vocab, cfg.vocab, cfg.vocab);

    let enc_cfg = EncoderConfig::new(
        cfg.n_heads, 1e-5, cfg.d_ff, Activation::ReLU, Loss::MSE, opt,
    );
    let dec_cfg = DecoderConfig::new(
        cfg.n_heads, 1e-5, cfg.d_ff, Activation::ReLU, Loss::MSE, opt,
    );
    model.set_encoder_configs((0..cfg.n_enc_layers).map(|_| enc_cfg).collect());
    model.set_decoder_configs((0..cfg.n_dec_layers).map(|_| dec_cfg).collect());

    // one fixed example, reused every step
    let mut task = CopyTask::new(cfg.vocab, cfg.seq_len, cfg.seed);
    let (src, tgt_in, labels) = task.sample();
    println!("overfitting on:  src={:?}  tgt_in={:?}  labels={:?}", src, tgt_in, labels);

    for step in 0..cfg.steps {
        model.clear_grads();
        let logits = model.forward(Some(&src), Some(&tgt_in));
        let loss   = cross_entropy_loss(&labels, &logits);
        let delta  = cross_entropy_grad(&labels, &logits);
        model.backward(delta);
        model.update(&opt);

        if step % cfg.log_every == 0 || step + 1 == cfg.steps {
            println!("step {step:>5}  loss = {loss:.4}");
        }
    }
}
