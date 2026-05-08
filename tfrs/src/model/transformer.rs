#![allow(dead_code)]

use std::fs::File;
use std::io::{BufReader, BufWriter, Result};
use std::path::Path;

use ndarray::{Array1, Array2};

use crate::{
    model::{
        decoder::{Decoder, DecoderBlock, DecoderConfig}, embedding::{Embeddings, positional_encoding}, encoder::{Encoder, EncoderBlock, EncoderConfig}
    },
    utils::{AdamState1, AdamState2, Trainable, persist, xavier_init}
};

#[derive(Debug, Clone)]
pub struct TransformerHeader {
    pub d_model: usize,
    pub src_vocab: usize,
    pub tgt_vocab: usize,
    pub num_classes: usize,
    pub n_enc_layers: usize,
    pub n_dec_layers: usize,
    pub categories: Vec<String>,
}

pub struct Transformer {
    d_model: usize,

    src_embed: Embeddings,
    tgt_embed: Embeddings,

    encoder: EncoderBlock,
    decoder: DecoderBlock,

    w_out: Array2<f32>,
    b_out: Array1<f32>,

    w_out_grad: Option<Array2<f32>>,
    b_out_grad: Option<Array1<f32>>,

    w_out_state: AdamState2,
    b_out_state: AdamState1,

    // ===== forward cache (for backward) =====
    h_cache:        Option<Array2<f32>>,
    used_encoder:   bool,
    used_decoder:   bool,
    decoder_had_memory: bool,
}

impl Transformer {
    // ============== PROPERTY FUNCTIONS ================
    pub fn new_empty(d_model: usize, src_vocab: usize, tgt_vocab: usize, num_classes: usize) -> Self {
        Self { 
            encoder: EncoderBlock::new(),
            decoder: DecoderBlock::new(),
            src_embed: Embeddings::new(src_vocab, d_model),
            tgt_embed: Embeddings::new(tgt_vocab, d_model),
            d_model,
            w_out: xavier_init(d_model, num_classes),
            b_out: Array1::zeros(num_classes),
            w_out_grad: None,
            b_out_grad: None,
            w_out_state: AdamState2::default(),
            b_out_state: AdamState1::default(),
            h_cache: None,
            used_encoder: false,
            used_decoder: false,
            decoder_had_memory: false,
        }
    }

    pub fn set_encoder(&mut self, encoder: EncoderBlock) { self.encoder = encoder; }
    pub fn set_decoder(&mut self, decoder: DecoderBlock) { self.decoder = decoder; }

    pub fn unset_encoder(&mut self) { self.encoder = EncoderBlock::new(); }
    pub fn unset_decoder(&mut self) { self.decoder = DecoderBlock::new(); }

    pub fn set_encoder_configs(&mut self, encoder: Vec<EncoderConfig>) {
        for e in encoder {
            self.encoder.insert(Encoder::new(e, self.d_model));
        }
    }

    pub fn set_decoder_configs(&mut self, decoder: Vec<DecoderConfig>) {
        for d in decoder {
            self.decoder.insert(Decoder::new(d, self.d_model));
        }
    }

    // ============== PERSISTENCE ======================
    /// Save magic header + architecture metadata + label names + all learnable
    /// weights to `path`. `categories` is the ordered list of label names (one
    /// per output class) so prediction can map logits back to human-readable
    /// tags without needing the original dataset.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P, categories: &[String]) -> Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);

        persist::write_magic(&mut w)?;
        persist::write_u32(&mut w, self.d_model as u32)?;
        persist::write_u32(&mut w, self.src_embed.table.shape()[0] as u32)?;
        persist::write_u32(&mut w, self.tgt_embed.table.shape()[0] as u32)?;
        persist::write_u32(&mut w, self.b_out.len() as u32)?;
        persist::write_u32(&mut w, self.encoder.n_layers() as u32)?;
        persist::write_u32(&mut w, self.decoder.n_layers() as u32)?;

        // category names — must match num_classes
        assert_eq!(
            categories.len(),
            self.b_out.len(),
            "category count {} must match num_classes {}",
            categories.len(),
            self.b_out.len()
        );
        persist::write_strings(&mut w, categories)?;

        self.src_embed.save_params(&mut w)?;
        self.tgt_embed.save_params(&mut w)?;
        self.encoder.save_params(&mut w)?;
        self.decoder.save_params(&mut w)?;

        persist::write_array2(&mut w, &self.w_out)?;
        persist::write_array1(&mut w, &self.b_out)?;
        Ok(())
    }

    /// Load weights from `path` into `self`, returning the category names.
    /// The model must already have the matching architecture (build it the
    /// same way you did before saving, then call this). Architectural ints
    /// in the header are checked.
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<String>> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        persist::read_magic(&mut r)?;
        let d_model     = persist::read_u32(&mut r)? as usize;
        let src_vocab   = persist::read_u32(&mut r)? as usize;
        let tgt_vocab   = persist::read_u32(&mut r)? as usize;
        let num_classes = persist::read_u32(&mut r)? as usize;
        let n_enc       = persist::read_u32(&mut r)? as usize;
        let n_dec       = persist::read_u32(&mut r)? as usize;

        let bad = |msg: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string());
        if d_model     != self.d_model                      { return Err(bad("d_model mismatch")); }
        if src_vocab   != self.src_embed.table.shape()[0]   { return Err(bad("src_vocab mismatch")); }
        if tgt_vocab   != self.tgt_embed.table.shape()[0]   { return Err(bad("tgt_vocab mismatch")); }
        if num_classes != self.b_out.len()                  { return Err(bad("num_classes mismatch")); }
        if n_enc       != self.encoder.n_layers()           { return Err(bad("encoder layer count mismatch")); }
        if n_dec       != self.decoder.n_layers()           { return Err(bad("decoder layer count mismatch")); }

        let categories = persist::read_strings(&mut r)?;
        if categories.len() != num_classes {
            return Err(bad("category count mismatch"));
        }

        self.src_embed.load_params(&mut r)?;
        self.tgt_embed.load_params(&mut r)?;
        self.encoder.load_params(&mut r)?;
        self.decoder.load_params(&mut r)?;

        self.w_out = persist::read_array2(&mut r)?;
        self.b_out = persist::read_array1(&mut r)?;
        Ok(categories)
    }

    /// Read just the header of a saved model — useful when you need to know
    /// the architecture / category count before constructing an empty model
    /// to load into.
    pub fn read_header<P: AsRef<Path>>(path: P) -> Result<TransformerHeader> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        persist::read_magic(&mut r)?;
        let d_model     = persist::read_u32(&mut r)? as usize;
        let src_vocab   = persist::read_u32(&mut r)? as usize;
        let tgt_vocab   = persist::read_u32(&mut r)? as usize;
        let num_classes = persist::read_u32(&mut r)? as usize;
        let n_enc       = persist::read_u32(&mut r)? as usize;
        let n_dec       = persist::read_u32(&mut r)? as usize;
        let categories  = persist::read_strings(&mut r)?;

        Ok(TransformerHeader {
            d_model,
            src_vocab,
            tgt_vocab,
            num_classes,
            n_enc_layers: n_enc,
            n_dec_layers: n_dec,
            categories,
        })
    }

    // ============== TRAINING =========================
    pub fn forward(&mut self, src_ids: Option<&[usize]>, tgt_ids: Option<&[usize]>) -> Array2<f32> {

        let memory = src_ids.map(|ids| {
            let s = self.src_embed.forward(ids) + positional_encoding(ids.len(), self.d_model);
            self.encoder.forward(&s)
        });

        let (h, used_encoder, used_decoder, decoder_had_memory) = match (memory, tgt_ids) {
            (Some(m), None)  => (m, true, false, false),
            (mem, Some(ids)) => {
                let had_mem = mem.is_some();
                let t       = self.tgt_embed.forward(ids) + positional_encoding(ids.len(), self.d_model);
                let mask    = causal_mask(ids.len());
                let out     = self.decoder.forward(&t, mem.as_ref(), Some(&mask));
                (out, had_mem, true, had_mem)
            },
            (None, None) => panic!("need atleast src or target. Got nothing")
        };

        let logits = h.dot(&self.w_out) + &self.b_out;

        self.h_cache            = Some(h);
        self.used_encoder       = used_encoder;
        self.used_decoder       = used_decoder;
        self.decoder_had_memory = decoder_had_memory;

        logits
    }

    pub fn backward(&mut self, delta: Array2<f32>) {
        // logits = h @ w_out + b_out
        let h = self.h_cache.as_ref().expect("forward must run before backward").clone();

        self.w_out_grad = Some(h.t().dot(&delta));
        self.b_out_grad = Some(delta.sum_axis(ndarray::Axis(0)));
        let d_h = delta.dot(&self.w_out.t());

        // dispatch mirrors forward's match
        let d_memory: Option<Array2<f32>> = if self.used_decoder {
            let (d_t, d_mem) = self.decoder.backward(d_h, self.decoder_had_memory);
            // PE has no params -> delta passes through unchanged
            self.tgt_embed.backward(&d_t);
            d_mem
        } else {
            // encoder-only path: d_h *is* d_memory
            Some(d_h)
        };

        if self.used_encoder {
            let d_mem = d_memory.expect("encoder ran but no memory delta");
            let d_s   = self.encoder.backward(d_mem);
            self.src_embed.backward(&d_s);
        }
    }
}


impl Trainable for Transformer {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        self.encoder.update(opt);
        self.decoder.update(opt);

        self.src_embed.update(opt);
        self.tgt_embed.update(opt);

        opt.step_w(&mut self.w_out, self.w_out_grad.as_ref().unwrap(), &mut self.w_out_state);
        opt.step_b(&mut self.b_out, self.b_out_grad.as_ref().unwrap(), &mut self.b_out_state);
    }

    fn clear_grads(&mut self) {
        self.encoder.clear_grads();
        self.decoder.clear_grads();

        self.src_embed.clear_grads();
        self.tgt_embed.clear_grads();

        self.w_out_grad = None;
        self.b_out_grad = None;

        self.h_cache            = None;
        self.used_encoder       = false;
        self.used_decoder       = false;
        self.decoder_had_memory = false;
    }
}


pub fn causal_mask(seq: usize) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((seq, seq));
    for i in 0..seq {
        for j in (i + 1)..seq {
            m[[i, j]] = f32::NEG_INFINITY;
        }
    }
    m
}
