#![allow(dead_code)]

use ndarray::Array2;

use std::io::{Read, Result, Write};
use crate::utils::{AdamState2, Trainable, persist, xavier_init};


pub struct Embeddings {
    pub table: Array2<f32>,
    vocab_size: usize,
    d_model: usize,

    // ===== cache =====
    last_ids: Option<Vec<usize>>,

    // ===== gradients =====
    pub table_grad: Option<Array2<f32>>,

    pub table_state: AdamState2,
}


impl Embeddings {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self {
            table: xavier_init(vocab_size, d_model),
            vocab_size,
            d_model,
            last_ids: None,
            table_grad: None,
            table_state: AdamState2::default(),
        }
    }

    pub fn backward(&mut self, d_out: &Array2<f32>) {
        let scale = (self.d_model as f32).sqrt();
        let ids   = self.last_ids.as_ref().expect("forward must run before backward").clone();

        let grad = self.table_grad.get_or_insert_with(
            || Array2::<f32>::zeros((self.vocab_size, self.d_model))
        );

        for (i, &id) in ids.iter().enumerate() {
            let row = &d_out.row(i) * scale;
            let mut dst = grad.row_mut(id);
            dst += &row;
        }
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        persist::write_array2(w, &self.table)
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        let t = persist::read_array2(r)?;
        assert_eq!(t.shape(), self.table.shape(), "embedding shape mismatch");
        self.table = t;
        Ok(())
    }

    pub fn forward(&mut self, ids: &[usize]) -> Array2<f32> {
        let scale = (self.d_model as f32).sqrt();
        let mut out = Array2::<f32>::zeros((ids.len(), self.d_model));
        for (i, &id) in ids.iter().enumerate() {
            assert!(id < self.vocab_size, "token id {} out of vocab range {}", id, self.vocab_size);
            out.row_mut(i).assign(&self.table.row(id));
        }
        self.last_ids = Some(ids.to_vec());
        out * scale
    }
}


impl Trainable for Embeddings {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        // Skip when this embedding wasn't used this step (e.g. tgt_embed in
        // encoder-only training). The corresponding weight stays untouched.
        if let Some(g) = self.table_grad.as_ref() {
            opt.step_w(&mut self.table, g, &mut self.table_state);
        }
    }

    fn clear_grads(&mut self) {
        self.table_grad = None;
        self.last_ids = None;
    }
}


pub fn positional_encoding(seq: usize, d_model: usize) -> Array2<f32> {
    let mut pe = Array2::<f32>::zeros((seq, d_model));
    for pos in 0..seq {
        for i in 0..d_model / 2 {
            let denom = 10000f32.powf(2.0 * i as f32 / d_model as f32);
            pe[[pos, 2 * i]]     = (pos as f32 / denom).sin();
            pe[[pos, 2 * i + 1]] = (pos as f32 / denom).cos();
        }
    }
    pe
}
