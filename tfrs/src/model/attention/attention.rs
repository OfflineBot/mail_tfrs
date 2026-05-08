use core::f32;

use ndarray::{Array2, Array3, Axis};

use std::io::{Read, Result, Write};
use crate::utils::{AdamState2, Trainable, persist, xavier_init};


#[derive(Clone)]
pub struct Attention {
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    pub w_o: Array2<f32>,

    n_heads: usize,
    d_head: usize, // d_model / n_heads

    // ===== cache =====
    input_q: Option<Array2<f32>>,
    input_kv: Option<Array2<f32>>,


    w_q_grad: Option<Array2<f32>>,
    w_k_grad: Option<Array2<f32>>,
    w_v_grad: Option<Array2<f32>>,
    w_o_grad: Option<Array2<f32>>,

    w_q_state: AdamState2,
    w_k_state: AdamState2,
    w_v_state: AdamState2,
    w_o_state: AdamState2,

    q: Option<Array3<f32>>,
    k: Option<Array3<f32>>,
    v: Option<Array3<f32>>,
    weights: Option<Array3<f32>>,
    concat: Option<Array2<f32>>,
}

impl Attention {
    pub fn new(d_model: usize, n_heads: usize) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        Self {
            d_head: d_model / n_heads,
            n_heads,
            w_q: xavier_init(d_model, d_model),
            w_k: xavier_init(d_model, d_model),
            w_v: xavier_init(d_model, d_model),
            w_o: xavier_init(d_model, d_model),

            input_q: None,
            input_kv: None,
            w_q_grad: None,
            w_k_grad: None,
            w_v_grad: None,
            w_o_grad: None,
            w_q_state: AdamState2::default(),
            w_k_state: AdamState2::default(),
            w_v_state: AdamState2::default(),
            w_o_state: AdamState2::default(),
            q: None,
            k: None,
            v: None,
            weights: None,
            concat: None,
        }
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        persist::write_array2(w, &self.w_q)?;
        persist::write_array2(w, &self.w_k)?;
        persist::write_array2(w, &self.w_v)?;
        persist::write_array2(w, &self.w_o)?;
        Ok(())
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.w_q = persist::read_array2(r)?;
        self.w_k = persist::read_array2(r)?;
        self.w_v = persist::read_array2(r)?;
        self.w_o = persist::read_array2(r)?;
        Ok(())
    }

    pub fn forward(&mut self, x_q: &Array2<f32>, x_kv: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let scale = (self.d_head as f32).sqrt();
        let seq_q = x_q.shape()[0];
        let seq_kv = x_kv.shape()[0];

        let q_full = x_q.dot(&self.w_q);
        let k_full = x_kv.dot(&self.w_k);
        let v_full = x_kv.dot(&self.w_v);

        let q = q_full.to_shape((seq_q, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);
        let k = k_full.to_shape((seq_kv, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);
        let v = v_full.to_shape((seq_kv, self.n_heads, self.d_head))
            .unwrap()
            .permuted_axes([1, 0, 2]);

        let mut weights = Array3::<f32>::zeros((self.n_heads, seq_q, seq_kv));
        let mut head_outs = Array3::<f32>::zeros((self.n_heads, seq_q, self.d_head));

        for h in 0..self.n_heads {
            let qh = q.index_axis(Axis(0), h).to_owned();
            let kh = k.index_axis(Axis(0), h).to_owned();
            let vh = v.index_axis(Axis(0), h).to_owned();

            let mut s = qh.dot(&kh.t()) / scale;
            if let Some(m) = mask { s = s + m; }
            let w = Self::softmax(&s);
            let o = w.dot(&vh);

            weights.index_axis_mut(Axis(0), h).assign(&w);
            head_outs.index_axis_mut(Axis(0), h).assign(&o);
        }

        let concat = head_outs
            .permuted_axes([1, 0, 2])
            .to_shape((seq_q, self.n_heads * self.d_head)).unwrap()
            .to_owned();

        let out = concat.dot(&self.w_o);

        self.input_q = Some(x_q.clone());
        self.input_kv = Some(x_kv.clone());
        self.q = Some(q.to_owned());
        self.k = Some(k.to_owned());
        self.v = Some(v.to_owned());
        self.weights = Some(weights);
        self.concat = Some(concat);

        out
    }


    fn softmax(x: &Array2<f32>) -> Array2<f32> {
        let max = x.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)))
                   .insert_axis(Axis(1));
        let exp = (x - &max).mapv(f32::exp);
        let sum = exp.sum_axis(Axis(1)).insert_axis(Axis(1));
        exp / sum
    }


    pub fn backward(&mut self, d_out: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let scale = (self.d_head as f32).sqrt();

        let concat  = self.concat.as_ref().unwrap().clone();
        let weights = self.weights.as_ref().unwrap().clone();
        let q       = self.q.as_ref().unwrap().clone();
        let k       = self.k.as_ref().unwrap().clone();
        let v       = self.v.as_ref().unwrap().clone();
        let x_q     = self.input_q.as_ref().unwrap().clone();
        let x_kv    = self.input_kv.as_ref().unwrap().clone();

        let seq_q  = x_q.shape()[0];
        let seq_kv = x_kv.shape()[0];

        self.w_o_grad = Some(concat.t().dot(d_out));
        let d_concat  = d_out.dot(&self.w_o.t());

        let d_head_outs = d_concat
            .to_shape((seq_q, self.n_heads, self.d_head)).unwrap()
            .permuted_axes([1, 0, 2])
            .to_owned();

        let mut dq = Array3::<f32>::zeros((self.n_heads, seq_q,  self.d_head));
        let mut dk = Array3::<f32>::zeros((self.n_heads, seq_kv, self.d_head));
        let mut dv = Array3::<f32>::zeros((self.n_heads, seq_kv, self.d_head));

        for h in 0..self.n_heads {
            let w_h  = weights.index_axis(Axis(0), h).to_owned();
            let q_h  = q.index_axis(Axis(0), h).to_owned();
            let k_h  = k.index_axis(Axis(0), h).to_owned();
            let v_h  = v.index_axis(Axis(0), h).to_owned();
            let do_h = d_head_outs.index_axis(Axis(0), h).to_owned();

            let dv_h = w_h.t().dot(&do_h);
            let dw_h = do_h.dot(&v_h.t());

            let row_sum = (&dw_h * &w_h).sum_axis(Axis(1)).insert_axis(Axis(1));
            let mut ds_h = &w_h * &(&dw_h - &row_sum);
            ds_h /= scale;

            let dq_h = ds_h.dot(&k_h);
            let dk_h = ds_h.t().dot(&q_h);

            dq.index_axis_mut(Axis(0), h).assign(&dq_h);
            dk.index_axis_mut(Axis(0), h).assign(&dk_h);
            dv.index_axis_mut(Axis(0), h).assign(&dv_h);
        }

        let d_q_2d = dq.permuted_axes([1, 0, 2])
            .to_shape((seq_q, self.n_heads * self.d_head)).unwrap().to_owned();
        let d_k_2d = dk.permuted_axes([1, 0, 2])
            .to_shape((seq_kv, self.n_heads * self.d_head)).unwrap().to_owned();
        let d_v_2d = dv.permuted_axes([1, 0, 2])
            .to_shape((seq_kv, self.n_heads * self.d_head)).unwrap().to_owned();

        self.w_q_grad = Some(x_q.t().dot(&d_q_2d));
        self.w_k_grad = Some(x_kv.t().dot(&d_k_2d));
        self.w_v_grad = Some(x_kv.t().dot(&d_v_2d));

        let d_x_q  = d_q_2d.dot(&self.w_q.t());
        let d_x_kv = d_k_2d.dot(&self.w_k.t()) + d_v_2d.dot(&self.w_v.t());

        (d_x_q, d_x_kv)
    }

}

impl Trainable for Attention {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        opt.step_w(&mut self.w_q, self.w_q_grad.as_ref().unwrap(), &mut self.w_q_state);
        opt.step_w(&mut self.w_k, self.w_k_grad.as_ref().unwrap(), &mut self.w_k_state);
        opt.step_w(&mut self.w_v, self.w_v_grad.as_ref().unwrap(), &mut self.w_v_state);
        opt.step_w(&mut self.w_o, self.w_o_grad.as_ref().unwrap(), &mut self.w_o_state);
    }

    fn clear_grads(&mut self) {
        self.w_q_grad = None;
        self.w_k_grad = None;
        self.w_v_grad = None;
        self.w_o_grad = None;
        self.input_q = None;
        self.input_kv = None;
        self.weights = None;
        self.q = None;
        self.k = None;
        self.v = None;
        self.concat = None;
    }
}

