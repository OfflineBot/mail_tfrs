
use std::io::{Read, Result, Write};
use ndarray::Array2;

use crate::{model::{attention::Attention, nn::{NeuralNetwork, NeuralNetworkConfig}, norm::AddNorm}, utils::{Activation, Loss, Optimizer, Trainable}};


#[derive(Clone, Copy)]
pub struct DecoderConfig {
    n_heads: usize,
    eps: f32,

    // ====== neural network =======
    /// dimension/size of hidden layer
    d_ff: usize,
    activation_ff: Activation,
    loss_ff: Loss,
    optim_ff: Optimizer,
}


impl DecoderConfig {
    #[allow(unused)]
    pub fn new(
        n_heads: usize,
        eps: f32,
        d_ff: usize,
        activation_ff: Activation,
        loss_ff: Loss,
        optim_ff: Optimizer,
    ) -> Self {
        Self {
            n_heads,
            eps,
            d_ff, 
            activation_ff,
            loss_ff, 
            optim_ff
        }
    }
}


#[derive(Clone)]
#[allow(dead_code)]
pub struct Decoder {

    config: DecoderConfig,

    self_attention: Attention,
    add_norm1: AddNorm,

    cross_attention: Attention,
    add_norm2: AddNorm,

    ff: NeuralNetwork,
    add_norm3: AddNorm,

}

impl Decoder {

    pub fn new(config: DecoderConfig, d_model: usize) -> Self {

        let nn_config = NeuralNetworkConfig::new(d_model, config.d_ff, d_model);

        Self {
            config,

            self_attention: Attention::new(d_model, config.n_heads),
            cross_attention: Attention::new(d_model, config.n_heads),

            ff: NeuralNetwork::new(nn_config, config.activation_ff, config.loss_ff, config.optim_ff),

            add_norm1: AddNorm::new(d_model, config.eps),
            add_norm2: AddNorm::new(d_model, config.eps),
            add_norm3: AddNorm::new(d_model, config.eps),
        }
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        self.self_attention.save_params(w)?;
        self.add_norm1.save_params(w)?;
        self.cross_attention.save_params(w)?;
        self.add_norm2.save_params(w)?;
        self.ff.save_params(w)?;
        self.add_norm3.save_params(w)
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.self_attention.load_params(r)?;
        self.add_norm1.load_params(r)?;
        self.cross_attention.load_params(r)?;
        self.add_norm2.load_params(r)?;
        self.ff.load_params(r)?;
        self.add_norm3.load_params(r)
    }

    pub fn forward(
        &mut self,
        x: &Array2<f32>,
        memory: Option<&Array2<f32>>,
        causal_mask: Option<&Array2<f32>>
    ) -> Array2<f32> {
        let sa = self.self_attention.forward(x, x, causal_mask);
        let x1 = self.add_norm1.forward(x, &sa);
        let x2 = match memory {
            Some(mem) => {
                let ca = self.cross_attention.forward(&x1, mem, None);
                self.add_norm2.forward(&x1, &ca)
            },
            None => x1,
        };
        let ff = self.ff.forward(&x2);
        let x3 = self.add_norm3.forward(&x2, &ff);
        x3
    }

    pub fn backward(&mut self, d_out: Array2<f32>, had_memory: bool) -> (Array2<f32>, Option<Array2<f32>>) {
        let (d_x2_a, d_ff)   = self.add_norm3.backward(&d_out);
        let d_x2_b           = self.ff.backward_delta(d_ff);
        let d_x2             = d_x2_a + d_x2_b;

        let (d_x1, d_memory) = if had_memory {
            let (d_x1_a, d_ca)        = self.add_norm2.backward(&d_x2);
            let (d_q, d_kv)           = self.cross_attention.backward(&d_ca);
            (d_x1_a + d_q, Some(d_kv))
        } else {
            (d_x2, None)
        };

        let (d_x_a, d_sa)    = self.add_norm1.backward(&d_x1);
        let (d_x_q, d_x_kv)  = self.self_attention.backward(&d_sa);
        (d_x_a + d_x_q + d_x_kv, d_memory)
    }
}


impl Trainable for Decoder {
    fn update(&mut self, opt: &Optimizer) {
        self.self_attention.update(opt);
        self.add_norm1.update(opt);
        self.cross_attention.update(opt);
        self.add_norm2.update(opt);
        self.ff.update(opt);
        self.add_norm3.update(opt);
    }

    fn clear_grads(&mut self) {
        self.self_attention.clear_grads();
        self.add_norm1.clear_grads();
        self.cross_attention.clear_grads();
        self.add_norm2.clear_grads();
        self.ff.clear_grads();
        self.add_norm3.clear_grads();
    }
}


pub struct DecoderBlock {
    layers: Vec<Decoder>,
}

impl DecoderBlock {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn insert(&mut self, encoder: Decoder) {
        self.layers.push(encoder);
    }

    pub fn forward(&mut self, x: &Array2<f32>, memory: Option<&Array2<f32>>, causal_mask: Option<&Array2<f32>>) -> Array2<f32> {
        let mut input = x.clone();

        for d in self.layers.iter_mut() {
            input = d.forward(&input, memory, causal_mask);
        }

        input
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        for d in &self.layers { d.save_params(w)?; }
        Ok(())
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        for d in self.layers.iter_mut() { d.load_params(r)?; }
        Ok(())
    }

    pub fn n_layers(&self) -> usize { self.layers.len() }

    /// Returns (d_x, d_memory_total). d_memory is summed across layers (memory was shared).
    pub fn backward(&mut self, d_out: Array2<f32>, had_memory: bool) -> (Array2<f32>, Option<Array2<f32>>) {
        let mut delta      = d_out;
        let mut d_memory: Option<Array2<f32>> = None;

        for d in self.layers.iter_mut().rev() {
            let (d_x, d_mem) = d.backward(delta, had_memory);
            delta = d_x;
            if let Some(dm) = d_mem {
                d_memory = Some(match d_memory.take() {
                    Some(acc) => acc + dm,
                    None      => dm,
                });
            }
        }

        (delta, d_memory)
    }
}


impl Trainable for DecoderBlock {
    fn update(&mut self, opt: &Optimizer) {
        for d in self.layers.iter_mut() {
            d.update(opt);
        }
    }

    fn clear_grads(&mut self) {
        for d in self.layers.iter_mut() {
            d.clear_grads();
        }
    }
}

