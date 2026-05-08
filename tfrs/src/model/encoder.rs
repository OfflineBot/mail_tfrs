
use std::io::{Read, Result, Write};
use ndarray::Array2;

use crate::{
    model::{
        attention::Attention,
        nn::{NeuralNetwork, NeuralNetworkConfig},
        norm::AddNorm
    },
    utils::{Activation, Loss, Optimizer, Trainable}};


#[derive(Clone, Copy)]
pub struct EncoderConfig {
    n_heads: usize,
    eps: f32,

    // ====== neural network =======
    /// dimension/size of hidden layer
    d_ff: usize,
    activation_ff: Activation,
    loss_ff: Loss,
    optim_ff: Optimizer,
}


impl EncoderConfig {
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
pub struct Encoder {

    config: EncoderConfig,

    self_attention: Attention,
    add_norm1: AddNorm,

    ff: NeuralNetwork,
    add_norm2: AddNorm,
}

impl Encoder {

    pub fn new(config: EncoderConfig, d_model: usize) -> Self {

        let nn_config = NeuralNetworkConfig::new(d_model, config.d_ff, d_model);

        Self {
            config,
            self_attention: Attention::new(d_model, config.n_heads),

            ff: NeuralNetwork::new(nn_config, config.activation_ff, config.loss_ff, config.optim_ff),

            add_norm1: AddNorm::new(d_model, config.eps),
            add_norm2: AddNorm::new(d_model, config.eps),
        }
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        self.self_attention.save_params(w)?;
        self.add_norm1.save_params(w)?;
        self.ff.save_params(w)?;
        self.add_norm2.save_params(w)
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.self_attention.load_params(r)?;
        self.add_norm1.load_params(r)?;
        self.ff.load_params(r)?;
        self.add_norm2.load_params(r)
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {

        let attn = self.self_attention.forward(x, x, None);
        let x1   = self.add_norm1.forward(x, &attn);

        let ff   = self.ff.forward(&x1);
        let x2   = self.add_norm2.forward(&x1, &ff);

        x2
    }

    pub fn backward(&mut self, d_out: Array2<f32>) -> Array2<f32> {
        let (d_x1_a, d_ff)  = self.add_norm2.backward(&d_out);
        let d_x1_b          = self.ff.backward_delta(d_ff);
        let d_x1            = d_x1_a + d_x1_b;

        let (d_x_a, d_attn) = self.add_norm1.backward(&d_x1);
        let (d_x_q, d_x_kv) = self.self_attention.backward(&d_attn);
        d_x_a + d_x_q + d_x_kv
    }
}

impl Trainable for Encoder {
    fn update(&mut self, opt: &Optimizer) {
        self.self_attention.update(opt);
        self.add_norm1.update(opt);
        self.ff.update(opt);
        self.add_norm2.update(opt);
    }

    fn clear_grads(&mut self) {
        self.self_attention.clear_grads();
        self.add_norm1.clear_grads();
        self.ff.clear_grads();
        self.add_norm2.clear_grads();
    }
}


pub struct EncoderBlock {
    layers: Vec<Encoder>,
}

impl EncoderBlock {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn insert(&mut self, encoder: Encoder) {
        self.layers.push(encoder);
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let mut input = x.clone();
        for e in self.layers.iter_mut() {
            input = e.forward(&input);
        }
        input
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        for e in &self.layers { e.save_params(w)?; }
        Ok(())
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        for e in self.layers.iter_mut() { e.load_params(r)?; }
        Ok(())
    }

    pub fn n_layers(&self) -> usize { self.layers.len() }

    pub fn backward(&mut self, d_out: Array2<f32>) -> Array2<f32> {
        let mut delta = d_out;
        for e in self.layers.iter_mut().rev() {
            delta = e.backward(delta);
        }
        delta
    }
}


impl Trainable for EncoderBlock {
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

