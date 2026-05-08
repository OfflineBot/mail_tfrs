#![allow(dead_code)]

use std::io::{Read, Result, Write};
use ndarray::{Array1, Array2};
use crate::utils::{Activation, AdamState1, AdamState2, Loss, Optimizer, Trainable, persist};


/// per definition *1 hidden layer* with *ReLU* activation
#[derive(Clone)]
pub struct NeuralNetworkConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl NeuralNetworkConfig {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            input_size,
            hidden_size,
            output_size,
        }
    }
}


#[derive(Clone)]
pub struct LayerParams {

    pub weights_1: Array2<f32>,
    pub biases_1: Array1<f32>,

    pub weights_2: Array2<f32>,
    pub biases_2: Array1<f32>,

    // ==== CACHE ====
    // ---- Foward ---
    pub(super) input: Option<Array2<f32>>,
    pub(super) z1: Option<Array2<f32>>,
    pub(super) a1: Option<Array2<f32>>,
    pub(super) z2: Option<Array2<f32>>,

    // --- Backward ---
    pub weight_grad_1: Option<Array2<f32>>,
    pub weight_grad_2: Option<Array2<f32>>,

    pub bias_grad_1: Option<Array1<f32>>,
    pub bias_grad_2: Option<Array1<f32>>,

    pub weight_state_1: AdamState2,
    pub weight_state_2: AdamState2,
    pub bias_state_1:   AdamState1,
    pub bias_state_2:   AdamState1,
}

impl LayerParams {
    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        persist::write_array2(w, &self.weights_1)?;
        persist::write_array1(w, &self.biases_1)?;
        persist::write_array2(w, &self.weights_2)?;
        persist::write_array1(w, &self.biases_2)
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.weights_1 = persist::read_array2(r)?;
        self.biases_1  = persist::read_array1(r)?;
        self.weights_2 = persist::read_array2(r)?;
        self.biases_2  = persist::read_array1(r)?;
        Ok(())
    }
}

impl NeuralNetwork {
    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        self.layer.save_params(w)
    }
    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.layer.load_params(r)
    }
}

impl Trainable for LayerParams {
    fn update(&mut self, opt: &Optimizer) {
        opt.step_w(&mut self.weights_1, self.weight_grad_1.as_ref().unwrap(), &mut self.weight_state_1);
        opt.step_w(&mut self.weights_2, self.weight_grad_2.as_ref().unwrap(), &mut self.weight_state_2);
        opt.step_b(&mut self.biases_1,  self.bias_grad_1.as_ref().unwrap(),   &mut self.bias_state_1);
        opt.step_b(&mut self.biases_2,  self.bias_grad_2.as_ref().unwrap(),   &mut self.bias_state_2);
    }

    fn clear_grads(&mut self) {
        self.input = None;
        self.z1 = None;
        self.a1 = None;
        self.z2 = None;

        self.weight_grad_1 = None;
        self.weight_grad_2 = None;

        self.bias_grad_1 = None;
        self.bias_grad_2 = None;
    }
}


/// per definition *1 hidden layer* with *ReLU* activation
#[derive(Clone)]
pub struct NeuralNetwork {
    pub config: NeuralNetworkConfig,
    pub activation: Activation,
    pub loss: Loss,
    pub optim: Optimizer,
    pub layer: LayerParams,
}


impl NeuralNetwork {
    pub fn new(config: NeuralNetworkConfig, activation: Activation, loss: Loss, optimizer: Optimizer) -> Self {

        let layer = LayerParams::init(config.input_size, config.hidden_size, config.output_size);

        Self {
            config,
            activation,
            loss,
            layer,
            optim: optimizer,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.layer.forward(input, self.activation)
    }

    pub fn backward(&mut self, truth: &Array2<f32>) {
        let Some(output) = self.layer.z2.clone() else {
            panic!("there is not z2 output in the fully connected");
        };
        let error = self.loss.deriv_loss(truth, &output);
        self.layer.backward(error, self.activation);
    }

    /// Delta-passthrough backward used inside the transformer.
    /// Takes upstream dL/dOutput, returns dL/dInput.
    pub fn backward_delta(&mut self, delta: Array2<f32>) -> Array2<f32> {
        self.layer.backward(delta, self.activation)
    }

    pub fn item_loss(&self, truth: &Array2<f32>) -> f32 {
        if let Some(prediction) = &self.layer.z2 {
            self.loss.loss_item(truth, &prediction)
        } else {
            println!("no prediction found!");
            0.
        }
    }
}

impl Trainable for NeuralNetwork {
    fn update(&mut self, _opt: &Optimizer) {
        self.layer.update(&self.optim);
    }

    fn clear_grads(&mut self) {
        self.layer.clear_grads();
    }
}

