use std::io::{Read, Result, Write};
use ndarray::{Array1, Array2, Axis};

use crate::utils::{AdamState1, Trainable, persist};

#[derive(Clone)]
#[allow(dead_code)]
pub struct AddNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,


    gamma_grad: Option<Array1<f32>>,
    beta_grad: Option<Array1<f32>>,
    input: Option<Array2<f32>>,
    x_norm: Option<Array2<f32>>,
    std: Option<Array2<f32>>,

    gamma_state: AdamState1,
    beta_state:  AdamState1,
}

impl AddNorm {

    pub fn new(d_model: usize, eps: f32) -> Self {
        Self {
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
            eps,
            gamma_grad: None,
            beta_grad: None,
            input: None,
            x_norm: None,
            std: None,
            gamma_state: AdamState1::default(),
            beta_state:  AdamState1::default(),
        }
    }

    pub fn save_params<W: Write>(&self, w: &mut W) -> Result<()> {
        persist::write_array1(w, &self.gamma)?;
        persist::write_array1(w, &self.beta)
    }

    pub fn load_params<R: Read>(&mut self, r: &mut R) -> Result<()> {
        self.gamma = persist::read_array1(r)?;
        self.beta  = persist::read_array1(r)?;
        Ok(())
    }

    pub fn backward(&mut self, d_out: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let x_norm = self.x_norm.as_ref().unwrap().clone();
        let std    = self.std.as_ref().unwrap().clone();
        let d      = x_norm.shape()[1] as f32;

        self.beta_grad  = Some(d_out.sum_axis(Axis(0)));
        self.gamma_grad = Some((d_out * &x_norm).sum_axis(Axis(0)));

        let dx_norm = d_out * &self.gamma;

        let mean_dx       = dx_norm.sum_axis(Axis(1)).insert_axis(Axis(1)) / d;
        let mean_dx_xhat  = (&dx_norm * &x_norm).sum_axis(Axis(1)).insert_axis(Axis(1)) / d;
        let d_summed      = (&dx_norm - &mean_dx - &x_norm * &mean_dx_xhat) / &std;

        (d_summed.clone(), d_summed)
    }

    pub fn forward(&mut self, x: &Array2<f32>, sublayer_output: &Array2<f32>) -> Array2<f32> {
        let summed  = x + sublayer_output;
        let mean    = summed.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let var     = summed.var_axis(Axis(1), 0.).insert_axis(Axis(1));
        let std     = (var + self.eps).mapv(f32::sqrt);
        let x_norm  = (&summed - &mean) / &std;
        let out     = &x_norm * &self.gamma + &self.beta;

        self.input  = Some(summed);
        self.x_norm = Some(x_norm);
        self.std    = Some(std);

        out
    }
}


impl Trainable for AddNorm {
    fn update(&mut self, opt: &crate::utils::Optimizer) {
        opt.step_b(&mut self.gamma, self.gamma_grad.as_ref().unwrap(), &mut self.gamma_state);
        opt.step_b(&mut self.beta,  self.beta_grad.as_ref().unwrap(),  &mut self.beta_state);
    }

    fn clear_grads(&mut self) {
        self.gamma_grad = None;
        self.beta_grad  = None;
        self.input      = None;
        self.x_norm     = None;
        self.std        = None;
    }
}

