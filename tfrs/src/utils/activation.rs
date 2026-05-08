#![allow(dead_code)]
use ndarray::Array2;


#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    LeakyReLU(f32),
    Custom {
        activate: fn(&Array2<f32>) -> Array2<f32>,
        derivative: fn(&Array2<f32>) -> Array2<f32>,
    }
}

impl Activation {
    pub fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::ReLU => relu(x),
            Self::LeakyReLU(a) => leaky_relu(x, *a),
            Self::Custom { activate: a, .. } => a(x),
        }
    }

    pub fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::ReLU => deriv_relu(x),
            Self::LeakyReLU(a) => leaky_deriv_relu(x, *a),
            Self::Custom { derivative: d, .. } => d(x),
        }
    }
}


fn relu(x: &Array2<f32>) -> Array2<f32> { x.mapv(|f| if f < 0. { 0. } else { f }) }
fn deriv_relu(x: &Array2<f32>) -> Array2<f32> { x.mapv(|f| if f < 0. { 0. } else { 1. }) }

fn leaky_relu(x: &Array2<f32>, alpha: f32) -> Array2<f32> { x.mapv(|f| if f < 0. { f * alpha } else { f }) }
fn leaky_deriv_relu(x: &Array2<f32>, alpha: f32) -> Array2<f32> { x.mapv(|f| if f < 0. { alpha } else { 1. }) }

