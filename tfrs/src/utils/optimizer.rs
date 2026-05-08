use ndarray::{Array1, Array2};

pub trait Trainable {
    fn update(&mut self, opt: &Optimizer);
    fn clear_grads(&mut self);
}


#[derive(Clone, Copy)]
pub enum Optimizer {
    /// SGD with learning rate.
    SGD(f32),
    /// Adam with `(lr, beta1, beta2, eps)`. Standard defaults: `(1e-3, 0.9, 0.999, 1e-8)`.
    Adam(f32, f32, f32, f32),
}

impl Optimizer {
    pub fn adam_default(lr: f32) -> Self { Self::Adam(lr, 0.9, 0.999, 1e-8) }

    pub fn step_w(&self, w: &mut Array2<f32>, g: &Array2<f32>, state: &mut AdamState2) {
        match *self {
            Self::SGD(lr) => *w -= &(g * lr),
            Self::Adam(lr, b1, b2, eps) => {
                state.t += 1;
                let m = state.m.get_or_insert_with(|| Array2::zeros(g.raw_dim()));
                let v = state.v.get_or_insert_with(|| Array2::zeros(g.raw_dim()));
                *m = &*m * b1 + &(g * (1.0 - b1));
                *v = &*v * b2 + &(g * g * (1.0 - b2));
                let bc1 = 1.0 - b1.powi(state.t as i32);
                let bc2 = 1.0 - b2.powi(state.t as i32);
                let m_hat = &*m / bc1;
                let v_hat = &*v / bc2;
                *w -= &(&m_hat * lr / (v_hat.mapv(f32::sqrt) + eps));
            }
        }
    }

    pub fn step_b(&self, b: &mut Array1<f32>, g: &Array1<f32>, state: &mut AdamState1) {
        match *self {
            Self::SGD(lr) => *b -= &(g * lr),
            Self::Adam(lr, b1, b2, eps) => {
                state.t += 1;
                let m = state.m.get_or_insert_with(|| Array1::zeros(g.raw_dim()));
                let v = state.v.get_or_insert_with(|| Array1::zeros(g.raw_dim()));
                *m = &*m * b1 + &(g * (1.0 - b1));
                *v = &*v * b2 + &(g * g * (1.0 - b2));
                let bc1 = 1.0 - b1.powi(state.t as i32);
                let bc2 = 1.0 - b2.powi(state.t as i32);
                let m_hat = &*m / bc1;
                let v_hat = &*v / bc2;
                *b -= &(&m_hat * lr / (v_hat.mapv(f32::sqrt) + eps));
            }
        }
    }
}


#[derive(Clone, Default)]
pub struct AdamState2 {
    pub m: Option<Array2<f32>>,
    pub v: Option<Array2<f32>>,
    pub t: usize,
}

#[derive(Clone, Default)]
pub struct AdamState1 {
    pub m: Option<Array1<f32>>,
    pub v: Option<Array1<f32>>,
    pub t: usize,
}
