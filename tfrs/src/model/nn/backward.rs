
use ndarray::{Array1, Array2, Axis};
use crate::{model::nn::nn::LayerParams, utils::Activation};


impl LayerParams {

    #[allow(dead_code)]
    pub fn backward(&mut self, mut delta: Array2<f32>, activation: Activation) -> Array2<f32> {

        let delta2 = delta;
        let a1 = self.a1.clone().unwrap();
        let grad_weight2 = a1.t().dot(&delta2);
        let grad_bias2: Array1<f32> = delta2.sum_axis(Axis(0));
        delta = delta2.dot(&self.weights_2.t());

        let z1 = self.z1.clone().unwrap();
        let delta1 = delta * &activation.derivative(&z1);
        let input = self.input.clone().unwrap();
        let grad_weight1 = input.t().dot(&delta1);
        let grad_bias1: Array1<f32> = delta1.sum_axis(Axis(0));
        let delta_out = delta1.dot(&self.weights_1.t());

        self.weight_grad_2 = Some(grad_weight2);
        self.bias_grad_2 = Some(grad_bias2);
        self.weight_grad_1 = Some(grad_weight1);
        self.bias_grad_1 = Some(grad_bias1);

        delta_out
    }
}

