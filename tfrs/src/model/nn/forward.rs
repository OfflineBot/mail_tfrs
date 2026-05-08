
use ndarray::Array2;
use crate::{model::nn::nn::LayerParams, utils::Activation};


impl LayerParams {

    #[allow(dead_code)]
    pub fn forward(&mut self, x: &Array2<f32>, activation: Activation) -> Array2<f32> {
        let z1 = x.dot(&self.weights_1) + &self.biases_1;
        let a1 = activation.activate(&z1);

        let z2 = a1.dot(&self.weights_2) + &self.biases_2;

        self.input = Some(x.clone());
        self.z1 = Some(z1);
        self.a1 = Some(a1);
        self.z2 = Some(z2.clone());

        z2
    }
}


