use ndarray::Array1;
use crate::{
    model::nn::nn::LayerParams,
    utils::{AdamState1, AdamState2, xavier_init},
};

impl LayerParams {

    #[allow(dead_code)]
    pub fn init(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            weights_1: xavier_init(input_size, hidden_size),
            biases_1: Array1::zeros(hidden_size),

            weights_2: xavier_init(hidden_size, output_size),
            biases_2: Array1::zeros(output_size),

            input: None,
            a1: None,
            z1: None,
            z2: None,

            weight_grad_1: None,
            weight_grad_2: None,

            bias_grad_1: None,
            bias_grad_2: None,

            weight_state_1: AdamState2::default(),
            weight_state_2: AdamState2::default(),
            bias_state_1:   AdamState1::default(),
            bias_state_2:   AdamState1::default(),
        }
    }
}
