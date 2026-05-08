
mod activation;
pub use activation::Activation;

mod loss;
pub use loss::{Loss, cross_entropy_grad, cross_entropy_loss};

mod optimizer;
pub use optimizer::{AdamState1, AdamState2, Optimizer, Trainable};

mod xavier;
pub use xavier::xavier_init;

pub mod persist;

