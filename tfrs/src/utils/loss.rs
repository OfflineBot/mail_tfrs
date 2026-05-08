
use ndarray::Array2;

#[derive(Clone, Copy)]
pub enum Loss {
    MSE,
    #[allow(unused)]
    Custom {
        loss: fn(&Array2<f32>, &Array2<f32>) -> f32,
        derivative: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    }
}


impl Loss {
    pub fn loss_item(&self, truth: &Array2<f32>, output: &Array2<f32>) -> f32 {
        match self {
            Self::MSE => mse_item(truth, output),
            Self::Custom { loss, .. } => loss(truth, output),
        }
    }

    pub fn deriv_loss(&self, truth: &Array2<f32>, output: &Array2<f32>) -> Array2<f32> {
        match self {
            Self::MSE => deriv_mse_item(truth, output),
            Self::Custom { derivative, .. } => derivative(truth, output),
        }
    }
}


fn mse_item(truth: &Array2<f32>, output: &Array2<f32>) -> f32 {
    (truth - output).powf(2.).mean().unwrap()
}
fn deriv_mse_item(truth: &Array2<f32>, output: &Array2<f32>) -> Array2<f32> { output - truth }


/// Mean cross-entropy loss across positions.
/// `targets` are class indices, `logits` is `(seq, vocab)`.
pub fn cross_entropy_loss(targets: &[usize], logits: &Array2<f32>) -> f32 {
    let n = logits.shape()[0];
    assert_eq!(targets.len(), n, "targets length must match logits rows");

    let mut total = 0.0_f32;
    for (i, &t) in targets.iter().enumerate() {
        let row    = logits.row(i);
        let max    = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let sumexp = row.iter().map(|&x| (x - max).exp()).sum::<f32>();
        let logsumexp = max + sumexp.ln();
        total += logsumexp - row[t];
    }
    total / n as f32
}

/// Gradient of `cross_entropy_loss` w.r.t. `logits`. Returns `(seq, vocab)`.
pub fn cross_entropy_grad(targets: &[usize], logits: &Array2<f32>) -> Array2<f32> {
    let n = logits.shape()[0];
    let v = logits.shape()[1];
    assert_eq!(targets.len(), n, "targets length must match logits rows");

    let mut grad = Array2::<f32>::zeros((n, v));
    for i in 0..n {
        let row    = logits.row(i);
        let max    = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for j in 0..v {
            grad[[i, j]] = exp[j] / sum;
        }
        grad[[i, targets[i]]] -= 1.0;
    }
    grad / n as f32
}

