
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub fn xavier_init(row: usize, col: usize) -> Array2<f32> {
    let limit = (6. / (row + col) as f32).sqrt();
    let dist = Uniform::new(-limit, limit).unwrap();
    Array2::random((row, col), dist)

}

