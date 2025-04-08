use dfdx::{
    data::*,
    prelude::*,
};
use peroxide::fuga::*;

fn main() {
    println!("Hello, world!");
}

struct SimpleDataset {
    x: Vec<f32>,
    y: Vec<f32>,
}

impl SimpleDataset {
    fn new(rng: &mut Rng) -> Self {
        let x = linspace(0.0, 3.0, 100);
        let y0 = 2.0;
        let v0 = 5.0;
        let g = -9.81;
        let y_true = x.fmap(|t| y0 + v0 * t + 0.5 * g * t.powi(2));
        let normal = Normal(0.0, 1.0);
        let eps = normal.sample_with_rng(&mut rng, 100);
        let y_noisy = y_true.add_v(&eps);

        let x = x.to_vec();
        let y = y_noisy.to_vec();
        Self { x, y }
    }
}
