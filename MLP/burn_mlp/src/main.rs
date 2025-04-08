use burn::{
    nn::{
        Linear, LinearConfig, Relu,
    },
    data::{
        dataset::{Dataset, InMemDataset},
        dataloader::batcher::Batcher,
    },
    prelude::*,
};

fn main() {
    type MyBackend = burn::backend::Wgpu<f32, i32>;

    let device = Default::default();
    let model = MLPConfig {
        input_size: 1,
        hidden_size: 32,
        output_size: 1,
        num_hidden_layers: 3,
    }.init::<MyBackend>(&device);

    println!("{}", model);
}

#[derive(Debug, Module)]
pub struct MLP<B: Backend> {
    pub linear_input: Linear<B>,
    pub linear_hidden_vec: Vec<Linear<B>>,
    pub linear_output: Linear<B>,
    pub activation: Relu,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        //let [batch_size, input_size] = input.dims();

        let x = self.linear_input.forward(input);
        let x = self.activation.forward(x);
        let x = {
            let mut x = x;
            for linear in &self.linear_hidden_vec {
                x = linear.forward(x);
                x = self.activation.forward(x);
            }
            x
        };
        let x = self.linear_output.forward(x);
        x
    }
}

#[derive(Debug, Config)]
pub struct MLPConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub num_hidden_layers: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        let mut linear_hidden_vec = Vec::with_capacity(self.num_hidden_layers);
        for _ in 0..self.num_hidden_layers {
            linear_hidden_vec.push(
                LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            );
        }
        MLP {
            linear_input: LinearConfig::new(self.input_size, self.hidden_size).init(device),
            linear_hidden_vec,
            linear_output: LinearConfig::new(self.hidden_size, self.output_size).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Clone)]
pub struct SimpleBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SimpleBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

pub struct SimpleBatch<B: Backend> {
    pub input: Tensor<B, 2>,
    pub target: Tensor<B, 2>,
}
