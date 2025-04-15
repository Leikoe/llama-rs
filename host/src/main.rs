use cust::prelude::*;
use half::bf16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::fs::File;
use std::io;
use tokenizers::Tokenizer;

mod cuda_execution_context;
use cuda_execution_context::CudaExecutionContext;

const B: usize = 1;

#[derive(Debug, Clone, Copy)]
struct Gpt2Config {
    n_vocab: usize,
    max_seq_len: usize,
    n_layers: usize,
    n_heads: usize,
    n_embed: usize,
}

const GPT2CONFIG: Gpt2Config = Gpt2Config {
    n_vocab: 50257,
    max_seq_len: 1024,
    n_layers: 12,
    n_heads: 12,
    n_embed: 768,
};

struct Gpt2Weights {
    wte: DeviceBuffer<bf16>,         // (V, D)
    wpe: DeviceBuffer<bf16>,         // (1024, D)
    ln_f_weight: DeviceBuffer<bf16>, // (D,)
    ln_f_bias: DeviceBuffer<bf16>,   // (D,)
}

impl Gpt2Weights {
    fn new(
        config: Gpt2Config,
        safetensors_path: &str,
    ) -> Result<Self, safetensors::SafeTensorError> {
        let file = File::open(safetensors_path)?;
        let mut model = Gpt2Weights {
            wte: unsafe { DeviceBuffer::uninitialized(config.n_vocab * config.n_embed).unwrap() },
            wpe: unsafe {
                DeviceBuffer::uninitialized(config.max_seq_len * config.n_embed).unwrap()
            },
            ln_f_weight: unsafe { DeviceBuffer::uninitialized(config.n_embed).unwrap() },
            ln_f_bias: unsafe { DeviceBuffer::uninitialized(config.n_embed).unwrap() },
        };

        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer)?;
        // for (k, v) in tensors.tensors() {
        //     println!("{}: shape={:?} dtype={:?}", k, v.shape(), v.dtype());
        // }

        Self::load_tensor(&mut model.wte, tensors.tensor("wte.weight").unwrap());
        Self::load_tensor(&mut model.wpe, tensors.tensor("wpe.weight").unwrap());
        Self::load_tensor(
            &mut model.ln_f_weight,
            tensors.tensor("ln_f.weight").unwrap(),
        );
        Self::load_tensor(&mut model.ln_f_bias, tensors.tensor("ln_f.bias").unwrap());

        Ok(model)
    }

    fn load_tensor(dst: &mut DeviceBuffer<bf16>, src: TensorView<'_>) {
        assert!(src.dtype() == Dtype::F32);
        let tensor_numel: usize = src.shape().into_iter().product();
        assert!(dst.len() == tensor_numel);
        let tensor_data_ptr = src.data().as_ptr() as *const f32;
        let mut host_buffer =
            unsafe { Box::<[bf16]>::new_uninit_slice(tensor_numel).assume_init() };
        for i in 0..tensor_numel {
            host_buffer[i] = bf16::from_f32(unsafe { tensor_data_ptr.add(i).read() });
        }
        dst.copy_from(&mut host_buffer).unwrap();
    }
}

struct Gpt2Model<'a> {
    config: Gpt2Config,
    weights: Gpt2Weights,
    execution_ctx: &'a CudaExecutionContext,
}

impl<'a> Gpt2Model<'a> {
    fn new(
        safetensors_path: &str,
        config: Gpt2Config,
        execution_ctx: &'a CudaExecutionContext,
    ) -> Result<Self, safetensors::SafeTensorError> {
        Ok(Self {
            config,
            weights: Gpt2Weights::new(config, safetensors_path)?,
            execution_ctx,
        })
    }

    fn forward(&self, token_ids: &DeviceBuffer<u32>) -> DeviceBuffer<bf16> {
        let seq_len = token_ids.len();

        // wte
        let token_embeddings: DeviceBuffer<bf16> = self.execution_ctx.embedding(
            &self.weights.wte,
            self.config.n_vocab,
            self.config.n_embed,
            &token_ids,
            B,
            seq_len,
        );

        // wpe
        let positions_host = (0..seq_len as u32).collect::<Vec<u32>>();
        let positions: DeviceBuffer<u32> = DeviceBuffer::from_slice(&positions_host).unwrap();
        let token_position_embeddings: DeviceBuffer<bf16> = self.execution_ctx.embedding(
            &self.weights.wpe,
            self.config.max_seq_len,
            self.config.n_embed,
            &positions,
            B,
            seq_len,
        );

        // h = wte + wpe
        let embeddings: DeviceBuffer<bf16> = self.execution_ctx.vec_add(
            &token_embeddings,
            &token_position_embeddings,
            B * seq_len * self.config.n_embed,
        );

        // apply blocks

        // h = self.ln_f(h)
        let h = self.execution_ctx.layer_norm(
            &self.weights.ln_f_weight,
            &self.weights.ln_f_bias,
            1e-5,
            &embeddings,
            B,
            seq_len,
            self.config.n_embed,
        );

        self.execution_ctx.synchronize();

        h
    }
}

fn main() -> io::Result<()> {
    let ctx = CudaExecutionContext::new();

    let model_config = GPT2CONFIG;
    let model = Gpt2Model::new("model.safetensors", model_config, &ctx).unwrap();
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();

    let prompt = "Hello, I'm a language model,";
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let mut token_ids_host = encoding.get_ids().to_vec();

    for _ in 0..1024 {
        let T: usize = token_ids_host.len();

        let token_ids: DeviceBuffer<u32> = DeviceBuffer::from_slice(&token_ids_host).unwrap(); // NOTE: this is very stupid, it could work without it because of BS=1 but I'm still doing it for correctness

        let logits_device = model.forward(&token_ids);

        let logits_host = logits_device.as_host_vec().unwrap();

        // argmax sampling
        let b = 0; // only decode batch 0
        let last_token_logits = &logits_host[(b * T * model.config.n_embed
            + (T - 1) * model.config.n_embed)
            ..(b * T * model.config.n_embed + T * model.config.n_embed)]; // (D)
        let (next_token_id, _max) = last_token_logits
            .into_iter()
            .enumerate()
            .max_by(|(_idx1, val1), (_idx2, val2)| val1.total_cmp(val2))
            .unwrap();

        // let next_token = tokenizer.decode(&[next_token_id as u32], false).unwrap();
        // println!("next token: {} ({})", next_token, next_token_id);
        token_ids_host.push(next_token_id as u32);
        println!("{}", tokenizer.decode(&token_ids_host, false).unwrap());
    }

    Ok(())
}
