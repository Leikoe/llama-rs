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
const V: usize = 50257;
const D: usize = 768;

struct Gpt2Weights {
    wte: DeviceBuffer<bf16>,         // (V, D)
    wpe: DeviceBuffer<bf16>,         // (1024, D)
    ln_f_weight: DeviceBuffer<bf16>, // (D,)
    ln_f_bias: DeviceBuffer<bf16>,   // (D,)
}

impl Gpt2Weights {
    fn new(safetensors_path: &str) -> Result<Self, safetensors::SafeTensorError> {
        let file = File::open(safetensors_path)?;
        let mut model = Gpt2Weights {
            wte: unsafe { DeviceBuffer::uninitialized(V * D).unwrap() },
            wpe: unsafe { DeviceBuffer::uninitialized(1024 * D).unwrap() },
            ln_f_weight: unsafe { DeviceBuffer::uninitialized(D).unwrap() },
            ln_f_bias: unsafe { DeviceBuffer::uninitialized(D).unwrap() },
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
    weights: Gpt2Weights,
    execution_ctx: &'a CudaExecutionContext,
}

impl<'a> Gpt2Model<'a> {
    fn new(
        safetensors_path: &str,
        execution_ctx: &'a CudaExecutionContext,
    ) -> Result<Self, safetensors::SafeTensorError> {
        Ok(Self {
            weights: Gpt2Weights::new(safetensors_path)?,
            execution_ctx,
        })
    }

    fn forward(&self, token_ids: &DeviceBuffer<u32>) -> DeviceBuffer<bf16> {
        let seq_len = token_ids.len();

        // wte
        let token_embeddings: DeviceBuffer<bf16> =
            self.execution_ctx
                .embedding(&self.weights.wte, V, D, &token_ids, B, seq_len);

        // wpe
        let positions_host = (0..seq_len as u32).collect::<Vec<u32>>();
        let positions: DeviceBuffer<u32> = DeviceBuffer::from_slice(&positions_host).unwrap();
        let token_position_embeddings: DeviceBuffer<bf16> =
            self.execution_ctx
                .embedding(&self.weights.wpe, 1024, D, &positions, B, seq_len);

        // h = wte + wpe
        let embeddings: DeviceBuffer<bf16> = self.execution_ctx.vec_add(
            &token_embeddings,
            &token_position_embeddings,
            B * seq_len * D,
        );

        self.execution_ctx.synchronize();

        embeddings
    }
}

fn main() -> io::Result<()> {
    let ctx = CudaExecutionContext::new();

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let model = Gpt2Model::new("model.safetensors", &ctx).unwrap();

    let prompt = "Hello, I'm a language model,";
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let token_ids_host = encoding.get_ids();
    let T: usize = token_ids_host.len();

    let token_ids: DeviceBuffer<u32> = DeviceBuffer::from_slice(&token_ids_host).unwrap();

    let logits_device = model.forward(&token_ids);

    let logits_host = logits_device.as_host_vec().unwrap();

    // argmax sampling
    // for b in 0..B {
    //     let last_token_logits = &logits_host[(b * T * D + (T - 1) * D)..(b * T * D + T * D)]; // (D)
    //     let argmax
    // }

    for b in 0..B {
        for t in 0..T {
            println!(
                "{:?}",
                &logits_host[(b * T * D + t * D)..(b * T * D + t * D + D)]
            )
        }
    }

    Ok(())
}
