use cust::{
    function::{BlockSize, GridSize},
    prelude::*,
};
use half::bf16;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;
use std::io;
use tokenizers::Tokenizer;

mod layers;

const kernels_ptx: &'static str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

const B: usize = 1;
const V: usize = 50257;
const D: usize = 768;

struct Gpt2 {
    wte: Box<[bf16; V * D]>,
    wpe: Box<[bf16; 1024 * D]>,
}

impl Gpt2 {
    fn new(safetensors_path: &str) -> Result<Self, safetensors::SafeTensorError> {
        let file = File::open(safetensors_path)?;
        let mut model = unsafe {
            Gpt2 {
                wte: Box::<[bf16; V * D]>::new_uninit().assume_init(),
                wpe: Box::<[bf16; 1024 * D]>::new_uninit().assume_init(),
            }
        };

        let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(&buffer)?;
        // for (k, v) in tensors.tensors() {
        //     println!("{}: shape={:?} dtype={:?}", k, v.shape(), v.dtype());
        // }
        let wte_view = tensors.tensor("wte.weight").unwrap();
        let wte_ptr = wte_view.data().as_ptr() as *const f32;
        for i in 0..wte_view.shape().into_iter().product() {
            model.wte[i] = bf16::from_f32(unsafe { wte_ptr.add(i).read() });
        }

        let wpe_view = tensors.tensor("wpe.weight").unwrap();
        let wpe_ptr = wpe_view.data().as_ptr() as *const f32;
        for i in 0..wpe_view.shape().into_iter().product() {
            model.wte[i] = bf16::from_f32(unsafe { wpe_ptr.add(i).read() });
        }

        Ok(model)
    }
}

fn main() -> io::Result<()> {
    let _ctx = cust::quick_init().unwrap();
    let module = Module::from_ptx(kernels_ptx, &[]).unwrap();
    // let vec_add = module.get_function("vecadd").unwrap();
    let embedding_kernel_fn = module.get_function("embedding").unwrap();
    let vecadd_kernel_fn = module.get_function("vecadd").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    // TOKENIZER
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();

    // MODEL
    let model = Gpt2::new("model.safetensors").unwrap();

    let prompt = "Hello, I'm a language model,";
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let token_ids_host = encoding.get_ids();
    let T: usize = token_ids_host.len();
    let token_ids: DeviceBuffer<u32> = DeviceBuffer::from_slice(&token_ids_host).unwrap();

    // wte
    let wte: DeviceBuffer<bf16> = DeviceBuffer::from_slice(model.wte.as_slice()).unwrap();
    let token_embeddings: DeviceBuffer<bf16> =
        layers::embedding(&embedding_kernel_fn, &stream, &wte, V, D, &token_ids, B, T);

    // wpe
    let wpe: DeviceBuffer<bf16> = DeviceBuffer::from_slice(model.wpe.as_slice()).unwrap();
    let positions_host = (0..T as u32).collect::<Vec<u32>>();
    let positions: DeviceBuffer<u32> = DeviceBuffer::from_slice(&positions_host).unwrap();
    let mut token_position_embeddings: DeviceBuffer<bf16> = layers::embedding(
        &embedding_kernel_fn,
        &stream,
        &wpe,
        1024,
        D,
        &positions,
        B,
        T,
    );

    // sum
    let embeddings: DeviceBuffer<bf16> = layers::vecadd(
        &vecadd_kernel_fn,
        &stream,
        &token_embeddings,
        &token_position_embeddings,
        B * T * D,
    );

    stream.synchronize().unwrap();

    let mut out_host = vec![bf16::ZERO; B * T * D];
    embeddings
        .copy_to(&mut out_host[..])
        .expect("out_host should be long enough");

    for b in 0..B {
        for t in 0..T {
            println!(
                "{:?}",
                &out_host[(b * T * D + t * D)..(b * T * D + t * D + D)]
            )
        }
    }

    Ok(())
}
