use cust::{
    function::{BlockSize, GridSize},
    prelude::*,
};
use half::bf16;
use std::io;
use tokenizers::Tokenizer;

const kernels_ptx: &'static str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> io::Result<()> {
    let _ctx = cust::quick_init().unwrap();
    let module = Module::from_ptx(kernels_ptx, &[]).unwrap();
    // let vec_add = module.get_function("vecadd").unwrap();
    let embedding = module.get_function("embedding").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let prompt = "Hello, I'm a language model,";
    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();
    let encoding = tokenizer.encode(prompt, false).unwrap();
    let token_ids_host = encoding.get_ids();

    const B: usize = 1;
    let T: usize = token_ids_host.len();
    let V: usize = tokenizer.get_vocab_size(false);
    const D: usize = 768;

    let mut embeddings_host = vec![bf16::ZERO; V * D];
    for i in 0..V {
        for d in 0..D {
            embeddings_host[i * D + d] = bf16::from_f32(i as f32);
        }
    }

    let embeddings: DeviceBuffer<bf16> = DeviceBuffer::from_slice(&embeddings_host).unwrap();
    let token_ids: DeviceBuffer<u32> = DeviceBuffer::from_slice(&token_ids_host).unwrap();
    let mut token_embeddings: DeviceBuffer<bf16> =
        unsafe { DeviceBuffer::uninitialized(B * T * D).unwrap() };

    let grid_size = GridSize::xy(B as u32, T as u32);
    let block_size = BlockSize::xy(1, 1);
    unsafe {
        launch!(
            embedding<<<grid_size, block_size, 0, stream>>>(
                embeddings.as_device_ptr(),
                token_ids.as_device_ptr(),
                token_embeddings.as_device_ptr(),
                B,
                T,
                D,
                V,
            )
        );
    }

    stream.synchronize().unwrap();

    let mut out_host = vec![bf16::ZERO; B * T * D];
    token_embeddings
        .copy_to(&mut out_host[..])
        .expect("out_host should be long enough");

    // for b in 0..B {
    //     for t in 0..T {
    //         println!(
    //             "{:?}",
    //             &out_host[(b * T * D + t * D)..(b * T * D + t * D + D)]
    //         )
    //     }
    // }

    Ok(())
}
