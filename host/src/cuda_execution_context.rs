use cust::{
    function::{BlockSize, GridSize},
    launch,
    memory::{DeviceBuffer, DeviceCopy},
    module::{Module, ModuleJitOption},
    prelude::Context,
    stream::{Stream, StreamFlags},
};
use half::bf16;

const KERNELS_PTX: &'static str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

pub struct CudaExecutionContext {
    _ctx: Context, // IMPORTANT: needs to be around for the full duration of the program
    module: Module,
    stream: Stream,
}

pub enum CudaErr {
    OOM,
}

impl CudaExecutionContext {
    pub fn new() -> Self {
        let ctx = cust::quick_init().unwrap();
        let module = Module::from_ptx(
            KERNELS_PTX,
            &[ModuleJitOption::Target(cust::module::JitTarget::Compute80)],
        )
        .unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        Self {
            _ctx: ctx,
            module,
            stream,
        }
    }

    pub fn embedding(
        &self,
        embedding_weights: &DeviceBuffer<bf16>, // (num_embeddings, embedding_dim)
        num_embeddings: usize,
        embedding_dim: usize,
        inputs: &DeviceBuffer<u32>, // (batch, tokens)
        batch: usize,
        tokens: usize,
    ) -> DeviceBuffer<bf16> {
        let module = &self.module;
        let stream = &self.stream;

        let mut embeddings: DeviceBuffer<bf16> =
            unsafe { DeviceBuffer::uninitialized(batch * tokens * embedding_dim).unwrap() }; // (batch, tokens, embedding_dim)

        let grid_size = GridSize::xy(batch as u32, tokens as u32);
        let block_size = BlockSize::xy(1, 1);
        unsafe {
            launch!(
                module.embedding<<<grid_size, block_size, 0, stream>>>(
                    embedding_weights.as_device_ptr(),
                    inputs.as_device_ptr(),
                    embeddings.as_device_ptr(),
                    batch,
                    tokens,
                    embedding_dim,
                    num_embeddings,
                )
            );
        }

        embeddings
    }

    pub fn vec_add(
        &self,
        a: &DeviceBuffer<bf16>, // (n,)
        b: &DeviceBuffer<bf16>, // (n,)
        n: usize,
    ) -> DeviceBuffer<bf16> {
        let module = &self.module;
        let stream = &self.stream;
        let mut res: DeviceBuffer<bf16> = unsafe { DeviceBuffer::uninitialized(n).unwrap() };

        let grid_size = GridSize::x((n as f32 / 256.0).ceil() as u32);
        let block_size = BlockSize::x(256);
        unsafe {
            launch!(
                module.vec_add<<<grid_size, block_size, 0, stream>>>(
                    a.as_device_ptr(),
                    b.as_device_ptr(),
                    res.as_device_ptr(),
                    n
                )
            );
        }

        res
    }

    pub fn empty<T: DeviceCopy>(len: usize) -> Result<DeviceBuffer<T>, CudaErr> {
        unsafe { DeviceBuffer::<T>::uninitialized(len).map_err(|_| CudaErr::OOM) }
    }

    pub fn synchronize(&self) {
        self.stream.synchronize().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use cust::memory::DeviceBuffer;
    use half::bf16;

    use crate::cuda_execution_context::CudaExecutionContext;

    #[test]
    fn test_vec_add_simple() {
        let ctx = CudaExecutionContext::new();
        const N: usize = 1024;

        // create a
        let mut a_host = unsafe { Box::<[bf16; N]>::new_uninit().assume_init() };
        for i in 0..N {
            a_host[i] = bf16::from_f32(i as f32);
        }
        let a = DeviceBuffer::<bf16>::from_slice(a_host.as_slice()).unwrap();

        // create b
        let mut b_host = unsafe { Box::<[bf16; N]>::new_uninit().assume_init() };
        for i in 0..N {
            b_host[i] = bf16::from_f32(N as f32 - i as f32);
        }
        let b = DeviceBuffer::<bf16>::from_slice(b_host.as_slice()).unwrap();

        let out = ctx.vec_add(&a, &b, N);
        let out_host = out.as_host_vec().unwrap();

        for i in 0..N {
            assert_eq!(out_host[i], a_host[i] + b_host[i]);
        }
    }

    #[test]
    fn test_embedding_simple() {
        let ctx = CudaExecutionContext::new();
        const B: usize = 16;
        const T: usize = 64;
        const V: usize = 50257;
        const D: usize = 768;

        // create embedding weights
        // 000000
        // 111111
        // 222222
        // ......
        let mut embedding_weights_host =
            unsafe { Box::<[bf16; V * D]>::new_uninit().assume_init() };
        for i in 0..V {
            for j in 0..D {
                embedding_weights_host[i * D + j] = bf16::from_f32(i as f32);
            }
        }
        let embedding_weights_device =
            DeviceBuffer::<bf16>::from_slice(embedding_weights_host.as_slice()).unwrap();

        // create token ids
        let mut token_ids_host = unsafe { Box::<[u32; B * T]>::new_uninit().assume_init() };
        for b in 0..B {
            for t in 0..T {
                token_ids_host[b * T + t] = t as u32;
            }
        }
        let token_ids_device = DeviceBuffer::<u32>::from_slice(token_ids_host.as_slice()).unwrap();

        let out = ctx.embedding(&embedding_weights_device, V, D, &token_ids_device, B, T);
        let out_host = out.as_host_vec().unwrap(); // (B, T, D)

        for b in 0..B {
            for t in 0..T {
                let token_id = token_ids_host[b * T + t] as usize;
                let token_embedding = &out_host[(b * T * D + t * D)..(b * T * D + t * D + D)];
                assert_eq!(
                    token_embedding,
                    &embedding_weights_host[(token_id * D)..(token_id * D + D)]
                );
            }
        }
    }
}
