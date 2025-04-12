use cust::{
    function::{BlockSize, GridSize},
    memory::DeviceCopy,
    prelude::*,
};

pub fn embedding<'a, T: DeviceCopy>(
    cuda_function: &Function<'a>,
    stream: &Stream,
    embedding_weights: &DeviceBuffer<T>, // (num_embeddings, embedding_dim)
    num_embeddings: usize,
    embedding_dim: usize,
    inputs: &DeviceBuffer<u32>, // (batch, tokens)
    batch: usize,
    tokens: usize,
) -> DeviceBuffer<T> {
    let mut embeddings: DeviceBuffer<T> =
        unsafe { DeviceBuffer::uninitialized(batch * tokens * embedding_dim).unwrap() }; // (batch, tokens, embedding_dim)

    let grid_size = GridSize::xy(batch as u32, tokens as u32);
    let block_size = BlockSize::xy(1, 1);
    unsafe {
        launch!(
            cuda_function<<<grid_size, block_size, 0, stream>>>(
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
