use cust::{
    function::{BlockSize, GridSize},
    memory::DeviceCopy,
    prelude::*,
};

pub fn vecadd<'a, T: DeviceCopy>(
    cuda_function: &Function<'a>,
    stream: &Stream,
    a: &DeviceBuffer<T>, // (n,)
    b: &DeviceBuffer<T>, // (n,)
    n: usize,
) -> DeviceBuffer<T> {
    let mut res: DeviceBuffer<T> = unsafe { DeviceBuffer::uninitialized(n).unwrap() };

    let grid_size = GridSize::x((n as f32 / 256.0).ceil() as u32);
    let block_size = BlockSize::x(256);
    unsafe {
        launch!(
            cuda_function<<<grid_size, block_size, 0, stream>>>(
                a.as_device_ptr(),
                b.as_device_ptr(),
                res.as_device_ptr(),
                n
            )
        );
    }

    res
}
