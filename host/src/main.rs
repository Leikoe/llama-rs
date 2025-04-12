use cust::{
    function::{BlockSize, GridSize},
    prelude::*,
};

const kernels_ptx: &'static str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() {
    let _ctx = cust::quick_init().unwrap();
    let module = Module::from_ptx(kernels_ptx, &[]).unwrap();
    let vec_add = module.get_function("vecadd").unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    const N: usize = 10240;
    let a = DeviceBuffer::from_slice(&[1.0f32; N]).unwrap();
    let b = DeviceBuffer::from_slice(&[2.0f32; N]).unwrap();
    let mut out = DeviceBuffer::from_slice(&[0.0f32; N]).unwrap();

    let grid_size = GridSize::x((N / 256) as u32);
    let block_size = BlockSize::x(256);
    for i in 0..100 {
        unsafe {
            launch!(
                vec_add<<<grid_size, block_size, 0, stream>>>(
                    a.as_device_ptr(),
                    b.as_device_ptr(),
                    out.as_device_ptr()
                )
            );
        }
    }

    stream.synchronize().unwrap();

    let mut out_host = [0.0f32; N];
    out.copy_to(&mut out_host[..]);
    println!("DONE! {:?}", out_host);
}
