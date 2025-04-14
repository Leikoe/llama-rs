# llama-rs

## Quickstart
```shell
rustup override set nightly
cd host
wget https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
cargo run
```

## Lil guide on nvptx64 target
The name of the target is `nvptx64-nvidia-cuda`.
You can compile to it using `cargo run --target nvptx64-nvidia-cuda`.

You can also specify a particular compute capability with `-C target-cpu=sm_80` rustflags options (usually through a `.cargo/config.toml`).
You can also specify a ptx version using `-C target-feature=+ptx86` (check llvm source for other ptx version feature names).
Use llvm-bitcode-linker with `-C linker-flavor=llbc`.


## Useful crates for nvptx64 (as of 04/14/2025)
- [nvptx-std](https://github.com/Gui-Yom/turbo-metrics/tree/master/crates/nvptx-std) for some std like experience on nvptx64
- [nvptx-builder](https://github.com/Gui-Yom/turbo-metrics/tree/master/crates/nvptx-builder) to build your kernels crate to a `.ptx` with a `build.rs` from your host crate.
- [cust](https://github.com/Rust-GPU/Rust-CUDA/tree/main/crates/cust) to easily use the `.ptx` kernels from host side Rust.
