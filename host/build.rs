fn main() {
    nvptx_builder::build_ptx_crate("kernels", "release-nvptx", true);
}
