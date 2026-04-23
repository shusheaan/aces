//! GPU hardware probe — run with:
//!   cargo run -p aces-batch-sim --example gpu_probe --features gpu
//!
//! Reports available GPU adapters, compute capabilities, and runs
//! a minimal compute shader verification.

#[cfg(feature = "gpu")]
fn main() {
    aces_batch_sim::gpu::adapter::print_gpu_report();
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("GPU feature not enabled. Run with: --features gpu");
    std::process::exit(1);
}
