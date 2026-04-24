//! GPU vs CPU-reference MPPI benchmark.
//!
//! Run with:
//!   cargo run -p aces-batch-sim --features gpu --example bench_gpu_vs_cpu --release
//!
//! Runs several (n_drones, n_samples, horizon) configurations on BOTH the
//! WGSL GPU pipeline and the pure-CPU reference implementation, measures
//! wall-clock per iteration, and prints a comparison table with the max
//! absolute difference between GPU and CPU outputs (end-to-end parity check).
//!
//! If no GPU adapter is available (e.g. inside some sandboxes), the CPU
//! columns still produce numbers while the GPU columns show NaN.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("GPU feature not enabled. Run with: --features gpu");
    std::process::exit(1);
}

#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use nalgebra::Vector3;
#[cfg(feature = "gpu")]
use rand::{rngs::SmallRng, Rng, SeedableRng};
#[cfg(feature = "gpu")]
use rand_distr::{Distribution, Normal};

#[cfg(feature = "gpu")]
use aces_batch_sim::f32_cost::CostWeightsF32;
#[cfg(feature = "gpu")]
use aces_batch_sim::f32_dynamics::DroneParamsF32;
#[cfg(feature = "gpu")]
use aces_batch_sim::f32_sdf::{ArenaF32, ObstacleF32};
#[cfg(feature = "gpu")]
use aces_batch_sim::gpu::adapter::probe_gpu;
#[cfg(feature = "gpu")]
use aces_batch_sim::gpu::pipeline::{
    compute_batch_actions_cpu_reference, CostWeightsGpu, GpuBatchMppi, MppiDims,
};

#[cfg(feature = "gpu")]
fn warehouse_arena() -> ArenaF32 {
    let mut arena = ArenaF32::new(Vector3::new(10.0, 10.0, 3.0));
    for (x, y) in [
        (2.0f32, 2.0f32),
        (2.0, 8.0),
        (5.0, 5.0),
        (8.0, 2.0),
        (8.0, 8.0),
    ] {
        arena.obstacles.push(ObstacleF32::Box {
            center: Vector3::new(x, y, 1.5),
            half_extents: Vector3::new(0.5, 0.5, 1.5),
        });
    }
    arena
}

#[cfg(feature = "gpu")]
fn default_gpu_weights() -> CostWeightsGpu {
    CostWeightsGpu::new(1.0, 5.0, 0.01, 1000.0, 0.3, 0.0, [10.0, 10.0, 3.0])
}

#[cfg(feature = "gpu")]
fn generate_inputs(
    n_drones: usize,
    n_samples: usize,
    horizon: usize,
    hover_thrust: f32,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = SmallRng::seed_from_u64(seed);

    // States: identity quaternion + random positions inside arena.
    let mut states = vec![0.0f32; n_drones * 13];
    for d in 0..n_drones {
        let b = d * 13;
        states[b] = rng.gen_range(2.0..8.0);
        states[b + 1] = rng.gen_range(2.0..8.0);
        states[b + 2] = rng.gen_range(1.0..2.5);
        states[b + 9] = 1.0; // quat w (identity)
    }

    // Enemies: pair drone d with drone (d ^ 1); if the sibling is out of
    // range (odd n_drones), self-pair to keep the buffer well-formed.
    let mut enemies = vec![0.0f32; n_drones * 13];
    for d in 0..n_drones {
        let enemy_d = d ^ 1;
        let enemy_idx = if enemy_d < n_drones { enemy_d } else { d };
        enemies[d * 13..(d + 1) * 13]
            .copy_from_slice(&states[enemy_idx * 13..(enemy_idx + 1) * 13]);
    }

    let mean_ctrls = vec![hover_thrust; n_drones * horizon * 4];

    let noise_len = n_drones * n_samples * horizon * 4;
    let normal = Normal::new(0.0f32, 0.03).unwrap();
    let noise: Vec<f32> = (0..noise_len).map(|_| normal.sample(&mut rng)).collect();

    (states, enemies, mean_ctrls, noise)
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn bench_config(
    label: &str,
    n_drones: usize,
    n_samples: usize,
    horizon: usize,
    gpu_available: bool,
) {
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena();
    let gpu_weights = default_gpu_weights();
    let cpu_weights = CostWeightsF32::default();
    let hover = params.hover_thrust();

    let (states, enemies, mean_ctrls, noise) =
        generate_inputs(n_drones, n_samples, horizon, hover, 42);

    let dims = MppiDims::new(
        n_drones as u32,
        n_samples as u32,
        horizon as u32,
        10,
        arena.obstacles.len() as u32,
        0.001,
    );

    // --- CPU reference ---
    // Warm-up
    let _ = compute_batch_actions_cpu_reference(
        &params,
        &arena,
        &cpu_weights,
        gpu_weights.hover,
        dims,
        &states,
        &enemies,
        &mean_ctrls,
        &noise,
    );

    let iters = if n_samples * horizon * n_drones > 100_000 {
        3
    } else {
        10
    };
    let t0 = Instant::now();
    let mut cpu_last = vec![];
    for _ in 0..iters {
        cpu_last = compute_batch_actions_cpu_reference(
            &params,
            &arena,
            &cpu_weights,
            gpu_weights.hover,
            dims,
            &states,
            &enemies,
            &mean_ctrls,
            &noise,
        );
    }
    let cpu_elapsed = t0.elapsed();
    let cpu_per_iter_ms = cpu_elapsed.as_secs_f64() * 1000.0 / iters as f64;

    // --- GPU ---
    let (gpu_per_iter_ms, max_diff) = if gpu_available {
        match GpuBatchMppi::new(n_drones, n_samples, horizon, &params, gpu_weights, &arena) {
            Ok(pipeline) => {
                // Warm-up
                let _ = pipeline.compute_batch_actions(&states, &enemies, &mean_ctrls, &noise);

                let t0 = Instant::now();
                let mut last = vec![];
                for _ in 0..iters {
                    last = pipeline.compute_batch_actions(&states, &enemies, &mean_ctrls, &noise);
                }
                let elapsed = t0.elapsed();
                let per_iter = elapsed.as_secs_f64() * 1000.0 / iters as f64;

                // Parity check: max absolute difference over the whole output.
                let diff = cpu_last
                    .iter()
                    .zip(last.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);

                (per_iter, diff)
            }
            Err(e) => {
                eprintln!("  [gpu pipeline init failed for {label}: {e}]");
                (f64::NAN, f32::NAN)
            }
        }
    } else {
        (f64::NAN, f32::NAN)
    };

    let speedup = if gpu_available && gpu_per_iter_ms > 0.0 && gpu_per_iter_ms.is_finite() {
        cpu_per_iter_ms / gpu_per_iter_ms
    } else {
        f64::NAN
    };

    println!(
        "{:<12} n_drones={:<4} n_samples={:<5} horizon={:<3} | CPU: {:>9.2} ms | GPU: {:>9.2} ms | speedup: {:>7.2}x | max_diff: {:.3e}",
        label, n_drones, n_samples, horizon, cpu_per_iter_ms, gpu_per_iter_ms, speedup, max_diff
    );
}

#[cfg(feature = "gpu")]
fn main() {
    let probe = probe_gpu();
    let gpu_available = probe.compute_capable;

    if gpu_available {
        let name = probe
            .adapters
            .first()
            .map(|a| a.name.as_str())
            .unwrap_or("unknown");
        println!("GPU available: {name}");
    } else {
        println!("No GPU adapter detected — CPU-only timings will be reported.");
    }
    println!();
    println!(
        "{:<12} {:<13} {:<14} {:<11} | {:>14} | {:>14} | {:>16} | max_diff",
        "config",
        "n_drones",
        "n_samples",
        "horizon",
        "CPU (ms/iter)",
        "GPU (ms/iter)",
        "speedup (CPU/GPU)"
    );
    println!("{}", "-".repeat(160));

    // Tiny (sanity check)
    bench_config("tiny", 4, 32, 10, gpu_available);
    // Small
    bench_config("small", 16, 64, 15, gpu_available);
    // Medium
    bench_config("medium", 64, 128, 20, gpu_available);
    // Large
    bench_config("large", 128, 256, 30, gpu_available);
    // Production-like. The CPU reference over 128×1024×50 can be slow on
    // machines without a GPU. The 3-iter floor inside `bench_config`
    // already limits this; we also skip it entirely if no GPU is present
    // so we don't burn minutes of CI time on a CPU-only comparison.
    if gpu_available {
        bench_config("production", 128, 1024, 50, gpu_available);
    } else {
        bench_config("prod-lite", 64, 512, 30, gpu_available);
    }
}
