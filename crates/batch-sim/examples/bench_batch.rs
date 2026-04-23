//! Batch simulation throughput benchmark.
//!
//! Run with:
//!   cargo run -p aces-batch-sim --example bench_batch --release
//!
//! Tests multiple batch sizes and MPPI configs to find optimal throughput.

use aces_batch_sim::battle::BatchConfig;
use aces_batch_sim::orchestrator::{BatchOrchestrator, MppiConfig};
use aces_batch_sim::reward::RewardConfig;
use aces_sim_core::dynamics::DroneParams;
use aces_sim_core::environment::{Arena, Obstacle};
use aces_sim_core::lockon::LockOnParams;
use nalgebra::Vector3;
use std::time::Instant;

fn test_arena() -> Arena {
    let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
    for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
        arena.obstacles.push(Obstacle::Box {
            center: Vector3::new(x, y, 1.5),
            half_extents: Vector3::new(0.5, 0.5, 1.5),
        });
    }
    arena
}

fn bench_config(n_battles: usize, mppi_samples: usize, mppi_horizon: usize, n_steps: usize) {
    let arena = test_arena();
    let params = DroneParams::crazyflie();
    let lockon = LockOnParams::default();
    let mppi_config = MppiConfig {
        num_samples: mppi_samples,
        horizon: mppi_horizon,
        ..Default::default()
    };
    let batch_config = BatchConfig::default();
    let reward_config = RewardConfig::default();

    let mut orch = BatchOrchestrator::new(
        n_battles,
        arena,
        params,
        lockon,
        mppi_config,
        batch_config,
        reward_config,
    );

    // Warm-up (2 steps)
    orch.step_all();
    orch.step_all();
    orch.reset_stats();

    // Timed run
    let start = Instant::now();
    for _ in 0..n_steps {
        orch.step_all();
    }
    let elapsed = start.elapsed();

    let total_env_steps = n_battles as u64 * n_steps as u64;
    let steps_per_sec = total_env_steps as f64 / elapsed.as_secs_f64();
    let ms_per_tick = elapsed.as_secs_f64() * 1000.0 / n_steps as f64;

    let stats = orch.stats();

    println!(
        "  battles={:<4} mppi_samples={:<5} horizon={:<3} | {:.0} env-steps/s | {:.2} ms/tick | episodes={} kills_a={} kills_b={} collisions={} timeouts={}",
        n_battles,
        mppi_samples,
        mppi_horizon,
        steps_per_sec,
        ms_per_tick,
        stats.total_episodes,
        stats.kills_a,
        stats.kills_b,
        stats.collisions_a + stats.collisions_b,
        stats.timeouts,
    );
}

fn main() {
    println!("=== ACES Batch Simulation Benchmark ===\n");
    println!("System: {} logical cores\n", rayon::current_num_threads());

    let n_steps = 50;

    println!("--- Varying batch size (MPPI: 64 samples, 10 horizon) ---");
    for n_battles in [1, 2, 4, 8, 16, 32, 64] {
        bench_config(n_battles, 64, 10, n_steps);
    }

    println!("\n--- Varying MPPI samples (16 battles, 10 horizon) ---");
    for samples in [16, 32, 64, 128, 256] {
        bench_config(16, samples, 10, n_steps);
    }

    println!("\n--- Varying MPPI horizon (16 battles, 64 samples) ---");
    for horizon in [5, 10, 15, 25, 50] {
        bench_config(16, 64, horizon, n_steps);
    }

    println!("\n--- Production configs ---");
    // Config matching rules.toml: 1024 samples, 50 horizon
    println!("  Full MPPI (rules.toml: 1024 samples, 50 horizon):");
    bench_config(1, 1024, 50, 10);
    bench_config(4, 1024, 50, 10);

    // Lighter config for batch training
    println!("  Batch training (128 samples, 15 horizon):");
    bench_config(16, 128, 15, n_steps);
    bench_config(32, 128, 15, n_steps);
    bench_config(64, 128, 15, n_steps);

    println!("\nDone.");
}
