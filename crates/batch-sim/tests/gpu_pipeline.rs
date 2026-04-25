//! Integration tests for the GPU MPPI pipeline skeleton.
//!
//! Feature-gated behind `gpu` — this file is empty unless the feature is
//! enabled. Tests that need a live GPU early-return on CI machines without
//! one (via `probe_gpu().compute_capable`). Pure struct-layout checks always
//! run.

#![cfg(feature = "gpu")]

use aces_batch_sim::f32_cost::CostWeightsF32;
use aces_batch_sim::f32_dynamics::DroneParamsF32;
use aces_batch_sim::f32_sdf::{ArenaF32, ObstacleF32};
use aces_batch_sim::gpu::adapter::probe_gpu;
use aces_batch_sim::gpu::pipeline::{
    compute_batch_actions_cpu_reference, compute_batch_actions_cpu_reference_with_intermediates,
    CostWeightsGpu, DroneParamsGpu, GpuBatchMppi, GpuInitError, MppiDims, ObstacleGpu,
    MAX_OBSTACLES,
};
use nalgebra::Vector3;

/// Default warehouse arena: 10x10x3m with 5 box pillars. Mirrors the helper
/// in `f32_sdf` tests but built directly in f32.
fn warehouse_arena_f32() -> ArenaF32 {
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

fn default_weights() -> CostWeightsGpu {
    CostWeightsGpu::new(1.0, 0.5, 0.01, 10.0, 0.5, 0.0, [10.0, 10.0, 3.0])
}

fn gpu_available_or_skip(test_name: &str) -> bool {
    let probe = probe_gpu();
    if !probe.compute_capable {
        eprintln!("[{test_name}] skipped: no GPU adapter available");
        return false;
    }
    true
}

// ----- Tests that require a live GPU -----

#[test]
fn test_pipeline_creates_with_default_warehouse() {
    if !gpu_available_or_skip("test_pipeline_creates_with_default_warehouse") {
        return;
    }
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena_f32();
    let weights = default_weights();

    let pipeline =
        GpuBatchMppi::new(8, 64, 20, &params, weights, &arena).expect("pipeline construction");

    assert_eq!(pipeline.n_drones, 8);
    assert_eq!(pipeline.n_samples, 64);
    assert_eq!(pipeline.horizon, 20);
    assert_eq!(pipeline.n_obstacles, 5);
}

#[test]
fn test_pipeline_buffer_sizes() {
    if !gpu_available_or_skip("test_pipeline_buffer_sizes") {
        return;
    }
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena_f32();
    let weights = default_weights();

    let n_drones = 8usize;
    let n_samples = 64usize;
    let horizon = 20usize;

    let pipeline = GpuBatchMppi::new(n_drones, n_samples, horizon, &params, weights, &arena)
        .expect("pipeline construction");

    // Literal expected sizes (bytes). 4 bytes/element (f32 or u32).
    // Derivation with n_drones=8, n_samples=64, horizon=20:
    //   states:          n_drones * 13 * 4                       = 8 * 13 * 4               = 416
    //   enemies:         n_drones * 13 * 4                       = 8 * 13 * 4               = 416
    //   mean_ctrls:      n_drones * horizon * 4 * 4              = 8 * 20 * 4 * 4           = 2560
    //   noise:           n_drones * n_samples * horizon * 4 * 4  = 8 * 64 * 20 * 4 * 4      = 163840
    //   costs:           n_drones * n_samples * 4                = 8 * 64 * 4               = 2048
    //   ctrls_out:       n_drones * n_samples * horizon * 4 * 4  = 8 * 64 * 20 * 4 * 4      = 163840
    //   result:          n_drones * horizon * 4 * 4              = 8 * 20 * 4 * 4           = 2560
    //   params_uniform:  size_of::<DroneParamsGpu>()             = 48
    //   weights_uniform: size_of::<CostWeightsGpu>()             = 48
    //   obstacles:       MAX_OBSTACLES * size_of::<ObstacleGpu>()= 32 * 48                  = 1536
    // Updating dims? Recompute the numbers above to match.
    assert_eq!(pipeline.states_buffer.size(), 416);
    assert_eq!(pipeline.enemies_buffer.size(), 416);
    assert_eq!(pipeline.mean_ctrls_buffer.size(), 2560);
    assert_eq!(pipeline.noise_buffer.size(), 163840);
    assert_eq!(pipeline.costs_buffer.size(), 2048);
    assert_eq!(pipeline.ctrls_out_buffer.size(), 163840);
    assert_eq!(pipeline.result_buffer.size(), 2560);
    assert_eq!(pipeline.params_uniform.size(), 48);
    assert_eq!(pipeline.weights_uniform.size(), 48);
    assert_eq!(pipeline.obstacles_buffer.size(), 1536);
    // `MppiDims` uniform is always 32 bytes — see `test_mppi_dims_size_is_32`.
    assert_eq!(pipeline.dims_uniform.size(), 32);
}

#[test]
fn test_pipeline_exposes_dims_uniform() {
    if !gpu_available_or_skip("test_pipeline_exposes_dims_uniform") {
        return;
    }
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena_f32();
    let weights = default_weights();

    let pipeline =
        GpuBatchMppi::new(4, 32, 10, &params, weights, &arena).expect("pipeline construction");

    // MppiDims is 32 bytes (size_of check above), and `new` must have
    // allocated a UNIFORM-usage buffer of exactly that size.
    assert_eq!(pipeline.dims_uniform.size(), 32);
    assert!(pipeline
        .dims_uniform
        .usage()
        .contains(wgpu::BufferUsages::UNIFORM));

    // update_dims must succeed without panicking. We can't easily read
    // back the uniform without a dispatch; a smoke-level write is enough
    // to catch buffer-kind mismatches in debug builds.
    let new_dims = MppiDims::new(4, 32, 10, 20, arena.obstacles.len() as u32, 0.0005);
    pipeline.update_dims(new_dims);
}

#[test]
fn test_pipeline_builds_compute_pipelines() {
    // Construction succeeds and both compute pipelines are owned by the
    // struct. We can't introspect wgpu::ComputePipeline much — the
    // fact that `create_compute_pipeline` didn't panic already tells us
    // the WGSL compiled on this device. Presence of the fields is the
    // assertion (they are non-Option, so the compile itself is the check).
    if !gpu_available_or_skip("test_pipeline_builds_compute_pipelines") {
        return;
    }
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena_f32();
    let weights = default_weights();

    let pipeline =
        GpuBatchMppi::new(4, 32, 10, &params, weights, &arena).expect("pipeline construction");

    // Take references to force the compiler to acknowledge the fields exist;
    // if any field is renamed or removed, this test will stop compiling.
    let _rollout: &wgpu::ComputePipeline = &pipeline.rollout_pipeline;
    let _softmax: &wgpu::ComputePipeline = &pipeline.softmax_pipeline;
    let _bgl: &wgpu::BindGroupLayout = &pipeline.bind_group_layout;
}

#[test]
fn test_bind_group_assembles_against_layout() {
    // Stronger than just field-presence: try to actually build a BindGroup
    // from the pipeline's layout + its own buffers. This exercises the
    // consistency between BindGroupLayoutEntry usage flags (Storage vs Uniform)
    // and the buffer creation flags (STORAGE vs UNIFORM). If any entry in
    // `bind_group_layout_entries()` disagrees with the buffer's `usage`,
    // this call panics with a descriptive wgpu validation error.
    if !gpu_available_or_skip("test_bind_group_assembles_against_layout") {
        return;
    }
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena_f32();
    let weights = default_weights();

    let pipeline =
        GpuBatchMppi::new(4, 32, 10, &params, weights, &arena).expect("pipeline construction");

    let _bind_group = pipeline
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mppi.test_bind_group"),
            layout: &pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pipeline.states_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pipeline.enemies_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pipeline.mean_ctrls_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pipeline.noise_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pipeline.costs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: pipeline.ctrls_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pipeline.result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pipeline.params_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: pipeline.weights_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: pipeline.obstacles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: pipeline.dims_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: pipeline.wind_buffer.as_entire_binding(),
                },
            ],
        });
}

#[test]
fn test_pipeline_rejects_too_many_obstacles() {
    if !gpu_available_or_skip("test_pipeline_rejects_too_many_obstacles") {
        return;
    }
    let params = DroneParamsF32::crazyflie();
    let weights = default_weights();
    let mut arena = ArenaF32::new(Vector3::new(10.0, 10.0, 3.0));
    // 33 = MAX_OBSTACLES + 1
    for i in 0..(MAX_OBSTACLES + 1) {
        arena.obstacles.push(ObstacleF32::Sphere {
            center: Vector3::new(i as f32 * 0.1, 0.0, 1.0),
            radius: 0.1,
        });
    }
    assert_eq!(arena.obstacles.len(), MAX_OBSTACLES + 1);

    let result = GpuBatchMppi::new(4, 32, 10, &params, weights, &arena);
    match result {
        Ok(_) => panic!("expected TooManyObstacles error, got Ok"),
        Err(GpuInitError::TooManyObstacles(n)) => assert_eq!(n, MAX_OBSTACLES + 1),
        Err(other) => panic!("expected TooManyObstacles, got {other:?}"),
    }
}

// ----- compute_batch_actions dispatch tests -----

/// Tiny helper: build a pipeline with the warehouse arena and default weights.
fn make_pipeline(n_drones: usize, n_samples: usize, horizon: usize) -> GpuBatchMppi {
    let params = DroneParamsF32::crazyflie();
    let arena = warehouse_arena_f32();
    let weights = default_weights();
    GpuBatchMppi::new(n_drones, n_samples, horizon, &params, weights, &arena)
        .expect("pipeline construction")
}

#[test]
fn test_compute_batch_actions_returns_correct_shape() {
    if !gpu_available_or_skip("test_compute_batch_actions_returns_correct_shape") {
        return;
    }
    let n_drones = 4usize;
    let n_samples = 64usize;
    let horizon = 10usize;
    let pipeline = make_pipeline(n_drones, n_samples, horizon);

    let states = vec![0.0f32; n_drones * 13];
    let enemies = vec![0.0f32; n_drones * 13];
    let mean_ctrls = vec![0.0f32; n_drones * horizon * 4];
    let noise = vec![0.0f32; n_drones * n_samples * horizon * 4];

    let out = pipeline.compute_batch_actions(&states, &enemies, &mean_ctrls, &noise);
    assert_eq!(out.len(), n_drones * horizon * 4);
    assert_eq!(out.len(), 160);
}

#[test]
fn test_compute_batch_actions_produces_finite_output() {
    if !gpu_available_or_skip("test_compute_batch_actions_produces_finite_output") {
        return;
    }
    let n_drones = 4usize;
    let n_samples = 64usize;
    let horizon = 10usize;
    let pipeline = make_pipeline(n_drones, n_samples, horizon);

    // States: drones at alternating corners. Identity quaternion is (0,0,0,1).
    // State layout per drone (13 floats): pos3, vel3, quat_xyzw4, angvel3.
    let mut states = vec![0.0f32; n_drones * 13];
    let positions = [
        [1.0f32, 1.0, 1.5],
        [9.0, 9.0, 1.5],
        [1.0, 9.0, 1.5],
        [9.0, 1.0, 1.5],
    ];
    for (d, pos) in positions.iter().enumerate().take(n_drones) {
        let base = d * 13;
        states[base] = pos[0];
        states[base + 1] = pos[1];
        states[base + 2] = pos[2];
        // vel = 0, quat = (0,0,0,1)
        states[base + 9] = 1.0; // quat w
                                // angvel = 0
    }

    // Enemies: pair (0,1) and (2,3).
    let mut enemies = vec![0.0f32; n_drones * 13];
    let pairs = [(0usize, 1usize), (1, 0), (2, 3), (3, 2)];
    for (d, enemy_d) in pairs.iter().take(n_drones) {
        let src = enemy_d * 13;
        let dst = d * 13;
        enemies[dst..dst + 13].copy_from_slice(&states[src..src + 13]);
    }

    // Hover thrust per motor = mass * g / 4.
    let hover_thrust = 0.027 * 9.81 / 4.0;
    let mean_ctrls = vec![hover_thrust; n_drones * horizon * 4];
    let noise = vec![0.0f32; n_drones * n_samples * horizon * 4];

    let out = pipeline.compute_batch_actions(&states, &enemies, &mean_ctrls, &noise);
    assert_eq!(out.len(), n_drones * horizon * 4);

    for (i, v) in out.iter().enumerate() {
        assert!(v.is_finite(), "output[{i}] = {v} not finite");
    }

    // With zero noise, every sample has identical cost -> uniform softmax
    // weights, so the softmax-weighted mean recovers the input mean_ctrls.
    for (i, (o, m)) in out.iter().zip(mean_ctrls.iter()).enumerate() {
        let diff = (o - m).abs();
        assert!(
            diff < 1e-3,
            "output[{i}] = {o}, expected ~{m} (diff {diff})"
        );
    }
}

#[test]
fn test_compute_batch_actions_panics_on_wrong_input_length() {
    if !gpu_available_or_skip("test_compute_batch_actions_panics_on_wrong_input_length") {
        return;
    }
    let n_drones = 4usize;
    let n_samples = 64usize;
    let horizon = 10usize;
    let pipeline = make_pipeline(n_drones, n_samples, horizon);

    let bad_states = vec![0.0f32; 7]; // wrong length
    let enemies = vec![0.0f32; n_drones * 13];
    let mean_ctrls = vec![0.0f32; n_drones * horizon * 4];
    let noise = vec![0.0f32; n_drones * n_samples * horizon * 4];

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline.compute_batch_actions(&bad_states, &enemies, &mean_ctrls, &noise);
    }));
    assert!(result.is_err(), "expected panic on wrong states length");
}

// ----- Pure struct-layout checks (no GPU needed) -----

#[test]
fn test_mppi_dims_size_is_32() {
    // MppiDims is a 32-byte uniform (7 scalars + 1 × f32 pad). WGSL
    // requires uniform-struct size to be a multiple of 16 B.
    assert_eq!(std::mem::size_of::<MppiDims>(), 32);
    assert_eq!(std::mem::size_of::<MppiDims>() % 16, 0);
}

#[test]
fn test_mppi_dims_has_temperature() {
    // `MppiDims::new` must initialise `temperature` to 10.0 to match
    // `configs/rules.toml [mppi] temperature`. Pure layout/default
    // check — no GPU needed.
    let dims = MppiDims::new(4, 64, 20, 10, 5, 0.001);
    assert_eq!(
        dims.temperature, 10.0,
        "MppiDims::new must default temperature to 10.0 (matches rules.toml)"
    );
}

#[test]
fn test_drone_params_gpu_layout() {
    assert_eq!(std::mem::size_of::<DroneParamsGpu>(), 48);
    // WGSL uniforms need 16-byte alignment; with explicit padding fields we
    // rely on std140-compatible offsets but the Rust struct itself aligns to
    // 4 (largest scalar). Sanity-check the size is a multiple of 16.
    assert_eq!(std::mem::size_of::<DroneParamsGpu>() % 16, 0);

    assert_eq!(std::mem::size_of::<CostWeightsGpu>(), 48);
    assert_eq!(std::mem::size_of::<CostWeightsGpu>() % 16, 0);

    assert_eq!(std::mem::size_of::<ObstacleGpu>(), 48);
    assert_eq!(std::mem::size_of::<ObstacleGpu>() % 16, 0);
}

// ----- GPU vs CPU-reference end-to-end parity -----
//
// Keystone validation: running the same inputs through both the GPU
// pipeline and a pure-f32 CPU reference (see
// `pipeline::compute_batch_actions_cpu_reference`) must produce numerically
// equivalent outputs (up to f32 accumulation noise). If this test passes,
// the entire GPU MPPI stack — RK4 dynamics, SDF, cost functions, softmax
// reduction, buffer layout, and dispatch — is trustworthy.
#[test]
fn test_gpu_matches_cpu_reference_parity() {
    if !gpu_available_or_skip("test_gpu_matches_cpu_reference_parity") {
        return;
    }

    // Small config for fast turnaround: 4 drones (2 battles), 32 samples,
    // horizon 10. All reductions / accumulations remain deterministic on a
    // given backend.
    let n_drones: usize = 4;
    let n_samples: usize = 32;
    let horizon: usize = 10;

    let params_f32 = DroneParamsF32::crazyflie();
    let arena_f32 = warehouse_arena_f32();

    // GPU cost weights must match the CPU cost weights value-for-value.
    // `default_weights()` sets hover = 0.0 (see its definition at top of
    // this file), and these five weights/safe-distance values.
    let gpu_weights = default_weights();
    let cost_weights = CostWeightsF32 {
        w_dist: gpu_weights.w_dist,
        w_face: gpu_weights.w_face,
        w_ctrl: gpu_weights.w_ctrl,
        w_obs: gpu_weights.w_obs,
        d_safe: gpu_weights.d_safe,
    };
    // The GPU cost kernels read `weights.hover` from the CostWeightsGpu
    // uniform and use that as the hover_thrust value. For parity the CPU
    // reference must pass the same numeric value.
    let hover_for_cost = gpu_weights.hover;

    // Seeded RNG for deterministic inputs.
    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = SmallRng::seed_from_u64(42);

    // Two-pass init to avoid a state-vs-enemy bootstrap aliasing issue:
    // first fill `states` for all drones, then fill `enemies` by copying
    // the paired drone's state. (Drone d's enemy is drone d XOR 1.)
    let mut states = vec![0.0f32; n_drones * 13];
    for d in 0..n_drones {
        let base = d * 13;
        // Inside arena (10x10x3), avoiding pillars at corners and center.
        states[base] = rng.gen_range(2.0f32..8.0);
        states[base + 1] = rng.gen_range(2.0f32..8.0);
        states[base + 2] = rng.gen_range(1.0f32..2.5);
        // Velocity (3..6) = 0; identity quaternion (xyzw) at 6..10.
        states[base + 9] = 1.0;
        // Angular velocity (10..13) = 0.
    }
    let mut enemies = vec![0.0f32; n_drones * 13];
    for d in 0..n_drones {
        let base = d * 13;
        let enemy_d = d ^ 1;
        let enemy_base = enemy_d * 13;
        enemies[base..base + 13].copy_from_slice(&states[enemy_base..enemy_base + 13]);
    }

    // Warm-start mean control: hover thrust everywhere. We use the
    // params-based hover (mass*g/4) for the warm-start, independent of
    // the hover value passed into the cost function — this is a realistic
    // MPPI setup.
    let hover_ctrl = params_f32.hover_thrust();
    let mean_ctrls = vec![hover_ctrl; n_drones * horizon * 4];

    // Noise: small normal perturbations. Keep std small so the clamp
    // branch rarely fires (would introduce extra f32 drift between GPU
    // and CPU implementations of `clamp`).
    let noise_std = 0.01f32;
    let noise_len = n_drones * n_samples * horizon * 4;
    let mut noise = Vec::with_capacity(noise_len);
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0f32, noise_std).unwrap();
    for _ in 0..noise_len {
        noise.push(normal.sample(&mut rng));
    }

    // Build the GPU pipeline. `GpuBatchMppi::new` initialises the dims
    // uniform with substeps=10, dt_sim=0.001, temperature=10.0 — we mirror
    // exactly those values into the `MppiDims` passed to the CPU reference.
    let pipeline = GpuBatchMppi::new(
        n_drones,
        n_samples,
        horizon,
        &params_f32,
        gpu_weights,
        &arena_f32,
    )
    .expect("pipeline construction");

    let dims = MppiDims::new(
        n_drones as u32,
        n_samples as u32,
        horizon as u32,
        10,
        arena_f32.obstacles.len() as u32,
        0.001,
    );
    // `MppiDims::new` defaults temperature to 10.0, matching the value
    // `GpuBatchMppi::new` uploaded into the uniform.
    assert_eq!(
        dims.temperature, 10.0,
        "CPU-reference dims.temperature must match GPU default"
    );

    // Run both implementations.
    let gpu_out = pipeline.compute_batch_actions(&states, &enemies, &mean_ctrls, &noise);
    let cpu_out = compute_batch_actions_cpu_reference(
        &params_f32,
        &arena_f32,
        &cost_weights,
        hover_for_cost,
        dims,
        &states,
        &enemies,
        &mean_ctrls,
        &noise,
    );

    assert_eq!(
        gpu_out.len(),
        cpu_out.len(),
        "GPU/CPU output lengths differ"
    );

    // Compare element-wise.
    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    for i in 0..gpu_out.len() {
        let d = (gpu_out[i] - cpu_out[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_diff_idx = i;
        }
    }

    // Tolerance: f32 accumulation over 10*10=100 RK4 steps per rollout,
    // then softmax reduction across 32 samples. Empirically prior parity
    // tests on the same-shaped pipelines land near 1e-3; 1e-2 gives a
    // safe margin for backend variation.
    let tol = 1e-2f32;
    if max_diff >= tol {
        // Print first 5 entries side-by-side to aid debugging.
        eprintln!("[parity] max abs diff = {max_diff:.6e} at idx {max_diff_idx}");
        for i in 0..5.min(gpu_out.len()) {
            eprintln!("  i={i}: gpu={:.6e} cpu={:.6e}", gpu_out[i], cpu_out[i]);
        }
    }
    assert!(
        max_diff < tol,
        "GPU-CPU parity failed: max abs diff = {max_diff:.6e} at idx {max_diff_idx}"
    );

    eprintln!("[parity] max abs diff = {max_diff:.6e}");
}

/// Extends `test_gpu_matches_cpu_reference_parity` by reading back the
/// per-sample intermediate buffers (costs and ctrls_out) from the GPU and
/// comparing them element-wise against the CPU reference.
///
/// This catches divergences that cancel in softmax — e.g. an off-by-one in
/// the noise index where two samples swap perturbations and produce the same
/// final weighted-mean action despite different intermediate traces.
#[test]
fn test_gpu_per_sample_parity() {
    if !gpu_available_or_skip("test_gpu_per_sample_parity") {
        return;
    }

    let n_drones: usize = 4;
    let n_samples: usize = 32;
    let horizon: usize = 10;

    let params_f32 = DroneParamsF32::crazyflie();
    let arena_f32 = warehouse_arena_f32();

    let gpu_weights = default_weights();
    let cost_weights = CostWeightsF32 {
        w_dist: gpu_weights.w_dist,
        w_face: gpu_weights.w_face,
        w_ctrl: gpu_weights.w_ctrl,
        w_obs: gpu_weights.w_obs,
        d_safe: gpu_weights.d_safe,
    };
    let hover_for_cost = gpu_weights.hover;

    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = SmallRng::seed_from_u64(77);

    let mut states = vec![0.0f32; n_drones * 13];
    for d in 0..n_drones {
        let base = d * 13;
        states[base] = rng.gen_range(2.0f32..8.0);
        states[base + 1] = rng.gen_range(2.0f32..8.0);
        states[base + 2] = rng.gen_range(1.0f32..2.5);
        states[base + 9] = 1.0;
    }
    let mut enemies = vec![0.0f32; n_drones * 13];
    for d in 0..n_drones {
        let base = d * 13;
        let eb = (d ^ 1) * 13;
        enemies[base..base + 13].copy_from_slice(&states[eb..eb + 13]);
    }

    let hover_ctrl = params_f32.hover_thrust();
    let mean_ctrls = vec![hover_ctrl; n_drones * horizon * 4];

    let noise_std = 0.01f32;
    let noise_len = n_drones * n_samples * horizon * 4;
    let mut noise = Vec::with_capacity(noise_len);
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0f32, noise_std).unwrap();
    for _ in 0..noise_len {
        noise.push(normal.sample(&mut rng));
    }

    let pipeline = GpuBatchMppi::new(
        n_drones,
        n_samples,
        horizon,
        &params_f32,
        gpu_weights,
        &arena_f32,
    )
    .expect("pipeline construction");

    let dims = MppiDims::new(
        n_drones as u32,
        n_samples as u32,
        horizon as u32,
        10,
        arena_f32.obstacles.len() as u32,
        0.001,
    );

    let (gpu_result, gpu_costs, gpu_ctrls): (Vec<f32>, Vec<f32>, Vec<f32>) =
        pipeline.compute_batch_actions_with_intermediates(&states, &enemies, &mean_ctrls, &noise);
    let (cpu_result, cpu_costs, cpu_ctrls): (Vec<f32>, Vec<f32>, Vec<f32>) =
        compute_batch_actions_cpu_reference_with_intermediates(
            &params_f32,
            &arena_f32,
            &cost_weights,
            hover_for_cost,
            dims,
            &states,
            &enemies,
            &mean_ctrls,
            &noise,
        );

    // Final reduced actions must agree (same tolerance as existing parity test).
    let final_max_diff: f32 = gpu_result
        .iter()
        .zip(cpu_result.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        final_max_diff < 1e-2,
        "per-sample test: final action parity failed max_diff={final_max_diff:.6e}"
    );

    // Per-sample costs: tolerance is permissive (costs can be large floats).
    assert_eq!(gpu_costs.len(), cpu_costs.len(), "costs length mismatch");
    let cost_max_diff: f32 = gpu_costs
        .iter()
        .zip(cpu_costs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[per-sample] cost max abs diff = {cost_max_diff:.6e}");
    assert!(
        cost_max_diff < 1.0,
        "GPU-CPU per-sample cost parity failed: max abs diff = {cost_max_diff:.6e}"
    );

    // Per-sample controls: motors are bounded [0, 0.15] so tight tolerance applies.
    assert_eq!(
        gpu_ctrls.len(),
        cpu_ctrls.len(),
        "ctrls_out length mismatch"
    );
    let ctrl_max_diff: f32 = gpu_ctrls
        .iter()
        .zip(cpu_ctrls.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("[per-sample] ctrls_out max abs diff = {ctrl_max_diff:.6e}");
    assert!(
        ctrl_max_diff < 1e-3,
        "GPU-CPU per-sample ctrls_out parity failed: max abs diff = {ctrl_max_diff:.6e}"
    );
}

#[test]
fn test_obstacle_gpu_conversion() {
    // Box
    let box_obs = ObstacleF32::Box {
        center: Vector3::new(1.0, 2.0, 3.0),
        half_extents: Vector3::new(0.5, 0.5, 1.5),
    };
    let gpu_box = ObstacleGpu::from_f32(&box_obs);
    assert_eq!(gpu_box.kind, 0);
    assert_eq!(gpu_box.center, [1.0, 2.0, 3.0]);
    assert_eq!(gpu_box.half_extents, [0.5, 0.5, 1.5]);

    // Sphere
    let sphere_obs = ObstacleF32::Sphere {
        center: Vector3::new(4.0, 5.0, 6.0),
        radius: 0.7,
    };
    let gpu_sphere = ObstacleGpu::from_f32(&sphere_obs);
    assert_eq!(gpu_sphere.kind, 1);
    assert_eq!(gpu_sphere.center, [4.0, 5.0, 6.0]);
    assert_eq!(gpu_sphere.param_a, 0.7);
    assert_eq!(gpu_sphere.param_b, 0.0);

    // Cylinder
    let cyl_obs = ObstacleF32::Cylinder {
        center: Vector3::new(7.0, 8.0, 9.0),
        radius: 0.3,
        height: 2.5,
    };
    let gpu_cyl = ObstacleGpu::from_f32(&cyl_obs);
    assert_eq!(gpu_cyl.kind, 2);
    assert_eq!(gpu_cyl.center, [7.0, 8.0, 9.0]);
    assert_eq!(gpu_cyl.param_a, 0.3);
    assert_eq!(gpu_cyl.param_b, 2.5);
}
