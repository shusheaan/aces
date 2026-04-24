//! Integration tests for the GPU MPPI pipeline skeleton.
//!
//! Feature-gated behind `gpu` — this file is empty unless the feature is
//! enabled. Tests that need a live GPU early-return on CI machines without
//! one (via `probe_gpu().compute_capable`). Pure struct-layout checks always
//! run.

#![cfg(feature = "gpu")]

use aces_batch_sim::f32_dynamics::DroneParamsF32;
use aces_batch_sim::f32_sdf::{ArenaF32, ObstacleF32};
use aces_batch_sim::gpu::adapter::probe_gpu;
use aces_batch_sim::gpu::pipeline::{
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
