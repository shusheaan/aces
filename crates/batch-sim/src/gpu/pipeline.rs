//! GPU MPPI pipeline.
//!
//! Builds the wgpu device/queue, allocates all storage + uniform buffers, and
//! constructs the bind group layout + compiled shader module + two compute
//! pipelines (`rollout_and_cost`, `softmax_reduce`) needed for Phase 2 GPU
//! MPPI trajectory optimization. Static uniforms (drone params, cost weights +
//! arena bounds, obstacle list) are uploaded at construction.
//!
//! Dispatch (submitting work to the GPU) is not yet wired — that is a
//! follow-up subtask.
//!
//! # Binding layout (per the parallel-simulation plan)
//!
//! | Binding | Kind     | Shape                                | Usage       |
//! |--------:|----------|--------------------------------------|-------------|
//! | 0       | storage  | states[N_DRONES × 13]                | read        |
//! | 1       | storage  | enemies[N_DRONES × 13]               | read        |
//! | 2       | storage  | mean_ctrls[N_DRONES × H × 4]         | read (write by future warm-start kernel) |
//! | 3       | storage  | noise[N_DRONES × N × H × 4]          | read        |
//! | 4       | storage  | costs[N_DRONES × N]                  | read_write  |
//! | 5       | storage  | ctrls_out[N_DRONES × N × H × 4]      | read_write  |
//! | 6       | storage  | result[N_DRONES × H × 4]             | read_write  |
//! | 7       | uniform  | DroneParamsGpu                       | read        |
//! | 8       | uniform  | CostWeightsGpu + arena bounds        | read        |
//! | 9       | storage  | obstacles[MAX_OBSTACLES]             | read        |
//! | 10      | uniform  | MppiDims                             | read        |
//!
//! `N_DRONES = n_battles * 2`, `N = n_samples`, `H = horizon`. All buffers use
//! f32 (u32 for obstacle kind tag). `MppiDims` is a 32-byte uniform carrying
//! the runtime-configurable rollout dimensions (`n_drones`, `n_samples`,
//! `horizon`, `substeps`, `n_obstacles`, `dt_sim`).

use bytemuck::{Pod, Zeroable};

use crate::f32_dynamics::DroneParamsF32;
use crate::f32_sdf::{ArenaF32, ObstacleF32};

/// Maximum number of obstacles that can be uploaded to the GPU obstacle buffer.
///
/// WGSL compute shaders prefer fixed-size arrays over dynamically-sized arrays
/// for performance; 32 covers the warehouse arena (5 pillars) and any
/// reasonable test scenario with headroom. Bumping this requires no shader
/// changes — just more VRAM for an unused tail.
pub const MAX_OBSTACLES: usize = 32;

/// Dimensionality of the 13-element drone state (pos3 + vel3 + quat4 + angvel3).
const STATE_DIM: usize = 13;

/// Drone physical parameters, GPU-compatible layout.
///
/// `repr(C)` so it can be bound as a WGSL `uniform` struct. Exactly 48 bytes.
/// Layout follows WGSL std140-like rules: six scalars (24 bytes), 8-byte pad
/// so the subsequent `vec3<f32>` inertia lands on a 16-byte boundary, then
/// inertia (12 bytes) + trailing pad (4 bytes) to keep struct size a multiple
/// of 16. All padding fields are explicit so `bytemuck::Pod` accepts the type.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct DroneParamsGpu {
    pub mass: f32,
    pub arm_length: f32,
    pub torque_coeff: f32,
    pub drag_coeff: f32,
    pub gravity: f32,
    pub max_thrust: f32,
    pub _pad_before_inertia: [f32; 2],
    pub inertia: [f32; 3],
    pub _pad_tail: f32,
}

impl DroneParamsGpu {
    pub fn from_f32(p: &DroneParamsF32) -> Self {
        Self {
            mass: p.mass,
            arm_length: p.arm_length,
            torque_coeff: p.torque_coeff,
            drag_coeff: p.drag_coeff,
            gravity: p.gravity,
            max_thrust: p.max_thrust,
            _pad_before_inertia: [0.0; 2],
            inertia: [p.inertia.x, p.inertia.y, p.inertia.z],
            _pad_tail: 0.0,
        }
    }
}

/// MPPI cost weights and arena bounds, GPU-compatible layout.
///
/// Combined into one uniform block to match binding 8 in the plan layout.
/// Exactly 48 bytes. Six scalars (24 bytes) + 8-byte pad so the following
/// `vec3<f32>` arena_bounds lands on a 16-byte boundary, then arena_bounds
/// (12 bytes) + trailing pad (4 bytes). All padding is explicit for `Pod`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct CostWeightsGpu {
    pub w_dist: f32,
    pub w_face: f32,
    pub w_ctrl: f32,
    pub w_obs: f32,
    pub d_safe: f32,
    pub hover: f32,
    pub _pad_before_bounds: [f32; 2],
    pub arena_bounds: [f32; 3],
    pub _pad_tail: f32,
}

impl CostWeightsGpu {
    pub fn new(
        w_dist: f32,
        w_face: f32,
        w_ctrl: f32,
        w_obs: f32,
        d_safe: f32,
        hover: f32,
        arena_bounds: [f32; 3],
    ) -> Self {
        Self {
            w_dist,
            w_face,
            w_ctrl,
            w_obs,
            d_safe,
            hover,
            _pad_before_bounds: [0.0; 2],
            arena_bounds,
            _pad_tail: 0.0,
        }
    }
}

/// Tagged-union obstacle, GPU-compatible layout.
///
/// `kind` tag:
/// - 0 → Box: uses `center` + `half_extents`
/// - 1 → Sphere: uses `center` + `param_a` (radius)
/// - 2 → Cylinder: uses `center` + `param_a` (radius) + `param_b` (height)
///
/// Exactly 48 bytes. `_pad0: [u32; 3]` after `kind` keeps the subsequent
/// `center: vec3<f32>` 16-byte aligned in WGSL. All padding is explicit so
/// `bytemuck::Pod` accepts the type.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ObstacleGpu {
    pub kind: u32,
    pub _pad0: [u32; 3],
    pub center: [f32; 3],
    pub param_a: f32,
    pub half_extents: [f32; 3],
    pub param_b: f32,
}

/// Runtime MPPI dimensions, GPU-compatible layout (binding 10 uniform).
///
/// The `rollout_and_cost` kernel uses these to index the states / noise /
/// cost buffers and to control the RK4 sub-step loop. Kept separate from
/// [`DroneParamsGpu`] so callers can retune `substeps` / `dt_sim` at
/// runtime without re-uploading physical parameters.
///
/// The `temperature` field (lambda in the CPU code) parameterises the
/// softmax weighting applied by the `softmax_reduce` kernel:
/// `w[k] ∝ exp(-(c[k] - c_min) / T)`. Defaults to `10.0` to match
/// `configs/rules.toml [mppi] temperature`.
///
/// Exactly 32 bytes: seven scalars (28 B) plus one trailing f32 pad to
/// keep the size a multiple of 16 (WGSL uniform alignment rule). The
/// padding field is explicit so `bytemuck::Pod` accepts the type.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MppiDims {
    pub n_drones: u32,
    pub n_samples: u32,
    pub horizon: u32,
    pub substeps: u32,
    pub n_obstacles: u32,
    pub dt_sim: f32,
    /// MPPI softmax temperature (lambda). Matches `configs/rules.toml
    /// [mppi] temperature` — default `10.0`.
    pub temperature: f32,
    pub _pad: f32,
}

impl MppiDims {
    /// Construct `MppiDims` from high-level rollout parameters. Defaults
    /// used by [`GpuBatchMppi::new`]:
    ///   * `substeps    = 10` (10 × 0.001 s = 10 ms control tick)
    ///   * `dt_sim      = 0.001` s (1 kHz physics)
    ///   * `temperature = 10.0` (MPPI lambda; matches rules.toml)
    ///
    /// Callers that need to retune the physics step or softmax
    /// temperature can build an `MppiDims` directly and push it via
    /// [`GpuBatchMppi::update_dims`].
    pub fn new(
        n_drones: u32,
        n_samples: u32,
        horizon: u32,
        substeps: u32,
        n_obstacles: u32,
        dt_sim: f32,
    ) -> Self {
        Self {
            n_drones,
            n_samples,
            horizon,
            substeps,
            n_obstacles,
            dt_sim,
            temperature: 10.0,
            _pad: 0.0,
        }
    }
}

impl ObstacleGpu {
    pub fn from_f32(obs: &ObstacleF32) -> Self {
        match obs {
            ObstacleF32::Box {
                center,
                half_extents,
            } => Self {
                kind: 0,
                _pad0: [0; 3],
                center: [center.x, center.y, center.z],
                param_a: 0.0,
                half_extents: [half_extents.x, half_extents.y, half_extents.z],
                param_b: 0.0,
            },
            ObstacleF32::Sphere { center, radius } => Self {
                kind: 1,
                _pad0: [0; 3],
                center: [center.x, center.y, center.z],
                param_a: *radius,
                half_extents: [0.0; 3],
                param_b: 0.0,
            },
            ObstacleF32::Cylinder {
                center,
                radius,
                height,
            } => Self {
                kind: 2,
                _pad0: [0; 3],
                center: [center.x, center.y, center.z],
                param_a: *radius,
                half_extents: [0.0; 3],
                param_b: *height,
            },
        }
    }
}

/// Errors returned by `GpuBatchMppi::new`.
#[derive(Debug)]
pub enum GpuInitError {
    /// No GPU adapter could be acquired.
    NoAdapter,
    /// Device request to the adapter failed.
    DeviceRequestFailed(wgpu::RequestDeviceError),
    /// Arena has more obstacles than `MAX_OBSTACLES`. Contains the actual count.
    TooManyObstacles(usize),
}

impl std::fmt::Display for GpuInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuInitError::NoAdapter => write!(f, "no wgpu adapter available"),
            GpuInitError::DeviceRequestFailed(e) => write!(f, "wgpu device request failed: {e}"),
            GpuInitError::TooManyObstacles(n) => write!(
                f,
                "arena has {n} obstacles, but MAX_OBSTACLES is {MAX_OBSTACLES}"
            ),
        }
    }
}

impl std::error::Error for GpuInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuInitError::DeviceRequestFailed(e) => Some(e),
            _ => None,
        }
    }
}

/// Minimal blocking executor for wgpu futures. Mirrors the helper in
/// `adapter.rs`; kept local here to avoid the two modules sharing a private
/// util file just for 10 lines of code.
fn block_on<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn no_op(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

/// Build the 11 bind-group layout entries that match the WGSL shader's
/// `@group(0) @binding(0..10)` declarations.
///
/// This is a pure function — no wgpu device required — so tests can
/// introspect the entries without needing GPU access. The actual
/// `BindGroupLayout` is constructed inside [`GpuBatchMppi::new`] by
/// calling `device.create_bind_group_layout` with these entries.
///
/// # Layout
///
/// | Binding | Usage                         | Maps to buffer        |
/// |--------:|-------------------------------|-----------------------|
/// | 0       | `Storage { read_only: true }` | `states_buffer`       |
/// | 1       | `Storage { read_only: true }` | `enemies_buffer`      |
/// | 2       | `Storage { read_only: true }` | `mean_ctrls_buffer`   |
/// | 3       | `Storage { read_only: true }` | `noise_buffer`        |
/// | 4       | `Storage { read_only: false}` | `costs_buffer`        |
/// | 5       | `Storage { read_only: false}` | `ctrls_out_buffer`    |
/// | 6       | `Storage { read_only: false}` | `result_buffer`       |
/// | 7       | `Uniform`                     | `params_uniform`      |
/// | 8       | `Uniform`                     | `weights_uniform`     |
/// | 9       | `Storage { read_only: true }` | `obstacles_buffer`    |
/// | 10      | `Uniform`                     | `dims_uniform`        |
///
/// All entries have `visibility = ShaderStages::COMPUTE` and `count = None`.
/// Note on binding 2: the rollout kernel reads `mean_ctrls`; a future
/// warm-start kernel will write it. Marked `read_only: true` here to match
/// the current WGSL declaration in `mppi_rollout.wgsl`.
pub(crate) fn bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 11] {
    let storage_read = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let storage_rw = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let uniform = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };

    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };

    [
        entry(0, storage_read), // states
        entry(1, storage_read), // enemies
        entry(2, storage_read), // mean_ctrls (rollout reads; future warm-start kernel writes)
        entry(3, storage_read), // noise
        entry(4, storage_rw),   // costs
        entry(5, storage_rw),   // ctrls_out
        entry(6, storage_rw),   // result
        entry(7, uniform),      // params
        entry(8, uniform),      // weights
        entry(9, storage_read), // obstacles
        entry(10, uniform),     // dims
    ]
}

/// GPU-resident MPPI batch pipeline — buffers + bind group layout +
/// compiled shader module + two compute pipelines.
pub struct GpuBatchMppi {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    pub n_drones: usize,
    pub n_samples: usize,
    pub horizon: usize,
    pub n_obstacles: usize,

    // Storage buffers (per binding index)
    pub states_buffer: wgpu::Buffer,
    pub enemies_buffer: wgpu::Buffer,
    pub mean_ctrls_buffer: wgpu::Buffer,
    pub noise_buffer: wgpu::Buffer,
    pub costs_buffer: wgpu::Buffer,
    pub ctrls_out_buffer: wgpu::Buffer,
    pub result_buffer: wgpu::Buffer,

    // Uniform buffers
    pub params_uniform: wgpu::Buffer,
    pub weights_uniform: wgpu::Buffer,

    // Obstacle storage buffer (fixed MAX_OBSTACLES slots)
    pub obstacles_buffer: wgpu::Buffer,

    // Rollout dims uniform (binding 10). 32 bytes of `MppiDims`.
    pub dims_uniform: wgpu::Buffer,

    // Bind-group layout + compiled shader + two compute pipelines.
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub shader_module: wgpu::ShaderModule,
    pub rollout_pipeline: wgpu::ComputePipeline,
    pub softmax_pipeline: wgpu::ComputePipeline,
}

impl GpuBatchMppi {
    pub fn new(
        n_drones: usize,
        n_samples: usize,
        horizon: usize,
        params: &DroneParamsF32,
        weights: CostWeightsGpu,
        arena: &ArenaF32,
    ) -> Result<Self, GpuInitError> {
        let n_obstacles = arena.obstacles.len();
        if n_obstacles > MAX_OBSTACLES {
            return Err(GpuInitError::TooManyObstacles(n_obstacles));
        }

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(GpuInitError::NoAdapter)?;

        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("aces-gpu-batch-mppi"),
                ..Default::default()
            },
            None,
        ))
        .map_err(GpuInitError::DeviceRequestFailed)?;

        // Sizes in bytes. All element widths are 4 bytes (f32 or u32).
        const F: u64 = 4;
        let states_size = (n_drones * STATE_DIM) as u64 * F;
        let enemies_size = (n_drones * STATE_DIM) as u64 * F;
        let mean_ctrls_size = (n_drones * horizon * 4) as u64 * F;
        let noise_size = (n_drones * n_samples * horizon * 4) as u64 * F;
        let costs_size = (n_drones * n_samples) as u64 * F;
        let ctrls_out_size = (n_drones * n_samples * horizon * 4) as u64 * F;
        let result_size = (n_drones * horizon * 4) as u64 * F;
        let params_uniform_size = std::mem::size_of::<DroneParamsGpu>() as u64;
        let weights_uniform_size = std::mem::size_of::<CostWeightsGpu>() as u64;
        let obstacles_size = (MAX_OBSTACLES * std::mem::size_of::<ObstacleGpu>()) as u64;
        let dims_uniform_size = std::mem::size_of::<MppiDims>() as u64;

        let storage_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        // obstacles are uploaded once and only read by the shader — no COPY_SRC needed
        let obstacles_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

        let make_buf =
            |label: &'static str, size: u64, usage: wgpu::BufferUsages| -> wgpu::Buffer {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false,
                })
            };

        let states_buffer = make_buf("mppi.states", states_size, storage_usage);
        let enemies_buffer = make_buf("mppi.enemies", enemies_size, storage_usage);
        let mean_ctrls_buffer = make_buf("mppi.mean_ctrls", mean_ctrls_size, storage_usage);
        let noise_buffer = make_buf("mppi.noise", noise_size, storage_usage);
        let costs_buffer = make_buf("mppi.costs", costs_size, storage_usage);
        let ctrls_out_buffer = make_buf("mppi.ctrls_out", ctrls_out_size, storage_usage);
        let result_buffer = make_buf("mppi.result", result_size, storage_usage);

        let params_uniform = make_buf("mppi.params", params_uniform_size, uniform_usage);
        let weights_uniform = make_buf("mppi.weights", weights_uniform_size, uniform_usage);
        // obstacles are uploaded once and only read by the shader — no COPY_SRC needed
        let obstacles_buffer = make_buf("mppi.obstacles", obstacles_size, obstacles_usage);
        let dims_uniform = make_buf("mppi.dims", dims_uniform_size, uniform_usage);

        // Upload static uniforms
        let params_gpu = DroneParamsGpu::from_f32(params);
        queue.write_buffer(&params_uniform, 0, bytemuck::bytes_of(&params_gpu));
        queue.write_buffer(&weights_uniform, 0, bytemuck::bytes_of(&weights));

        // Upload obstacles padded with zero entries up to MAX_OBSTACLES.
        let mut obstacles_gpu: Vec<ObstacleGpu> = Vec::with_capacity(MAX_OBSTACLES);
        for obs in &arena.obstacles {
            obstacles_gpu.push(ObstacleGpu::from_f32(obs));
        }
        while obstacles_gpu.len() < MAX_OBSTACLES {
            obstacles_gpu.push(ObstacleGpu::zeroed());
        }
        queue.write_buffer(&obstacles_buffer, 0, bytemuck::cast_slice(&obstacles_gpu));

        // Upload default MppiDims. `substeps=10`, `dt_sim=0.001` gives a
        // 10 ms control tick at 1 kHz physics — matches the CPU pipeline
        // defaults. Callers can retune via `update_dims`.
        let dims = MppiDims::new(
            n_drones as u32,
            n_samples as u32,
            horizon as u32,
            10,
            n_obstacles as u32,
            0.001,
        );
        queue.write_buffer(&dims_uniform, 0, bytemuck::bytes_of(&dims));

        // -----------------------------------------------------------------
        // Pipeline construction: bind group layout, shader module, two
        // compute pipelines.
        //
        // `create_shader_module` and `create_compute_pipeline` do not
        // return `Result`s in wgpu 23 — they panic on invalid WGSL via the
        // device error callback. The full concatenated shader source is
        // naga-validated by `crate::gpu::shader::validate_full_mppi()` and
        // exercised by the `shader.rs::tests` suite, so a shader compile
        // failure here would indicate a test gap, not a production bug.
        // -----------------------------------------------------------------
        let entries = bind_group_layout_entries();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mppi.bind_group_layout"),
            entries: &entries,
        });

        let shader_source = crate::gpu::shader::full_mppi_source();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mppi.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Local; only needs to outlive the two pipeline creations below.
        // The pipelines hold their own internal references to it.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mppi.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let rollout_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mppi.rollout_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("rollout_and_cost"),
            compilation_options: Default::default(),
            cache: None,
        });

        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mppi.softmax_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("softmax_reduce"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            n_drones,
            n_samples,
            horizon,
            n_obstacles,
            states_buffer,
            enemies_buffer,
            mean_ctrls_buffer,
            noise_buffer,
            costs_buffer,
            ctrls_out_buffer,
            result_buffer,
            params_uniform,
            weights_uniform,
            obstacles_buffer,
            dims_uniform,
            bind_group_layout,
            shader_module,
            rollout_pipeline,
            softmax_pipeline,
        })
    }

    /// Overwrite the `MppiDims` uniform buffer with new values.
    ///
    /// Useful when the caller wants to change `substeps` / `dt_sim` at
    /// runtime without rebuilding the pipeline (e.g. switching between a
    /// 1 kHz and 2 kHz physics tick). The buffer write is queued on the
    /// internal queue; the new values take effect on the next compute
    /// dispatch.
    pub fn update_dims(&self, dims: MppiDims) {
        self.queue
            .write_buffer(&self.dims_uniform, 0, bytemuck::bytes_of(&dims));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL shader uses exactly 11 bindings (0..=10). If this count
    /// ever drifts from the shader, the pipeline will fail to bind.
    #[test]
    fn test_bind_group_layout_has_11_entries() {
        let entries = bind_group_layout_entries();
        assert_eq!(entries.len(), 11);
    }

    /// Bindings must be the contiguous range 0..=10 with no gaps or
    /// duplicates.
    #[test]
    fn test_bind_group_layout_bindings_are_0_through_10() {
        let entries = bind_group_layout_entries();
        let mut bindings: Vec<u32> = entries.iter().map(|e| e.binding).collect();
        bindings.sort();
        let expected: Vec<u32> = (0u32..=10).collect();
        assert_eq!(bindings, expected);
    }

    /// Every entry must match the usage pattern the shader expects:
    /// bindings 0,1,2,3,9 are read-only storage; 4,5,6 are read-write
    /// storage; 7,8,10 are uniforms. All stages COMPUTE, no `count`.
    #[test]
    fn test_bind_group_layout_usage_matches_buffers() {
        let entries = bind_group_layout_entries();

        // Shared sanity: all entries are COMPUTE visibility, no array count.
        for e in &entries {
            assert_eq!(
                e.visibility,
                wgpu::ShaderStages::COMPUTE,
                "binding {} must be COMPUTE-visible",
                e.binding,
            );
            assert!(
                e.count.is_none(),
                "binding {} must have count = None",
                e.binding,
            );
        }

        // Helper to classify `BindingType::Buffer` variants.
        fn is_storage_read(ty: &wgpu::BindingType) -> bool {
            matches!(
                ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    ..
                }
            )
        }
        fn is_storage_rw(ty: &wgpu::BindingType) -> bool {
            matches!(
                ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    ..
                }
            )
        }
        fn is_uniform(ty: &wgpu::BindingType) -> bool {
            matches!(
                ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    ..
                }
            )
        }

        let get = |binding: u32| -> &wgpu::BindingType {
            &entries
                .iter()
                .find(|e| e.binding == binding)
                .unwrap_or_else(|| panic!("binding {binding} missing"))
                .ty
        };

        for b in [0u32, 1, 2, 3, 9] {
            assert!(
                is_storage_read(get(b)),
                "binding {b} must be read-only storage, got {:?}",
                get(b),
            );
        }
        for b in [4u32, 5, 6] {
            assert!(
                is_storage_rw(get(b)),
                "binding {b} must be read-write storage, got {:?}",
                get(b),
            );
        }
        for b in [7u32, 8, 10] {
            assert!(
                is_uniform(get(b)),
                "binding {b} must be uniform, got {:?}",
                get(b),
            );
        }
    }
}
