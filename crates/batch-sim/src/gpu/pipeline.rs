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
//! | 11      | storage  | wind_per_drone[N_DRONES × 4]         | read        |
//!
//! `N_DRONES = n_battles * 2`, `N = n_samples`, `H = horizon`. All buffers use
//! f32 (u32 for obstacle kind tag). `MppiDims` is a 32-byte uniform carrying
//! the runtime-configurable rollout dimensions (`n_drones`, `n_samples`,
//! `horizon`, `substeps`, `n_obstacles`, `dt_sim`).

use bytemuck::{Pod, Zeroable};

use crate::f32_cost::{evasion_cost_f32, pursuit_cost_f32, CostWeightsF32};
use crate::f32_dynamics::{step_rk4_f32, DroneParamsF32, DroneStateF32};
use crate::f32_sdf::{ArenaF32, ObstacleF32};
use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};

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

/// Build the 12 bind-group layout entries that match the WGSL shader's
/// `@group(0) @binding(0..11)` declarations.
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
/// | 11      | `Storage { read_only: true }` | `wind_buffer`         |
///
/// All entries have `visibility = ShaderStages::COMPUTE` and `count = None`.
/// Note on binding 2: the rollout kernel reads `mean_ctrls`; a future
/// warm-start kernel will write it. Marked `read_only: true` here to match
/// the current WGSL declaration in `mppi_rollout.wgsl`.
pub(crate) fn bind_group_layout_entries() -> [wgpu::BindGroupLayoutEntry; 12] {
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
        entry(0, storage_read),  // states
        entry(1, storage_read),  // enemies
        entry(2, storage_read),  // mean_ctrls (rollout reads; future warm-start kernel writes)
        entry(3, storage_read),  // noise
        entry(4, storage_rw),    // costs
        entry(5, storage_rw),    // ctrls_out
        entry(6, storage_rw),    // result
        entry(7, uniform),       // params
        entry(8, uniform),       // weights
        entry(9, storage_read),  // obstacles
        entry(10, uniform),      // dims
        entry(11, storage_read), // wind per drone (vec3 padded to vec4)
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

    // Per-drone wind storage buffer (binding 11). `n_drones * 4` f32
    // (vec3 padded to vec4 for std140 alignment).
    pub wind_buffer: wgpu::Buffer,

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
        // Per-drone wind: 4 f32 per drone (vec3 + 1 pad for std140 alignment).
        let wind_size = (n_drones * 4) as u64 * F;

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
        // Wind buffer: host-updated each tick via `set_wind`. STORAGE + COPY_DST
        // — host-updated each tick via `set_wind`; no COPY_SRC needed (read-only from shader).
        let wind_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let wind_buffer = make_buf("mppi.wind", wind_size, wind_usage);

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

        // Initialize wind buffer to zeros. Callers must use `set_wind` to
        // upload actual per-drone wind forces before dispatch if wind is
        // desired in the rollout.
        let zero_wind = vec![0.0f32; n_drones * 4];
        queue.write_buffer(&wind_buffer, 0, bytemuck::cast_slice(&zero_wind));

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
            wind_buffer,
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

    /// Upload per-drone wind vectors. `winds` must be length `n_drones`,
    /// each element is a 3-component wind force in Newtons (world frame).
    /// Internally padded to `vec4` for std140 alignment (the 4th slot is
    /// unused).
    ///
    /// Wind is applied as a constant external force across the entire
    /// MPPI horizon for each drone — an approximation relative to the
    /// CPU OU-process wind (which resamples at every physics sub-step),
    /// but realistic for the short rollout horizons used in practice.
    ///
    /// # Panics
    /// Panics if `winds.len() != n_drones`.
    pub fn set_wind(&self, winds: &[[f32; 3]]) {
        assert_eq!(
            winds.len(),
            self.n_drones,
            "winds len {} != n_drones {}",
            winds.len(),
            self.n_drones,
        );
        let mut padded = vec![0.0f32; self.n_drones * 4];
        for (i, w) in winds.iter().enumerate() {
            padded[i * 4] = w[0];
            padded[i * 4 + 1] = w[1];
            padded[i * 4 + 2] = w[2];
            // padded[i*4 + 3] remains 0 (vec4 padding).
        }
        self.queue
            .write_buffer(&self.wind_buffer, 0, bytemuck::cast_slice(&padded));
    }

    /// Run one full MPPI iteration on the GPU: rollout + softmax-weighted mean.
    ///
    /// Inputs: current state per drone, enemy state per drone, warm-start mean
    /// control sequence per drone, pre-generated noise per (drone, sample,
    /// horizon).
    ///
    /// Returns the new optimal mean control sequence per drone, shape
    /// `[n_drones][horizon][4]` flattened row-major.
    ///
    /// Pipeline:
    ///   1. Upload `states`, `enemies`, `mean_ctrls`, `noise` to their
    ///      respective storage buffers.
    ///   2. Build a `BindGroup` for all 12 bindings.
    ///   3. Dispatch `rollout_pipeline` with `(n_samples, n_drones, 1)`
    ///      workgroups (matches kernel: `wid.x=sample`, `wid.y=drone`).
    ///   4. Dispatch `softmax_pipeline` with `(n_drones, 1, 1)` workgroups
    ///      (matches kernel: `wid.x=drone`).
    ///   5. Copy `result_buffer` into a staging buffer, submit, map-read.
    ///
    /// # Contract
    /// The quaternion portion of each drone state (`states[base+6..=9]` and
    /// `enemies[base+6..=9]`, xyzw layout) MUST be a unit quaternion. The GPU
    /// kernel does NOT renormalize on unpack; if input drift past unit norm,
    /// trajectory integration will drift and GPU/CPU parity will not hold.
    ///
    /// Wind is NOT set by this method — callers must use [`Self::set_wind`]
    /// separately if wind is desired in the rollout. On construction the
    /// wind buffer is initialized to zero, so an unset buffer behaves like
    /// the previous wind-hardcoded-to-zero rollout.
    ///
    /// # Panics
    /// Panics if the input slice lengths don't match the configured
    /// dimensions.
    pub fn compute_batch_actions(
        &self,
        states: &[f32],
        enemies: &[f32],
        mean_ctrls: &[f32],
        noise: &[f32],
    ) -> Vec<f32> {
        // -----------------------------------------------------------------
        // 1. Validate input lengths.
        // -----------------------------------------------------------------
        let expected_states = self.n_drones * STATE_DIM;
        let expected_enemies = self.n_drones * STATE_DIM;
        let expected_mean_ctrls = self.n_drones * self.horizon * 4;
        let expected_noise = self.n_drones * self.n_samples * self.horizon * 4;
        let expected_result = self.n_drones * self.horizon * 4;

        assert_eq!(
            states.len(),
            expected_states,
            "states len {} != expected n_drones * 13 = {}",
            states.len(),
            expected_states,
        );
        assert_eq!(
            enemies.len(),
            expected_enemies,
            "enemies len {} != expected n_drones * 13 = {}",
            enemies.len(),
            expected_enemies,
        );
        assert_eq!(
            mean_ctrls.len(),
            expected_mean_ctrls,
            "mean_ctrls len {} != expected n_drones * horizon * 4 = {}",
            mean_ctrls.len(),
            expected_mean_ctrls,
        );
        assert_eq!(
            noise.len(),
            expected_noise,
            "noise len {} != expected n_drones * n_samples * horizon * 4 = {}",
            noise.len(),
            expected_noise,
        );

        // -----------------------------------------------------------------
        // 2. Upload inputs.
        // -----------------------------------------------------------------
        self.queue
            .write_buffer(&self.states_buffer, 0, bytemuck::cast_slice(states));
        self.queue
            .write_buffer(&self.enemies_buffer, 0, bytemuck::cast_slice(enemies));
        self.queue
            .write_buffer(&self.mean_ctrls_buffer, 0, bytemuck::cast_slice(mean_ctrls));
        self.queue
            .write_buffer(&self.noise_buffer, 0, bytemuck::cast_slice(noise));

        // -----------------------------------------------------------------
        // 3. Create the BindGroup for this dispatch. Cheap — just a handle.
        // -----------------------------------------------------------------
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mppi.compute_batch_actions.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.states_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.enemies_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.mean_ctrls_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.noise_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.costs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.ctrls_out_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.params_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.weights_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.obstacles_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.dims_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.wind_buffer.as_entire_binding(),
                },
            ],
        });

        // -----------------------------------------------------------------
        // 4. Record the compute dispatches + result copy.
        // -----------------------------------------------------------------
        let result_size_bytes = (expected_result * std::mem::size_of::<f32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mppi.compute_batch_actions.staging"),
            size: result_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mppi.compute_batch_actions.encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mppi.rollout"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rollout_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Kernel convention: wid.x = sample, wid.y = drone.
            pass.dispatch_workgroups(self.n_samples as u32, self.n_drones as u32, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mppi.softmax"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.softmax_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // Kernel convention: wid.x = drone.
            pass.dispatch_workgroups(self.n_drones as u32, 1, 1);
        }

        // copy_buffer_to_buffer must be outside a compute pass.
        encoder.copy_buffer_to_buffer(&self.result_buffer, 0, &staging, 0, result_size_bytes);

        // -----------------------------------------------------------------
        // 5. Submit + map-read.
        // -----------------------------------------------------------------
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .expect("map_async channel closed")
            .expect("staging buffer map_async failed");

        let result: Vec<f32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        };
        staging.unmap();

        result
    }
}

/// Pure-CPU f32 reference implementation of the two GPU kernels
/// (`rollout_and_cost` + `softmax_reduce`).
///
/// Mirrors [`GpuBatchMppi::compute_batch_actions`] exactly in f32
/// arithmetic. Used as the keystone parity oracle for GPU validation:
/// if the GPU output and this CPU output agree to within f32 accumulation
/// tolerance, the GPU pipeline is trustworthy.
///
/// Semantics (per drone `d`):
///  1. Unpack the starting state from `states[d*13..]` and the enemy
///     state from `enemies[d*13..]` (the caller pre-pairs opponents —
///     same convention as the GPU kernel).
///  2. For each of `dims.n_samples` samples:
///      * Clone the starting state.
///      * For each horizon step `h`:
///         * `u = clamp(mean_ctrl[h] + noise[s, h], 0, max_thrust)`
///           component-wise. Store `u` in a per-sample ctrls buffer.
///         * Run `dims.substeps` RK4 sub-steps with wind = 0 (matches
///           the GPU kernel — wind is currently not wired on the GPU
///           path).
///         * Accumulate stage cost. Drone pairing: even `d` → pursuit,
///           odd `d` → evasion. Mirrors `mppi_rollout.wgsl`
///           `(drone_idx & 1u) == 0u`.
///      * Store total cost in `costs[s]`.
///  3. Softmax-weighted mean control per (horizon, motor):
///      * `min_cost = min(costs)`
///      * `total_w = Σ exp(-(costs[s] - min_cost) / T)`
///      * `inv_total = 1 / total_w` if `total_w > 0` else `0`
///      * `result[d][h][m] = Σ_s ctrls[s][h][m] * exp(...) * inv_total`
///
/// The hover-thrust value used inside the cost functions comes from
/// `weights_cost.hover_thrust_for_gpu_parity` — see below. The GPU
/// kernel reads `weights.hover` from the `CostWeightsGpu` uniform
/// (binding 8), so callers wanting parity must pass the same numeric
/// value here that was packed into `CostWeightsGpu::hover`.
///
/// # Panics
/// Panics if slice lengths don't match the configured dimensions.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn compute_batch_actions_cpu_reference(
    params: &DroneParamsF32,
    arena: &ArenaF32,
    weights_cost: &CostWeightsF32,
    hover_thrust: f32,
    dims: MppiDims,
    states: &[f32],
    enemies: &[f32],
    mean_ctrls: &[f32],
    noise: &[f32],
) -> Vec<f32> {
    let n_drones = dims.n_drones as usize;
    let n_samples = dims.n_samples as usize;
    let horizon = dims.horizon as usize;
    let substeps = dims.substeps as usize;
    let dt_sim = dims.dt_sim;
    let temperature = dims.temperature;

    assert_eq!(states.len(), n_drones * STATE_DIM);
    assert_eq!(enemies.len(), n_drones * STATE_DIM);
    assert_eq!(mean_ctrls.len(), n_drones * horizon * 4);
    assert_eq!(noise.len(), n_drones * n_samples * horizon * 4);

    // Unpack a 13-float state at offset `base` → DroneStateF32.
    // Layout: pos3 (0..3), vel3 (3..6), quat_xyzw (6..10), angvel3 (10..13).
    // `UnitQuaternion::from_quaternion` takes `Quaternion::new(w, x, y, z)`
    // (w-first), so we reorder explicitly.
    let unpack = |buf: &[f32], d: usize| -> DroneStateF32 {
        let base = d * STATE_DIM;
        DroneStateF32 {
            position: Vector3::new(buf[base], buf[base + 1], buf[base + 2]),
            velocity: Vector3::new(buf[base + 3], buf[base + 4], buf[base + 5]),
            attitude: UnitQuaternion::from_quaternion(Quaternion::new(
                buf[base + 9],
                buf[base + 6],
                buf[base + 7],
                buf[base + 8],
            )),
            angular_velocity: Vector3::new(buf[base + 10], buf[base + 11], buf[base + 12]),
        }
    };

    let mut result = vec![0.0f32; n_drones * horizon * 4];
    let wind = Vector3::<f32>::zeros();

    for d in 0..n_drones {
        let state0 = unpack(states, d);
        let enemy = unpack(enemies, d);
        let is_pursuit = d % 2 == 0;

        // Per-sample perturbed control buffer: [n_samples][horizon][4].
        let mut sim_ctrls = vec![0.0f32; n_samples * horizon * 4];
        let mut costs = vec![0.0f32; n_samples];

        let mean_drone_stride = horizon * 4;
        let noise_drone_stride = n_samples * horizon * 4;
        let sample_stride = horizon * 4;

        for s in 0..n_samples {
            let mut state_sim = state0.clone();
            let mut total_cost = 0.0f32;

            for h in 0..horizon {
                let mean_base = d * mean_drone_stride + h * 4;
                let noise_base = d * noise_drone_stride + s * sample_stride + h * 4;

                let u_raw = Vector4::new(
                    mean_ctrls[mean_base] + noise[noise_base],
                    mean_ctrls[mean_base + 1] + noise[noise_base + 1],
                    mean_ctrls[mean_base + 2] + noise[noise_base + 2],
                    mean_ctrls[mean_base + 3] + noise[noise_base + 3],
                );
                let u = Vector4::new(
                    u_raw[0].clamp(0.0, params.max_thrust),
                    u_raw[1].clamp(0.0, params.max_thrust),
                    u_raw[2].clamp(0.0, params.max_thrust),
                    u_raw[3].clamp(0.0, params.max_thrust),
                );

                // Persist perturbed control.
                let out_base = s * sample_stride + h * 4;
                sim_ctrls[out_base] = u[0];
                sim_ctrls[out_base + 1] = u[1];
                sim_ctrls[out_base + 2] = u[2];
                sim_ctrls[out_base + 3] = u[3];

                // RK4 sub-stepping (wind = 0 to match GPU kernel).
                for _ in 0..substeps {
                    state_sim = step_rk4_f32(&state_sim, &u, params, dt_sim, &wind);
                }

                // Stage cost.
                let stage_cost = if is_pursuit {
                    pursuit_cost_f32(&state_sim, &enemy, &u, hover_thrust, arena, weights_cost)
                } else {
                    evasion_cost_f32(&state_sim, &enemy, &u, hover_thrust, arena, weights_cost)
                };
                total_cost += stage_cost;
            }

            costs[s] = total_cost;
        }

        // -----------------------------------------------------------------
        // Softmax step — mirrors `softmax_reduce` kernel.
        // -----------------------------------------------------------------
        let min_cost = costs.iter().cloned().fold(f32::INFINITY, f32::min);
        let total_w: f32 = costs
            .iter()
            .map(|c| (-(c - min_cost) / temperature).exp())
            .sum();
        let inv_total = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };

        let result_drone_stride = horizon * 4;
        for h in 0..horizon {
            for m in 0..4 {
                let mut acc = 0.0f32;
                for s in 0..n_samples {
                    let w = (-(costs[s] - min_cost) / temperature).exp() * inv_total;
                    acc += sim_ctrls[s * sample_stride + h * 4 + m] * w;
                }
                result[d * result_drone_stride + h * 4 + m] = acc;
            }
        }
    }

    result
}

#[cfg(test)]
mod cpu_ref_tests {
    use super::*;

    /// With zero noise every sample has identical cost → uniform softmax
    /// weights → the weighted mean equals the shared per-sample ctrl,
    /// which (since noise is zero) equals the input `mean_ctrls`.
    ///
    /// This isolates CPU-reference correctness before comparing against
    /// the GPU, so a parity failure can be blamed on the GPU rather than
    /// on the reference.
    #[test]
    fn test_cpu_reference_zero_noise_returns_mean_ctrls() {
        let params = DroneParamsF32::crazyflie();
        let arena = ArenaF32::new(Vector3::new(10.0, 10.0, 3.0));
        let cost_weights = CostWeightsF32::default();
        let dims = MppiDims::new(2, 16, 5, 10, 0, 0.001);

        // Drone 0 at (0, 0, 1), drone 1 at (0, 0, 1). Identity quaternion.
        let mut states = vec![0.0f32; 2 * STATE_DIM];
        states[2] = 1.0; // drone 0 z
        states[9] = 1.0; // drone 0 quat w
        states[STATE_DIM + 2] = 1.0; // drone 1 z
        states[STATE_DIM + 9] = 1.0; // drone 1 quat w

        // Enemy: drone 0 sees enemy at (2, 0, 1); drone 1 sees (5, 0, 1).
        let mut enemies = vec![0.0f32; 2 * STATE_DIM];
        enemies[0] = 2.0;
        enemies[2] = 1.0;
        enemies[9] = 1.0;
        enemies[STATE_DIM] = 5.0;
        enemies[STATE_DIM + 2] = 1.0;
        enemies[STATE_DIM + 9] = 1.0;

        let hover = params.hover_thrust();
        let mean_ctrls = vec![hover; 2 * 5 * 4];
        let noise = vec![0.0f32; 2 * 16 * 5 * 4];

        let result = compute_batch_actions_cpu_reference(
            &params,
            &arena,
            &cost_weights,
            hover,
            dims,
            &states,
            &enemies,
            &mean_ctrls,
            &noise,
        );

        for (i, &r) in result.iter().enumerate() {
            assert!(r.is_finite(), "non-finite at index {i}: {r}");
            assert!(
                (r - hover).abs() < 1e-4,
                "result[{i}] = {r}, expected {hover}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WGSL shader uses exactly 12 bindings (0..=11). If this count
    /// ever drifts from the shader, the pipeline will fail to bind.
    #[test]
    fn test_bind_group_layout_has_12_entries() {
        let entries = bind_group_layout_entries();
        assert_eq!(entries.len(), 12);
    }

    /// Bindings must be the contiguous range 0..=11 with no gaps or
    /// duplicates.
    #[test]
    fn test_bind_group_layout_bindings_are_0_through_11() {
        let entries = bind_group_layout_entries();
        let mut bindings: Vec<u32> = entries.iter().map(|e| e.binding).collect();
        bindings.sort();
        let expected: Vec<u32> = (0u32..=11).collect();
        assert_eq!(bindings, expected);
    }

    /// Every entry must match the usage pattern the shader expects:
    /// bindings 0,1,2,3,9,11 are read-only storage; 4,5,6 are read-write
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

        for b in [0u32, 1, 2, 3, 9, 11] {
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
