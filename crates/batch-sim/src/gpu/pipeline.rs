//! GPU MPPI pipeline skeleton.
//!
//! Builds the wgpu device/queue and allocates all storage + uniform buffers
//! needed for Phase 2 GPU MPPI trajectory optimization. Static uniforms
//! (drone params, cost weights + arena bounds, obstacle list) are uploaded at
//! construction.
//!
//! This file is a skeleton — no bind group layout, no compute shader, no
//! dispatch. Those land in a follow-up subtask. Downstream shader integration
//! will wrap these buffers; public fields are intentional here.
//!
//! # Binding layout (per the parallel-simulation plan)
//!
//! | Binding | Kind     | Shape                                | Usage       |
//! |--------:|----------|--------------------------------------|-------------|
//! | 0       | storage  | states[N_DRONES × 13]                | read        |
//! | 1       | storage  | enemies[N_DRONES × 13]               | read        |
//! | 2       | storage  | mean_ctrls[N_DRONES × H × 4]         | read_write  |
//! | 3       | storage  | noise[N_DRONES × N × H × 4]          | read        |
//! | 4       | storage  | costs[N_DRONES × N]                  | read_write  |
//! | 5       | storage  | ctrls_out[N_DRONES × N × H × 4]      | read_write  |
//! | 6       | storage  | result[N_DRONES × H × 4]             | write       |
//! | 7       | uniform  | DroneParamsGpu                       | read        |
//! | 8       | uniform  | CostWeightsGpu + arena bounds        | read        |
//! | 9       | storage  | obstacles[MAX_OBSTACLES]             | read        |
//!
//! `N_DRONES = n_battles * 2`, `N = n_samples`, `H = horizon`. All buffers use
//! f32 (u32 for obstacle kind tag).

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

/// GPU-resident MPPI batch pipeline — buffers only (no shader yet).
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
        })
    }
}
