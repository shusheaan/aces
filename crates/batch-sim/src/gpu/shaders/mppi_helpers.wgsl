// MPPI helper functions (WGSL).
//
// This shader contains **helper functions only** — no @compute entry points.
// Entry points (rollout / cost / weighted-sum kernels) land in a later slice
// of the Phase 2 GPU MPPI port. This slice exists so the Rust-side integration
// can be validated structurally (via naga) without GPU access.
//
// Struct byte-layouts mirror the Rust POD types in
// `crates/batch-sim/src/gpu/pipeline.rs` exactly (DroneParamsGpu,
// CostWeightsGpu, ObstacleGpu — 48 bytes each). Numeric logic mirrors
// `crates/batch-sim/src/f32_dynamics.rs` and `crates/batch-sim/src/f32_sdf.rs`.
//
// Quaternion convention: `vec4<f32>(x, y, z, w)` where `w` is the scalar
// component. This matches nalgebra's `Quaternion::coords` layout used by the
// f32 CPU reference. `quat_mul` below is the standard Hamilton product written
// out for this convention.
//
// MAX_OBSTACLES is fixed at 32 to match `pipeline.rs::MAX_OBSTACLES`. Using a
// value-typed `array<Obstacle, 32>` parameter for `arena_sdf` avoids needing
// a storage binding declaration in this helpers-only slice; the eventual
// kernel will bind the obstacle buffer directly.

const FRAC_1_SQRT_2: f32 = 0.70710678118654752;
const MAX_OBSTACLES: u32 = 32u;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

// 48 bytes. Layout matches `DroneParamsGpu` in pipeline.rs.
//   6 scalars (24B) + 8B pad + vec3 inertia (12B) + 4B tail pad.
struct DroneParams {
    mass: f32,
    arm_length: f32,
    torque_coeff: f32,
    drag_coeff: f32,
    gravity: f32,
    max_thrust: f32,
    _pad0: f32,
    _pad1: f32,
    inertia: vec3<f32>,
    _pad_tail: f32,
}

// 48 bytes. Layout matches `CostWeightsGpu` in pipeline.rs.
//   6 scalars (24B) + fov_half (4B) + 1 pad (4B) + vec3 arena_bounds (12B) + 4B tail pad.
//
// `fov_half` is the half-angle of the lock-on cone in radians, read from
// `configs/rules.toml [lockon] fov_degrees / 2`. Previously hardcoded as
// PI/4 (45 deg) in `evasion_cost_gpu`; now plumbed via this uniform.
struct CostWeights {
    w_dist: f32,
    w_face: f32,
    w_ctrl: f32,
    w_obs: f32,
    d_safe: f32,
    hover: f32,
    fov_half: f32,
    _pad1: f32,
    arena_bounds: vec3<f32>,
    _pad_tail: f32,
}

// 48 bytes. Layout matches `ObstacleGpu` in pipeline.rs.
//   kind (u32) + 3×u32 pad + vec3 center (12B) + param_a (4B) +
//   vec3 half_extents (12B) + param_b (4B).
//
// kind: 0 = Box, 1 = Sphere, 2 = Cylinder.
struct Obstacle {
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    center: vec3<f32>,
    param_a: f32,       // radius for sphere/cylinder; unused for box
    half_extents: vec3<f32>,
    param_b: f32,       // height for cylinder; unused for sphere/box
}

// Local struct for passing state around inside the shader. NOT meant to match
// a Rust POD — the actual states buffer (binding 0/1) is a flat
// `array<f32, 13>` per drone, and kernels will pack/unpack from that layout.
// Attitude is (x, y, z, w).
struct DroneState {
    position: vec3<f32>,
    velocity: vec3<f32>,
    attitude: vec4<f32>,
    angular_velocity: vec3<f32>,
}

// Return packet for `state_derivative` — matches the four outputs of the f32
// reference (p_dot, v_dot, q_dot, w_dot).
struct StateDerivative {
    p_dot: vec3<f32>,
    v_dot: vec3<f32>,
    q_dot: vec4<f32>,
    w_dot: vec3<f32>,
}

// ---------------------------------------------------------------------------
// Quaternion helpers — convention: q = (x, y, z, w), w is scalar part.
// ---------------------------------------------------------------------------

// Hamilton product of two quaternions. Written out for (x, y, z, w) layout.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let ax = a.x; let ay = a.y; let az = a.z; let aw = a.w;
    let bx = b.x; let by = b.y; let bz = b.z; let bw = b.w;
    return vec4<f32>(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    );
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let n = length(q);
    // Guard against division by zero — mirrors UnitQuaternion::from_quaternion's
    // behaviour (which normalizes; if norm is 0 it would blow up on CPU too).
    if (n < 1e-20) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return q / n;
}

// Rotate a vector by a quaternion (body -> world).
// Standard Rodrigues-form: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w*v).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    let t = cross(u, v) + s * v;
    return v + 2.0 * cross(u, t);
}

// ---------------------------------------------------------------------------
// Dynamics
// ---------------------------------------------------------------------------

// X-configuration motor mixing. Returns (total_thrust, tau_x, tau_y, tau_z).
fn motor_mixing(motors: vec4<f32>, params: DroneParams) -> vec4<f32> {
    let d = params.arm_length;
    let c = params.torque_coeff;
    let s = FRAC_1_SQRT_2;

    let total_thrust = motors.x + motors.y + motors.z + motors.w;
    let tau_x = d * s * (motors.x - motors.y - motors.z + motors.w);
    let tau_y = d * s * (motors.x + motors.y - motors.z - motors.w);
    let tau_z = c * (motors.x - motors.y + motors.z - motors.w);

    return vec4<f32>(total_thrust, tau_x, tau_y, tau_z);
}

// Compute state derivative. Mirrors `state_derivative_f32` in f32_dynamics.rs.
fn state_derivative(
    state: DroneState,
    motors: vec4<f32>,
    params: DroneParams,
    external_force: vec3<f32>,
) -> StateDerivative {
    let mix = motor_mixing(motors, params);
    let total_thrust = mix.x;
    let torque = vec3<f32>(mix.y, mix.z, mix.w);

    // Position derivative = velocity
    let p_dot = state.velocity;

    // Velocity derivative (world frame)
    let thrust_body = vec3<f32>(0.0, 0.0, total_thrust);
    let thrust_world = quat_rotate(state.attitude, thrust_body);
    let gravity = vec3<f32>(0.0, 0.0, -params.gravity * params.mass);
    let drag = -params.drag_coeff * state.velocity;
    let v_dot = (thrust_world + gravity + drag + external_force) / params.mass;

    // Quaternion derivative: q_dot = 0.5 * q * [w_x, w_y, w_z, 0]
    // (omega as a pure-vector quaternion with scalar part 0 — (x,y,z,w) layout).
    let w = state.angular_velocity;
    let omega_quat = vec4<f32>(w.x, w.y, w.z, 0.0);
    let q_dot = 0.5 * quat_mul(state.attitude, omega_quat);

    // Angular velocity derivative: I * w_dot = tau - w x (I * w).
    let iw = vec3<f32>(
        params.inertia.x * w.x,
        params.inertia.y * w.y,
        params.inertia.z * w.z,
    );
    let gyro = cross(w, iw);
    let w_dot = vec3<f32>(
        (torque.x - gyro.x) / params.inertia.x,
        (torque.y - gyro.y) / params.inertia.y,
        (torque.z - gyro.z) / params.inertia.z,
    );

    return StateDerivative(p_dot, v_dot, q_dot, w_dot);
}

// Integrate quaternion: q_new = normalize(q + q_dot * dt).
fn integrate_quaternion(q: vec4<f32>, q_dot: vec4<f32>, dt: f32) -> vec4<f32> {
    return quat_normalize(q + q_dot * dt);
}

// RK4 step. Mirrors `step_rk4_f32` in f32_dynamics.rs. Motors are clamped at
// the start of the step. Quaternion is integrated via the average of the four
// stage derivatives and renormalized.
fn rk4_step(
    state: DroneState,
    motors: vec4<f32>,
    params: DroneParams,
    dt: f32,
    external_force: vec3<f32>,
) -> DroneState {
    let motors_clamped = vec4<f32>(
        clamp(motors.x, 0.0, params.max_thrust),
        clamp(motors.y, 0.0, params.max_thrust),
        clamp(motors.z, 0.0, params.max_thrust),
        clamp(motors.w, 0.0, params.max_thrust),
    );

    let k1 = state_derivative(state, motors_clamped, params, external_force);

    var s2: DroneState;
    s2.position = state.position + k1.p_dot * dt * 0.5;
    s2.velocity = state.velocity + k1.v_dot * dt * 0.5;
    s2.attitude = integrate_quaternion(state.attitude, k1.q_dot, dt * 0.5);
    s2.angular_velocity = state.angular_velocity + k1.w_dot * dt * 0.5;
    let k2 = state_derivative(s2, motors_clamped, params, external_force);

    var s3: DroneState;
    s3.position = state.position + k2.p_dot * dt * 0.5;
    s3.velocity = state.velocity + k2.v_dot * dt * 0.5;
    s3.attitude = integrate_quaternion(state.attitude, k2.q_dot, dt * 0.5);
    s3.angular_velocity = state.angular_velocity + k2.w_dot * dt * 0.5;
    let k3 = state_derivative(s3, motors_clamped, params, external_force);

    var s4: DroneState;
    s4.position = state.position + k3.p_dot * dt;
    s4.velocity = state.velocity + k3.v_dot * dt;
    s4.attitude = integrate_quaternion(state.attitude, k3.q_dot, dt);
    s4.angular_velocity = state.angular_velocity + k3.w_dot * dt;
    let k4 = state_derivative(s4, motors_clamped, params, external_force);

    let new_pos = state.position
        + (k1.p_dot + 2.0 * k2.p_dot + 2.0 * k3.p_dot + k4.p_dot) * dt / 6.0;
    let new_vel = state.velocity
        + (k1.v_dot + 2.0 * k2.v_dot + 2.0 * k3.v_dot + k4.v_dot) * dt / 6.0;
    let new_w = state.angular_velocity
        + (k1.w_dot + 2.0 * k2.w_dot + 2.0 * k3.w_dot + k4.w_dot) * dt / 6.0;

    // Average quaternion derivative, integrate + renormalize.
    let avg_q_dot = (k1.q_dot + 2.0 * k2.q_dot + 2.0 * k3.q_dot + k4.q_dot) * (1.0 / 6.0);
    let new_attitude = integrate_quaternion(state.attitude, avg_q_dot, dt);

    var result: DroneState;
    result.position = new_pos;
    result.velocity = new_vel;
    result.attitude = new_attitude;
    result.angular_velocity = new_w;
    return result;
}

// ---------------------------------------------------------------------------
// SDF helpers — mirror `f32_sdf.rs`.
// ---------------------------------------------------------------------------

fn box_sdf(p: vec3<f32>, center: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let d = abs(p - center) - half_extents;
    let outside = length(max(d, vec3<f32>(0.0, 0.0, 0.0)));
    let inside = min(max(d.x, max(d.y, d.z)), 0.0);
    return outside + inside;
}

fn sphere_sdf(p: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    return length(p - center) - radius;
}

// Cylinder with flat-bottom center at `center`, axis along Z, matching the
// CPU reference (`f32_sdf.rs`): radial distance − radius, axial distance uses
// the half-height trick centered at center.z + height/2.
fn cylinder_sdf(p: vec3<f32>, center: vec3<f32>, radius: f32, height: f32) -> f32 {
    let dx = sqrt((p.x - center.x) * (p.x - center.x)
                + (p.y - center.y) * (p.y - center.y)) - radius;
    let dz = abs(p.z - center.z - height * 0.5) - height * 0.5;
    let outside = length(vec2<f32>(max(dx, 0.0), max(dz, 0.0)));
    let inside = min(max(dx, dz), 0.0);
    return outside + inside;
}

// Dispatch on obstacle kind.
fn obstacle_sdf(p: vec3<f32>, obs: Obstacle) -> f32 {
    // 0 = Box, 1 = Sphere, 2 = Cylinder.
    if (obs.kind == 0u) {
        return box_sdf(p, obs.center, obs.half_extents);
    } else if (obs.kind == 1u) {
        return sphere_sdf(p, obs.center, obs.param_a);
    } else {
        return cylinder_sdf(p, obs.center, obs.param_a, obs.param_b);
    }
}

// Min of 6 axis-aligned walls. Negative means outside the arena.
fn boundary_sdf(p: vec3<f32>, arena_bounds: vec3<f32>) -> f32 {
    let dx_min = p.x;
    let dx_max = arena_bounds.x - p.x;
    let dy_min = p.y;
    let dy_max = arena_bounds.y - p.y;
    let dz_min = p.z;
    let dz_max = arena_bounds.z - p.z;
    return min(min(min(dx_min, dx_max), min(dy_min, dy_max)), min(dz_min, dz_max));
}

// Combined arena SDF: min of boundary distance and min over all obstacles.
// Obstacles are passed as a fixed-size array to avoid needing a storage
// binding in this helpers-only slice (see file-level comment).
fn arena_sdf(
    p: vec3<f32>,
    arena_bounds: vec3<f32>,
    obstacles: array<Obstacle, 32>,
    n_obstacles: u32,
) -> f32 {
    var d = boundary_sdf(p, arena_bounds);
    // Local var copy so we can index it (array function params are immutable).
    var obs_local = obstacles;
    for (var i: u32 = 0u; i < n_obstacles; i = i + 1u) {
        let d_obs = obstacle_sdf(p, obs_local[i]);
        d = min(d, d_obs);
    }
    return d;
}
