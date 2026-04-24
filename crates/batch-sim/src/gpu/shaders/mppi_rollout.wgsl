// MPPI rollout + stage-cost kernel (WGSL).
//
// This file is concatenated at load time with `mppi_helpers.wgsl`
// (see `shader.rs::full_mppi_source`). The helpers file supplies the
// 12 named helper functions (`rk4_step`, `quat_rotate`, `arena_sdf`, …)
// plus the shared struct declarations (`DroneParams`, `CostWeights`,
// `Obstacle`, `DroneState`, `StateDerivative`). This file adds:
//
//   * `MppiDims` — runtime-configurable rollout dims (binding 10)
//   * All bind-group declarations (bindings 0..11)
//   * Thin wrappers (`pursuit_cost_gpu`, `evasion_cost_gpu`,
//     `arena_sdf_from_storage`) that adapt the helper signatures to the
//     storage-bound obstacle buffer / externally-computed sdf.
//   * The `rollout_and_cost` @compute entry point.
//
// # Dispatch convention
//
// One workgroup = one (drone_idx, sample_idx) pair. Invoke with
// `dispatch_workgroups(n_samples, n_drones, 1)` and `@workgroup_size(1)`.
// Inside the kernel we use `workgroup_id.x` as the sample index and
// `workgroup_id.y` as the drone index.
//
// # Drone pairing convention
//
// `n_drones = n_battles * 2`. Drones are laid out as
// `[battle0_A, battle0_B, battle1_A, battle1_B, …]`. Drone A (even
// idx) pursues; drone B (odd idx) evades. This mirrors
// `BatchOrchestrator` in `orchestrator.rs`:
//     `let motors_a = opt_a.compute_action(..., true);  // pursuit`
//     `let motors_b = opt_b.compute_action(..., false); // evasion`
//
// Any future dispatch code must keep this ordering to match CPU parity.
//
// # Wind
//
// Per-drone wind is uploaded via binding 11 (`wind_per_drone`) as a
// flat `array<f32>` with 4 floats per drone (vec3 + 1 pad slot for
// std140 alignment). The host writes it each tick via
// `GpuBatchMppi::set_wind`; when unset it is zero-initialized on
// construction, matching the previous hardcoded-zero behaviour. Wind
// is held constant across the rollout horizon — an approximation of
// the OU-process wind applied by the CPU physics step — but realistic
// for short horizons.

struct MppiDims {
    n_drones: u32,
    n_samples: u32,
    horizon: u32,
    substeps: u32,
    n_obstacles: u32,
    dt_sim: f32,
    // MPPI softmax temperature (lambda). Consumed by `softmax_reduce`
    // (see `mppi_softmax.wgsl`).
    temperature: f32,
    _pad: f32,
}

// ---------------------------------------------------------------------------
// Bindings — must match the buffer layout in `pipeline.rs`.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read>       states:      array<f32>;
@group(0) @binding(1) var<storage, read>       enemies:     array<f32>;
@group(0) @binding(2) var<storage, read>       mean_ctrls:  array<f32>;
@group(0) @binding(3) var<storage, read>       noise:       array<f32>;
@group(0) @binding(4) var<storage, read_write> costs:       array<f32>;
@group(0) @binding(5) var<storage, read_write> ctrls_out:   array<f32>;
@group(0) @binding(6) var<storage, read_write> result:      array<f32>;
@group(0) @binding(7) var<uniform>             params:      DroneParams;
@group(0) @binding(8) var<uniform>             weights:     CostWeights;
@group(0) @binding(9) var<storage, read>       obstacles:   array<Obstacle, 32>;
@group(0) @binding(10) var<uniform>            dims:        MppiDims;
// Per-drone wind force (world frame, Newtons). Layout: flat f32 array,
// drone `i` occupies indices `[i*4 .. i*4+3]` with `i*4+3` reserved as
// padding (vec4 alignment).
@group(0) @binding(11) var<storage, read>      wind_per_drone: array<f32>;

// ---------------------------------------------------------------------------
// State packing helpers.
// ---------------------------------------------------------------------------

// Note: `unpack_state` used to live here as a helper taking
// `ptr<storage, array<f32>, read>`, but naga 23 rejects storage
// pointers as function arguments (InvalidArgumentPointerSpace).
// The unpack logic is inlined at each call site in `rollout_and_cost`.

// Wrapper around helpers' `arena_sdf` that sources the obstacle array
// from binding 9 (a `storage, read` binding). WGSL forbids passing a
// storage-qualified array directly where a value-typed `array<Obstacle, 32>`
// is expected, so we copy into a function-scope local. The copy is once
// per stage cost, 32 obstacles × 48 bytes = 1.5 KiB — negligible vs. the
// four RK4 stages that dominate each step.
fn arena_sdf_from_storage(p: vec3<f32>) -> f32 {
    var obs_local: array<Obstacle, 32>;
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {
        obs_local[i] = obstacles[i];
    }
    return arena_sdf(p, weights.arena_bounds, obs_local, dims.n_obstacles);
}

// ---------------------------------------------------------------------------
// Stage cost wrappers — mirror f32_cost.rs exactly, but take a
// pre-computed SDF so the kernel can reuse it.
// ---------------------------------------------------------------------------

// Pursuit cost: distance² + facing + control smoothness + SDF margin.
fn pursuit_cost_gpu(
    self_state: DroneState,
    enemy_state: DroneState,
    ctrl: vec4<f32>,
    hover_thrust: f32,
    sdf_val: f32,
) -> f32 {
    var cost: f32 = 0.0;

    // Distance² term.
    let to_enemy = enemy_state.position - self_state.position;
    let dist = length(to_enemy);
    cost = cost + weights.w_dist * dist * dist;

    // Facing term: angle between forward() and direction to enemy.
    // forward = attitude * (+X_body). Mirrors DroneStateF32::angle_to:
    // when distance < 1e-6, return 0 (no meaningful direction).
    let forward = quat_rotate(self_state.attitude, vec3<f32>(1.0, 0.0, 0.0));
    var angle: f32 = 0.0;
    if (dist >= 1e-6) {
        let cos_angle = clamp(dot(forward, to_enemy) / dist, -1.0, 1.0);
        angle = acos(cos_angle);
    }
    cost = cost + weights.w_face * (1.0 - cos(angle));

    // Control smoothness: ||u - hover||².
    let ctrl_diff = ctrl - vec4<f32>(hover_thrust);
    cost = cost + weights.w_ctrl * dot(ctrl_diff, ctrl_diff);

    // Obstacle.
    if (sdf_val <= 0.0) {
        cost = cost + 1.0e6;
    } else if (sdf_val < weights.d_safe) {
        let margin = weights.d_safe - sdf_val;
        cost = cost + weights.w_obs * margin * margin;
    }

    return cost;
}

// Evasion cost: close-range penalty + in-enemy-FOV penalty + control
// smoothness + SDF margin. Mirrors evasion_cost_f32.
fn evasion_cost_gpu(
    self_state: DroneState,
    enemy_state: DroneState,
    ctrl: vec4<f32>,
    hover_thrust: f32,
    sdf_val: f32,
) -> f32 {
    var cost: f32 = 0.0;

    // Close-range penalty when dist < 3.0 m.
    let to_enemy = enemy_state.position - self_state.position;
    let dist = length(to_enemy);
    let safe_dist: f32 = 3.0;
    if (dist < safe_dist) {
        let margin = safe_dist - dist;
        cost = cost + weights.w_dist * margin * margin;
    }

    // In-enemy-FOV penalty: angle_from_enemy = enemy.forward() · (self - enemy).
    // Mirrors evasion_cost_f32: if dist < 1e-6, angle_to returns 0,
    // so the FOV check fires with cost = w_face * 1.0.
    let enemy_forward = quat_rotate(enemy_state.attitude, vec3<f32>(1.0, 0.0, 0.0));
    let to_self = self_state.position - enemy_state.position;
    let dist_es = length(to_self);
    var angle_from_enemy: f32 = 0.0;
    if (dist_es >= 1e-6) {
        let cos_e = clamp(dot(enemy_forward, to_self) / dist_es, -1.0, 1.0);
        angle_from_enemy = acos(cos_e);
    }
    let fov_half: f32 = 0.7853981633974483; // PI / 4
    if (angle_from_enemy < fov_half) {
        cost = cost + weights.w_face * (1.0 - angle_from_enemy / fov_half);
    }

    // Control smoothness.
    let ctrl_diff = ctrl - vec4<f32>(hover_thrust);
    cost = cost + weights.w_ctrl * dot(ctrl_diff, ctrl_diff);

    // Obstacle.
    if (sdf_val <= 0.0) {
        cost = cost + 1.0e6;
    } else if (sdf_val < weights.d_safe) {
        let margin = weights.d_safe - sdf_val;
        cost = cost + weights.w_obs * margin * margin;
    }

    return cost;
}

// ---------------------------------------------------------------------------
// Entry point: rollout + cost integration for one (drone, sample).
// ---------------------------------------------------------------------------

@compute @workgroup_size(1)
fn rollout_and_cost(@builtin(workgroup_id) wid: vec3<u32>) {
    let sample_idx = wid.x;
    let drone_idx = wid.y;
    if (sample_idx >= dims.n_samples) { return; }
    if (drone_idx >= dims.n_drones) { return; }

    // Load initial state + enemy state. Each drone occupies 13 contiguous
    // f32s in the states/enemies buffers. Unpack is inlined because naga
    // rejects storage-qualified pointer function arguments.
    let state_base = drone_idx * 13u;
    var state: DroneState;
    state.position = vec3<f32>(
        states[state_base + 0u], states[state_base + 1u], states[state_base + 2u],
    );
    state.velocity = vec3<f32>(
        states[state_base + 3u], states[state_base + 4u], states[state_base + 5u],
    );
    state.attitude = vec4<f32>(
        states[state_base + 6u], states[state_base + 7u],
        states[state_base + 8u], states[state_base + 9u],
    );
    state.angular_velocity = vec3<f32>(
        states[state_base + 10u], states[state_base + 11u], states[state_base + 12u],
    );

    var enemy: DroneState;
    enemy.position = vec3<f32>(
        enemies[state_base + 0u], enemies[state_base + 1u], enemies[state_base + 2u],
    );
    enemy.velocity = vec3<f32>(
        enemies[state_base + 3u], enemies[state_base + 4u], enemies[state_base + 5u],
    );
    enemy.attitude = vec4<f32>(
        enemies[state_base + 6u], enemies[state_base + 7u],
        enemies[state_base + 8u], enemies[state_base + 9u],
    );
    enemy.angular_velocity = vec3<f32>(
        enemies[state_base + 10u], enemies[state_base + 11u], enemies[state_base + 12u],
    );

    // Per-drone wind, uploaded by host before dispatch (see file header).
    let wind_base = drone_idx * 4u;
    let wind = vec3<f32>(
        wind_per_drone[wind_base + 0u],
        wind_per_drone[wind_base + 1u],
        wind_per_drone[wind_base + 2u],
    );

    let hover = weights.hover;
    var total_cost: f32 = 0.0;

    // Strides for the per-timestep control buffers.
    let mean_drone_stride = dims.horizon * 4u;
    let sample_drone_stride = dims.n_samples * dims.horizon * 4u;
    let sample_stride = dims.horizon * 4u;

    for (var h: u32 = 0u; h < dims.horizon; h = h + 1u) {
        // Compose perturbed control for this timestep.
        let mean_base = drone_idx * mean_drone_stride + h * 4u;
        let noise_base = drone_idx * sample_drone_stride
                       + sample_idx * sample_stride
                       + h * 4u;
        var u = vec4<f32>(
            mean_ctrls[mean_base + 0u] + noise[noise_base + 0u],
            mean_ctrls[mean_base + 1u] + noise[noise_base + 1u],
            mean_ctrls[mean_base + 2u] + noise[noise_base + 2u],
            mean_ctrls[mean_base + 3u] + noise[noise_base + 3u],
        );
        u = clamp(u, vec4<f32>(0.0), vec4<f32>(params.max_thrust));

        // Persist perturbed control for the downstream softmax-weighted
        // average. Uses the same layout as the noise buffer.
        let out_base = drone_idx * sample_drone_stride
                     + sample_idx * sample_stride
                     + h * 4u;
        ctrls_out[out_base + 0u] = u.x;
        ctrls_out[out_base + 1u] = u.y;
        ctrls_out[out_base + 2u] = u.z;
        ctrls_out[out_base + 3u] = u.w;

        // RK4 sub-stepping: substeps × dt_sim = dt_ctrl. Wind is constant.
        for (var s: u32 = 0u; s < dims.substeps; s = s + 1u) {
            state = rk4_step(state, u, params, dims.dt_sim, wind);
        }

        // Stage cost. Even drone_idx → pursuit; odd → evasion.
        let sdf_val = arena_sdf_from_storage(state.position);
        if ((drone_idx & 1u) == 0u) {
            total_cost = total_cost + pursuit_cost_gpu(state, enemy, u, hover, sdf_val);
        } else {
            total_cost = total_cost + evasion_cost_gpu(state, enemy, u, hover, sdf_val);
        }
    }

    costs[drone_idx * dims.n_samples + sample_idx] = total_cost;
}
