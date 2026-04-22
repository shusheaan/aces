use bevy::prelude::*;

use aces_mppi::optimizer::MppiOptimizer;
use aces_sim_core::collision::{check_line_of_sight, Visibility};
use aces_sim_core::dynamics::{step_rk4, DroneParams};
use aces_sim_core::environment::Arena;
use aces_sim_core::lockon::LockOnTracker;
use aces_sim_core::recorder::{severity_to_u8, SimFrame, SimRecorder};
use aces_sim_core::safety::{SafetyEnvelope, SafetyStatus};
use aces_sim_core::state::DroneState;
use aces_sim_core::wind::WindModel;
use nalgebra::Vector4;
use rand::SeedableRng;

use crate::config::GameConfig;
use crate::input::DroneCommand;
use crate::policy::{self as policy, MlpPolicy};
use crate::{ActiveDrone, GameState};

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_sim)
            .add_systems(FixedUpdate, sim_step.run_if(in_state(GameState::Running)));
    }
}

/// How often the MPPI AI recomputes (every N control ticks).
/// At 100Hz control, 10 → MPPI runs at 10Hz. Cached action used in between.
const MPPI_EVERY_N_TICKS: u32 = 10;

/// All simulation state kept in a single resource.
#[derive(Resource)]
pub struct SimState {
    pub state_a: DroneState,
    pub state_b: DroneState,
    pub params: DroneParams,
    pub arena: Arena,
    pub lock_a: LockOnTracker,
    pub lock_b: LockOnTracker,
    pub wind_a: WindModel,
    pub wind_b: WindModel,
    pub mppi: MppiOptimizer,
    /// Trained neural network policy (if policy.bin found at startup).
    pub policy: Option<MlpPolicy>,
    pub dt_ctrl: f64,
    pub substeps: usize,
    pub rng: rand::rngs::StdRng,
    /// Tick counter for throttling MPPI
    pub tick: u32,
    /// Cached AI motor command (reused between MPPI calls)
    pub cached_ai_motors: Vector4<f64>,
    // Safety & recording
    pub safety: SafetyEnvelope,
    pub safety_status_a: SafetyStatus,
    pub safety_status_b: SafetyStatus,
    pub recorder: SimRecorder,
    // Cached per-frame data for HUD
    pub distance: f64,
    pub a_sees_b: bool,
    pub b_sees_a: bool,
    pub sdf_a: f64,
    pub sdf_b: f64,
    pub collision_a: bool,
    pub collision_b: bool,
    pub kill_a: bool,
    pub kill_b: bool,
}

fn init_sim(mut commands: Commands, config: Res<GameConfig>) {
    let state_a = DroneState::hover_at(config.spawn_a);
    let state_b = DroneState::hover_at(config.spawn_b);

    let wind_a = if config.wind_sigma > 0.0 {
        WindModel::new(config.wind_theta, config.wind_mu, config.wind_sigma)
    } else {
        WindModel::disabled()
    };
    let wind_b = if config.wind_sigma > 0.0 {
        WindModel::new(config.wind_theta, config.wind_mu, config.wind_sigma)
    } else {
        WindModel::disabled()
    };

    let mppi = MppiOptimizer::new(
        config.mppi_num_samples,
        config.mppi_horizon,
        config.mppi_noise_std,
        config.mppi_temperature,
        config.drone_params.clone(),
        config.arena.clone(),
        config.cost_weights.clone(),
        config.dt_ctrl,
        config.substeps,
    );

    let hover = config.drone_params.hover_thrust();
    let hover_motors = Vector4::new(hover, hover, hover, hover);

    // Try to load a trained neural network policy
    let policy = MlpPolicy::load("policy.bin", hover, config.drone_params.max_thrust);
    if policy.is_some() {
        println!("[ACES] Loaded neural network policy from policy.bin");
    } else {
        println!("[ACES] No policy.bin found — using MPPI AI opponent");
    }

    commands.insert_resource(SimState {
        state_a,
        state_b,
        params: config.drone_params.clone(),
        arena: config.arena.clone(),
        lock_a: LockOnTracker::new(config.lock_on_params.clone()),
        lock_b: LockOnTracker::new(config.lock_on_params.clone()),
        wind_a,
        wind_b,
        mppi,
        policy,
        dt_ctrl: config.dt_ctrl,
        substeps: config.substeps,
        rng: rand::rngs::StdRng::seed_from_u64(42),
        tick: 0,
        cached_ai_motors: hover_motors,
        safety: SafetyEnvelope::crazyflie(),
        safety_status_a: SafetyStatus::ok(),
        safety_status_b: SafetyStatus::ok(),
        recorder: SimRecorder::new(true),
        distance: 0.0,
        a_sees_b: false,
        b_sees_a: false,
        sdf_a: 10.0,
        sdf_b: 10.0,
        collision_a: false,
        collision_b: false,
        kill_a: false,
        kill_b: false,
    });
}

/// Convert high-level DroneCommand to 4 motor thrusts using inverse motor mixing.
fn command_to_motors(cmd: &DroneCommand, params: &DroneParams) -> Vector4<f64> {
    let hover = params.hover_thrust();
    let max_t = params.max_thrust;
    let d = params.arm_length;
    let c = params.torque_coeff;
    let s = std::f64::consts::FRAC_1_SQRT_2;

    // Throttle: 0 → hover, +1 → max, -1 → 0
    let collective = (hover * 4.0 + cmd.throttle as f64 * hover * 4.0).clamp(0.0, max_t * 4.0);

    // Torques from stick input (scaled to reasonable values)
    let torque_scale = max_t * d * 2.0; // scale factor for torque commands
    let tau_x = cmd.roll as f64 * torque_scale;
    let tau_y = cmd.pitch as f64 * torque_scale;
    let tau_z = cmd.yaw as f64 * torque_scale * 0.5;

    // Inverse of X-config mixing matrix
    // Forward: tau_x = d*s*(m0 - m1 - m2 + m3)
    //          tau_y = d*s*(m0 + m1 - m2 - m3)
    //          tau_z = c*(m0 - m1 + m2 - m3)
    let m0 = collective / 4.0 + tau_x / (4.0 * d * s) + tau_y / (4.0 * d * s) + tau_z / (4.0 * c);
    let m1 = collective / 4.0 - tau_x / (4.0 * d * s) + tau_y / (4.0 * d * s) - tau_z / (4.0 * c);
    let m2 = collective / 4.0 - tau_x / (4.0 * d * s) - tau_y / (4.0 * d * s) + tau_z / (4.0 * c);
    let m3 = collective / 4.0 + tau_x / (4.0 * d * s) - tau_y / (4.0 * d * s) - tau_z / (4.0 * c);

    Vector4::new(
        m0.clamp(0.0, max_t),
        m1.clamp(0.0, max_t),
        m2.clamp(0.0, max_t),
        m3.clamp(0.0, max_t),
    )
}

fn sim_step(mut sim: ResMut<SimState>, cmd: Res<DroneCommand>, active: Res<ActiveDrone>) {
    // Plain &mut allows field-level borrow splitting (unlike DerefMut on ResMut)
    let s: &mut SimState = &mut sim;

    let player_motors = command_to_motors(&cmd, &s.params);

    // AI opponent: neural network (every tick) or MPPI (throttled)
    let ai_motors = if let Some(ref nn) = s.policy {
        let lock_a_p = s.lock_a.progress();
        let lock_b_p = s.lock_b.progress();
        let (ai_own, ai_opp, lock_p, locked_p) = match *active {
            ActiveDrone::A => (&s.state_b, &s.state_a, lock_b_p, lock_a_p),
            ActiveDrone::B => (&s.state_a, &s.state_b, lock_a_p, lock_b_p),
        };
        let obs_dist = s.arena.obstacle_sdf(&ai_own.position);
        let obs = policy::build_obs(ai_own, ai_opp, obs_dist, lock_p, locked_p);
        let action = nn.infer(&obs);
        let m = nn.action_to_motors(&action);
        Vector4::new(m[0], m[1], m[2], m[3])
    } else {
        if s.tick.is_multiple_of(MPPI_EVERY_N_TICKS) {
            let (ai_state, player_state) = match *active {
                ActiveDrone::A => (s.state_b.clone(), s.state_a.clone()),
                ActiveDrone::B => (s.state_a.clone(), s.state_b.clone()),
            };
            s.cached_ai_motors = s.mppi.compute_action(&ai_state, &player_state, true);
        }
        s.tick = s.tick.wrapping_add(1);
        s.cached_ai_motors
    };

    let (raw_motors_a, raw_motors_b) = match *active {
        ActiveDrone::A => (player_motors, ai_motors),
        ActiveDrone::B => (ai_motors, player_motors),
    };

    // Safety: sanitize NaN motors before physics
    let motors_a = s
        .safety
        .sanitize_motors(raw_motors_a, &s.state_a, &s.params);
    let motors_b = s
        .safety
        .sanitize_motors(raw_motors_b, &s.state_b, &s.params);

    let dt_ctrl = s.dt_ctrl;
    let substeps = s.substeps;
    let dt_sim = dt_ctrl / substeps as f64;

    let mut last_wind_a = nalgebra::Vector3::zeros();
    let mut last_wind_b = nalgebra::Vector3::zeros();

    // Physics substeps with per-substep wind (matches py-bridge behavior)
    for _ in 0..substeps {
        last_wind_a = s.wind_a.step(dt_sim, &mut s.rng);
        last_wind_b = s.wind_b.step(dt_sim, &mut s.rng);
        s.state_a = step_rk4(&s.state_a, &motors_a, &s.params, dt_sim, &last_wind_a);
        s.state_b = step_rk4(&s.state_b, &motors_b, &s.params, dt_sim, &last_wind_b);
    }

    // Clone states for lock-on / collision checks (avoids borrow conflicts)
    let state_a = s.state_a.clone();
    let state_b = s.state_b.clone();

    // Safety checks
    s.safety_status_a = s.safety.check(&state_a, &s.arena);
    s.safety_status_b = s.safety.check(&state_b, &s.arena);

    // Lock-on
    let kill_a = s.lock_a.update(&state_a, &state_b, &s.arena, dt_ctrl);
    let kill_b = s.lock_b.update(&state_b, &state_a, &s.arena, dt_ctrl);
    s.kill_a = kill_a;
    s.kill_b = kill_b;

    // Collision & SDF
    s.collision_a = s.arena.is_collision(&state_a.position);
    s.collision_b = s.arena.is_collision(&state_b.position);
    s.sdf_a = s.arena.sdf(&state_a.position);
    s.sdf_b = s.arena.sdf(&state_b.position);

    // Visibility
    s.a_sees_b =
        check_line_of_sight(&s.arena, &state_a.position, &state_b.position) == Visibility::Visible;
    s.b_sees_a =
        check_line_of_sight(&s.arena, &state_b.position, &state_a.position) == Visibility::Visible;

    // Distance
    s.distance = state_a.distance_to(&state_b);

    // Record frame
    s.recorder.record(SimFrame {
        tick: s.tick as u64,
        timestamp: s.tick as f64 * dt_ctrl,
        state_a: state_a.to_array(),
        state_b: state_b.to_array(),
        motors_a: [motors_a[0], motors_a[1], motors_a[2], motors_a[3]],
        motors_b: [motors_b[0], motors_b[1], motors_b[2], motors_b[3]],
        wind_a: [last_wind_a.x, last_wind_a.y, last_wind_a.z],
        wind_b: [last_wind_b.x, last_wind_b.y, last_wind_b.z],
        safety_a: severity_to_u8(s.safety_status_a.severity),
        safety_b: severity_to_u8(s.safety_status_b.severity),
        lock_progress_a: s.lock_a.progress(),
        lock_progress_b: s.lock_b.progress(),
        a_sees_b: s.a_sees_b,
        b_sees_a: s.b_sees_a,
    });

    // Log drone state every 100 ticks (1 second at 100Hz)
    if s.tick.is_multiple_of(100) {
        bevy::log::info!(
            "tick={} A=[{:.2},{:.2},{:.2}] B=[{:.2},{:.2},{:.2}] dist={:.2} lock_a={:.0}% lock_b={:.0}%",
            s.tick,
            s.state_a.position.x, s.state_a.position.y, s.state_a.position.z,
            s.state_b.position.x, s.state_b.position.y, s.state_b.position.z,
            s.distance,
            s.lock_a.progress() * 100.0,
            s.lock_b.progress() * 100.0,
        );
    }
}

/// Reset sim to initial state.
pub fn reset_sim(sim: &mut SimState, config: &GameConfig) {
    // Save recording before clearing if there are frames
    if !sim.recorder.is_empty() {
        let filename = format!("sim_recording_{}.csv", sim.tick);
        if let Err(e) = sim.recorder.save_csv(&filename) {
            bevy::log::warn!("Failed to save recording: {e}");
        }
    }

    sim.state_a = DroneState::hover_at(config.spawn_a);
    sim.state_b = DroneState::hover_at(config.spawn_b);
    sim.lock_a.reset();
    sim.lock_b.reset();
    sim.wind_a.reset();
    sim.wind_b.reset();
    sim.mppi.reset();
    sim.tick = 0;
    let hover = sim.params.hover_thrust();
    sim.cached_ai_motors = Vector4::new(hover, hover, hover, hover);
    sim.safety_status_a = SafetyStatus::ok();
    sim.safety_status_b = SafetyStatus::ok();
    sim.recorder.clear();
    sim.kill_a = false;
    sim.kill_b = false;
    sim.collision_a = false;
    sim.collision_b = false;
}
