use aces_sim_core::dynamics::{step_rk4, DroneParams};
use aces_sim_core::environment::Arena;
use aces_sim_core::state::DroneState;
use aces_sim_core::wind::WindModel;
use nalgebra::{Vector3, Vector4};
use rand::Rng;

/// Simulate a trajectory forward given a control sequence.
///
/// Returns the sequence of states (length = horizon + 1, including initial state).
/// Uses zero external force (planning model, no wind).
pub fn rollout(
    initial_state: &DroneState,
    controls: &[Vector4<f64>],
    params: &DroneParams,
    dt_ctrl: f64,
    substeps: usize,
) -> Vec<DroneState> {
    let dt_sim = dt_ctrl / substeps as f64;
    let no_wind = Vector3::zeros();
    let mut states = Vec::with_capacity(controls.len() + 1);
    let mut state = initial_state.clone();
    states.push(state.clone());

    for u in controls {
        for _ in 0..substeps {
            state = step_rk4(&state, u, params, dt_sim, &no_wind);
        }
        states.push(state.clone());
    }

    states
}

/// Result of a wind-aware rollout, including collision risk info.
pub struct WindRolloutResult {
    pub states: Vec<DroneState>,
    /// Maximum penetration depth (positive = collision). 0 or negative = safe.
    pub max_penetration: f64,
}

/// Simulate a trajectory with sampled wind disturbances.
///
/// Each rollout gets its own wind realization from the OU process,
/// modeling the fact that future wind is uncertain.
#[allow(clippy::too_many_arguments)]
pub fn rollout_with_wind<R: Rng>(
    initial_state: &DroneState,
    controls: &[Vector4<f64>],
    params: &DroneParams,
    arena: &Arena,
    dt_ctrl: f64,
    substeps: usize,
    wind_model: &WindModel,
    rng: &mut R,
) -> WindRolloutResult {
    let dt_sim = dt_ctrl / substeps as f64;
    let mut states = Vec::with_capacity(controls.len() + 1);
    let mut state = initial_state.clone();
    states.push(state.clone());

    // Clone wind model so each rollout starts from the same state
    // but evolves independently
    let mut wind = wind_model.clone();
    let mut max_penetration = f64::NEG_INFINITY;

    for u in controls {
        for _ in 0..substeps {
            let wind_force = wind.step(dt_sim, rng);
            state = step_rk4(&state, u, params, dt_sim, &wind_force);
        }

        // Track collision risk: drone_radius - SDF (positive means inside obstacle)
        let sdf = arena.sdf(&state.position);
        let penetration = arena.drone_radius - sdf;
        if penetration > max_penetration {
            max_penetration = penetration;
        }

        states.push(state.clone());
    }

    WindRolloutResult {
        states,
        max_penetration,
    }
}
