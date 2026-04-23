//! Load and run a trained MLP policy exported from Python.
//!
//! Binary format (little-endian): see `aces/export.py` and `crates/game/src/weights.rs`.
//! Pure nalgebra, zero extra dependencies.

use aces_sim_core::state::DroneState;
use nalgebra::{DMatrix, DVector};

use crate::weights::load_mlp_weights;

/// A loaded MLP actor network (typically 21→64→64→4 with Tanh).
pub struct MlpPolicy {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
    hover_thrust: f64,
    max_thrust: f64,
}

impl MlpPolicy {
    /// Try to load a policy from the binary file. Returns `None` on any error.
    pub fn load(path: &str, hover_thrust: f64, max_thrust: f64) -> Option<Self> {
        let layers = load_mlp_weights(path)?;

        // Sanity check: first layer input should be 21 (vector obs)
        if let Some((w, _)) = layers.first() {
            if w.ncols() != 21 {
                eprintln!(
                    "[ACES] Warning: policy input dim is {} (expected 21 for MlpPolicy)",
                    w.ncols()
                );
            }
        }

        Some(Self {
            layers,
            hover_thrust,
            max_thrust,
        })
    }

    /// Forward pass: 21-dim observation → 4-dim action in [-1, 1].
    pub fn infer(&self, obs: &[f64; 21]) -> [f64; 4] {
        let mut x = DVector::from_column_slice(obs);
        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = w * &x + b;
            if i < self.layers.len() - 1 {
                x = x.map(|v| v.tanh());
            }
        }
        // Tanh squashing on final layer
        x = x.map(|v| v.tanh());
        [x[0], x[1], x[2], x[3]]
    }

    /// Map action [-1, 1] → motor thrusts [0, max_thrust].
    pub fn action_to_motors(&self, action: &[f64; 4]) -> [f64; 4] {
        core::array::from_fn(|i| {
            (self.hover_thrust + action[i] * (self.max_thrust - self.hover_thrust))
                .clamp(0.0, self.max_thrust)
        })
    }
}

/// Build the 21-dim observation vector that matches `env.py::_build_obs`.
pub fn build_obs(
    own: &DroneState,
    opp: &DroneState,
    nearest_obs_dist: f64,
    lock_progress: f64,
    being_locked_progress: f64,
) -> [f64; 21] {
    let rel_pos = opp.position - own.position;
    let rel_vel = opp.velocity - own.velocity;
    let (roll, pitch, yaw) = own.attitude.euler_angles();

    [
        own.velocity.x,
        own.velocity.y,
        own.velocity.z,
        own.angular_velocity.x,
        own.angular_velocity.y,
        own.angular_velocity.z,
        rel_pos.x,
        rel_pos.y,
        rel_pos.z,
        rel_vel.x,
        rel_vel.y,
        rel_vel.z,
        roll,
        pitch,
        yaw,
        nearest_obs_dist,
        lock_progress,
        being_locked_progress,
        1.0, // opponent_visible (always visible in game)
        0.0, // belief_uncertainty (no PF in game)
        0.0, // time_since_last_seen
    ]
}
