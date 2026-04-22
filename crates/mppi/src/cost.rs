use aces_sim_core::environment::Arena;
use aces_sim_core::state::DroneState;
use nalgebra::Vector4;

/// Cost function weights for the MPPI controller.
#[derive(Debug, Clone)]
pub struct CostWeights {
    /// Weight for distance to opponent
    pub w_dist: f64,
    /// Weight for facing opponent
    pub w_face: f64,
    /// Weight for control smoothness
    pub w_ctrl: f64,
    /// Weight for obstacle avoidance
    pub w_obs: f64,
    /// Safe distance from obstacles (meters)
    pub d_safe: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            w_dist: 1.0,
            w_face: 5.0,
            w_ctrl: 0.01,
            w_obs: 1000.0,
            d_safe: 0.3,
        }
    }
}

/// Compute stage cost for pursuit behavior.
pub fn pursuit_cost(
    self_state: &DroneState,
    enemy_state: &DroneState,
    control: &Vector4<f64>,
    hover_thrust: f64,
    arena: &Arena,
    weights: &CostWeights,
) -> f64 {
    let mut cost = 0.0;

    // Distance to opponent
    let dist = self_state.distance_to(enemy_state);
    cost += weights.w_dist * dist * dist;

    // Facing opponent (1 - cos(angle))
    let angle = self_state.angle_to(enemy_state);
    cost += weights.w_face * (1.0 - angle.cos());

    // Control smoothness (deviation from hover)
    let hover = Vector4::new(hover_thrust, hover_thrust, hover_thrust, hover_thrust);
    let ctrl_diff = control - hover;
    cost += weights.w_ctrl * ctrl_diff.norm_squared();

    // Obstacle avoidance
    let sdf = arena.sdf(&self_state.position);
    if sdf <= 0.0 {
        cost += 1e6; // Inside obstacle — extreme penalty
    } else if sdf < weights.d_safe {
        let margin = weights.d_safe - sdf;
        cost += weights.w_obs * margin * margin;
    }

    cost
}

/// Belief-weighted pursuit cost (Level 3).
///
/// Scales opponent-relative cost terms by a confidence factor derived from
/// the belief state variance. When uncertainty is high, the controller
/// prioritises obstacle avoidance and control smoothness over pursuit.
///
/// `belief_var` is the position variance from the particle filter (0 = certain).
pub fn belief_pursuit_cost(
    self_state: &DroneState,
    enemy_state: &DroneState,
    control: &Vector4<f64>,
    hover_thrust: f64,
    arena: &Arena,
    weights: &CostWeights,
    belief_var: f64,
) -> f64 {
    // Confidence: 1.0 when certain, decays towards 0 as variance grows
    let confidence = 1.0 / (1.0 + belief_var);
    let mut cost = 0.0;

    // Opponent-relative terms scaled by confidence
    let dist = self_state.distance_to(enemy_state);
    cost += confidence * weights.w_dist * dist * dist;

    let angle = self_state.angle_to(enemy_state);
    cost += confidence * weights.w_face * (1.0 - angle.cos());

    // Control smoothness (always active)
    let hover = Vector4::new(hover_thrust, hover_thrust, hover_thrust, hover_thrust);
    let ctrl_diff = control - hover;
    cost += weights.w_ctrl * ctrl_diff.norm_squared();

    // Obstacle avoidance (always active, more conservative under uncertainty)
    let sdf = arena.sdf(&self_state.position);
    let effective_safe = weights.d_safe * (1.0 + belief_var.sqrt());
    if sdf <= 0.0 {
        cost += 1e6;
    } else if sdf < effective_safe {
        let margin = effective_safe - sdf;
        cost += weights.w_obs * margin * margin;
    }

    cost
}

/// Belief-weighted evasion cost (Level 3).
///
/// When belief uncertainty is high, obstacle avoidance margins increase.
pub fn belief_evasion_cost(
    self_state: &DroneState,
    enemy_state: &DroneState,
    control: &Vector4<f64>,
    hover_thrust: f64,
    arena: &Arena,
    weights: &CostWeights,
    belief_var: f64,
) -> f64 {
    let confidence = 1.0 / (1.0 + belief_var);
    let mut cost = 0.0;

    let dist = self_state.distance_to(enemy_state);
    let safe_dist = 3.0;
    if dist < safe_dist {
        cost += confidence * weights.w_dist * (safe_dist - dist).powi(2);
    }

    let angle_from_enemy = enemy_state.angle_to(self_state);
    let fov_half = std::f64::consts::FRAC_PI_4;
    if angle_from_enemy < fov_half {
        cost += confidence * weights.w_face * (1.0 - angle_from_enemy / fov_half);
    }

    let hover = Vector4::new(hover_thrust, hover_thrust, hover_thrust, hover_thrust);
    let ctrl_diff = control - hover;
    cost += weights.w_ctrl * ctrl_diff.norm_squared();

    let sdf = arena.sdf(&self_state.position);
    let effective_safe = weights.d_safe * (1.0 + belief_var.sqrt());
    if sdf <= 0.0 {
        cost += 1e6;
    } else if sdf < effective_safe {
        let margin = effective_safe - sdf;
        cost += weights.w_obs * margin * margin;
    }

    cost
}

/// Compute stage cost for evasion behavior.
pub fn evasion_cost(
    self_state: &DroneState,
    enemy_state: &DroneState,
    control: &Vector4<f64>,
    hover_thrust: f64,
    arena: &Arena,
    weights: &CostWeights,
) -> f64 {
    let mut cost = 0.0;

    // Reward distance from opponent (penalize being too close)
    let dist = self_state.distance_to(enemy_state);
    let safe_dist = 3.0; // meters
    if dist < safe_dist {
        cost += weights.w_dist * (safe_dist - dist).powi(2);
    }

    // Penalize being in enemy's FOV
    let angle_from_enemy = enemy_state.angle_to(self_state);
    let fov_half = std::f64::consts::FRAC_PI_4; // 45 degrees
    if angle_from_enemy < fov_half {
        cost += weights.w_face * (1.0 - angle_from_enemy / fov_half);
    }

    // Control smoothness
    let hover = Vector4::new(hover_thrust, hover_thrust, hover_thrust, hover_thrust);
    let ctrl_diff = control - hover;
    cost += weights.w_ctrl * ctrl_diff.norm_squared();

    // Obstacle avoidance
    let sdf = arena.sdf(&self_state.position);
    if sdf <= 0.0 {
        cost += 1e6;
    } else if sdf < weights.d_safe {
        let margin = weights.d_safe - sdf;
        cost += weights.w_obs * margin * margin;
    }

    cost
}
