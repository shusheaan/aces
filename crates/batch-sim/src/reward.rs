use crate::battle::BattleInfo;

/// Reward configuration for shaped rewards.
#[derive(Debug, Clone)]
pub struct RewardConfig {
    pub kill_reward: f64,
    pub killed_penalty: f64,
    pub collision_penalty: f64,
    pub opponent_crash_reward: f64,
    pub lock_progress_reward: f64,
    pub approach_reward: f64,
    pub survival_bonus: f64,
    pub control_penalty: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            kill_reward: 100.0,
            killed_penalty: -100.0,
            collision_penalty: -50.0,
            opponent_crash_reward: 5.0,
            lock_progress_reward: 5.0,
            approach_reward: 3.0,
            survival_bonus: 0.01,
            control_penalty: 0.01,
        }
    }
}

/// Compute shaped reward for drone A after a battle step.
///
/// `prev_distance`: distance between drones at the previous step.
/// `prev_lock_progress_a`: A's lock-on progress at the previous step.
/// `control_cost`: sum of squared motor deviations from hover.
pub fn compute_reward_a(
    info: &BattleInfo,
    prev_distance: f64,
    prev_lock_progress_a: f64,
    control_cost: f64,
    config: &RewardConfig,
) -> f64 {
    // Terminal rewards
    if info.kill_a {
        return config.kill_reward;
    }
    if info.kill_b {
        return config.killed_penalty;
    }
    if info.collision_a {
        return config.collision_penalty;
    }
    if info.collision_b {
        return config.opponent_crash_reward;
    }
    if info.timeout {
        return 0.0;
    }

    // Shaping rewards (per step, non-terminal)
    let mut reward = config.survival_bonus;

    // Approach reward: positive when closing distance
    let delta_distance = prev_distance - info.distance;
    reward += config.approach_reward * delta_distance;

    // Lock progress reward: positive when increasing lock
    let delta_lock = info.lock_progress_a - prev_lock_progress_a;
    if delta_lock > 0.0 {
        reward += config.lock_progress_reward * delta_lock;
    }

    // Control penalty
    reward -= config.control_penalty * control_cost;

    reward
}

/// Compute control cost (deviation from hover).
pub fn control_cost(motors: &[f64; 4], hover_thrust: f64) -> f64 {
    motors
        .iter()
        .map(|m| (m - hover_thrust).powi(2))
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kill_reward() {
        let config = RewardConfig::default();
        let info = BattleInfo {
            kill_a: true,
            ..Default::default()
        };
        let r = compute_reward_a(&info, 5.0, 0.0, 0.0, &config);
        assert_eq!(r, 100.0);
    }

    #[test]
    fn test_approach_reward() {
        let config = RewardConfig::default();
        let info = BattleInfo {
            distance: 3.0,
            ..Default::default()
        };
        // Closed 2m of distance
        let r = compute_reward_a(&info, 5.0, 0.0, 0.0, &config);
        assert!(r > config.survival_bonus, "should get approach bonus");
    }

    #[test]
    fn test_retreat_penalty() {
        let config = RewardConfig::default();
        let info = BattleInfo {
            distance: 7.0,
            ..Default::default()
        };
        // Increased distance by 2m
        let r = compute_reward_a(&info, 5.0, 0.0, 0.0, &config);
        assert!(
            r < config.survival_bonus,
            "should be penalized for retreating"
        );
    }
}
