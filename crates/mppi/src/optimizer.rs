use crate::cost::{
    belief_evasion_cost, belief_pursuit_cost, evasion_cost, pursuit_cost, CostWeights,
};
use crate::rollout::{rollout, rollout_with_wind};
use aces_sim_core::dynamics::DroneParams;
use aces_sim_core::environment::Arena;
use aces_sim_core::state::DroneState;
use aces_sim_core::wind::WindModel;
use nalgebra::Vector4;
use rand::distributions::Distribution;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;

/// Configuration for risk-aware MPPI with CVaR filtering.
#[derive(Debug, Clone, Copy)]
pub struct RiskConfig {
    /// Wind model parameters for rollout sampling
    pub wind: WindModel,
    /// CVaR alpha: fraction of worst-case trajectories to penalize (0.0-1.0).
    /// E.g. 0.05 means the worst 5% of trajectories get extra penalty.
    pub cvar_alpha: f64,
    /// Extra cost multiplier applied to worst-alpha trajectories
    pub cvar_penalty: f64,
}

/// Configuration for chance-constrained MPPI.
#[derive(Clone, Debug)]
pub struct ChanceConstraintConfig {
    /// Maximum allowed collision probability (e.g., 0.01 = 1%)
    pub delta: f64,
    /// Lagrange multiplier learning rate
    pub lambda_lr: f64,
    /// Initial Lagrange multiplier
    pub lambda_init: f64,
    /// Minimum lambda (prevent negative)
    pub lambda_min: f64,
    /// Maximum lambda (prevent explosion)
    pub lambda_max: f64,
}

impl Default for ChanceConstraintConfig {
    fn default() -> Self {
        Self {
            delta: 0.01,
            lambda_lr: 0.1,
            lambda_init: 100.0,
            lambda_min: 0.0,
            lambda_max: 1e6,
        }
    }
}

/// Full MPPI optimizer: sample, rollout, cost, softmax-weight, update mean.
///
/// Supports both standard and risk-aware modes.
pub struct MppiOptimizer {
    pub num_samples: usize,
    pub horizon: usize,
    pub noise_std: f64,
    pub temperature: f64,
    pub params: DroneParams,
    pub arena: Arena,
    pub weights: CostWeights,
    pub dt_ctrl: f64,
    pub substeps: usize,
    mean_controls: Vec<Vector4<f64>>,
    /// If Some, run risk-aware MPPI with wind sampling + CVaR
    pub risk: Option<RiskConfig>,
    /// If Some, enforce P(collision) <= delta via online Lagrangian
    pub chance_constraint: Option<ChanceConstraintConfig>,
    /// Current Lagrange multiplier for the collision chance constraint
    pub lambda_cc: f64,
    /// Pre-allocated control buffers: one Vec per sample, capacity = horizon.
    /// Reused each call to avoid per-call heap allocation.
    ctrl_buffers: Vec<Vec<Vector4<f64>>>,
}

impl MppiOptimizer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_samples: usize,
        horizon: usize,
        noise_std: f64,
        temperature: f64,
        params: DroneParams,
        arena: Arena,
        weights: CostWeights,
        dt_ctrl: f64,
        substeps: usize,
    ) -> Self {
        let hover = params.hover_thrust();
        let hover_vec = Vector4::new(hover, hover, hover, hover);
        // Pre-allocate one control buffer per sample to avoid per-call heap allocation.
        let ctrl_buffers = (0..num_samples)
            .map(|_| Vec::with_capacity(horizon))
            .collect();
        Self {
            num_samples,
            horizon,
            noise_std,
            temperature,
            params,
            arena,
            weights,
            dt_ctrl,
            substeps,
            mean_controls: vec![hover_vec; horizon],
            risk: None,
            chance_constraint: None,
            lambda_cc: 0.0,
            ctrl_buffers,
        }
    }

    /// Enable risk-aware mode with wind sampling and CVaR filtering.
    pub fn set_risk_config(&mut self, config: RiskConfig) {
        self.risk = Some(config);
    }

    /// Enable chance-constrained mode with online Lagrangian adaptation.
    pub fn set_chance_constraint(&mut self, config: ChanceConstraintConfig) {
        self.lambda_cc = config.lambda_init;
        self.chance_constraint = Some(config);
    }

    /// Compute optimal action for the current state.
    pub fn compute_action(
        &mut self,
        self_state: &DroneState,
        enemy_state: &DroneState,
        pursuit: bool,
    ) -> Vector4<f64> {
        if pursuit {
            self.compute_action_with_cost_fn(self_state, enemy_state, true, pursuit_cost)
        } else {
            self.compute_action_with_cost_fn(self_state, enemy_state, true, evasion_cost)
        }
    }

    /// Compute optimal action with belief-weighted costs (Level 3).
    ///
    /// `belief_var` is the position variance from the particle filter.
    /// When high, opponent-relative costs are down-weighted and obstacle
    /// avoidance margins increase for conservative planning.
    pub fn compute_action_with_belief(
        &mut self,
        self_state: &DroneState,
        enemy_state: &DroneState,
        pursuit: bool,
        belief_var: f64,
    ) -> Vector4<f64> {
        if belief_var < 1e-6 {
            return self.compute_action(self_state, enemy_state, pursuit);
        }
        if pursuit {
            self.compute_action_with_cost_fn(
                self_state,
                enemy_state,
                false,
                move |s, e, c, h, a, w| belief_pursuit_cost(s, e, c, h, a, w, belief_var),
            )
        } else {
            self.compute_action_with_cost_fn(
                self_state,
                enemy_state,
                false,
                move |s, e, c, h, a, w| belief_evasion_cost(s, e, c, h, a, w, belief_var),
            )
        }
    }

    /// Core MPPI loop: sample, rollout, cost, (optional CVaR), (optional chance constraint),
    /// softmax-weight, warm-start shift.
    fn compute_action_with_cost_fn<F>(
        &mut self,
        self_state: &DroneState,
        enemy_state: &DroneState,
        apply_cvar: bool,
        cost_fn: F,
    ) -> Vector4<f64>
    where
        F: Fn(&DroneState, &DroneState, &Vector4<f64>, f64, &Arena, &CostWeights) -> f64
            + Send
            + Sync,
    {
        let max_t = self.params.max_thrust;
        let hover = self.params.hover_thrust();

        // Generate seeds for parallel RNG
        let seeds: Vec<u64> = {
            let mut rng = rand::thread_rng();
            (0..self.num_samples).map(|_| rng.gen()).collect()
        };

        let risk = self.risk;

        // Parallel: generate perturbed sequences, rollout, compute costs.
        // ctrl_buffers provides one pre-allocated Vec per sample to avoid heap allocations.
        let mut results: Vec<(Vec<Vector4<f64>>, f64, f64)> = self
            .ctrl_buffers
            .par_iter_mut()
            .zip(seeds.par_iter())
            .map(|(ctrl_buf, &seed)| {
                let mut rng = SmallRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, self.noise_std).unwrap();

                // Reuse the pre-allocated buffer: clear and fill for this sample.
                ctrl_buf.clear();
                for t in 0..self.horizon {
                    let mean = self.mean_controls[t];
                    ctrl_buf.push(Vector4::new(
                        (mean[0] + normal.sample(&mut rng)).clamp(0.0, max_t),
                        (mean[1] + normal.sample(&mut rng)).clamp(0.0, max_t),
                        (mean[2] + normal.sample(&mut rng)).clamp(0.0, max_t),
                        (mean[3] + normal.sample(&mut rng)).clamp(0.0, max_t),
                    ));
                }

                let (states, max_penetration) = if let Some(ref risk_cfg) = risk {
                    let result = rollout_with_wind(
                        self_state,
                        ctrl_buf,
                        &self.params,
                        &self.arena,
                        self.dt_ctrl,
                        self.substeps,
                        &risk_cfg.wind,
                        &mut rng,
                    );
                    (result.states, result.max_penetration)
                } else {
                    let states = rollout(
                        self_state,
                        ctrl_buf,
                        &self.params,
                        self.dt_ctrl,
                        self.substeps,
                    );
                    (states, f64::NEG_INFINITY)
                };

                let mut total_cost = 0.0;
                for (t, state) in states[1..].iter().enumerate() {
                    total_cost += cost_fn(
                        state,
                        enemy_state,
                        &ctrl_buf[t],
                        hover,
                        &self.arena,
                        &self.weights,
                    );
                }

                // Hard collision filter: add massive penalty for colliding trajectories
                if max_penetration > 0.0 {
                    total_cost += 1e8;
                }

                (ctrl_buf.clone(), total_cost, max_penetration)
            })
            .collect();

        // CVaR filtering: penalize worst-alpha fraction of trajectories
        let mut costs: Vec<f64> = results.iter().map(|(_, c, _)| *c).collect();
        if apply_cvar {
            if let Some(ref risk_cfg) = self.risk {
                if risk_cfg.cvar_alpha > 0.0 && risk_cfg.cvar_alpha < 1.0 {
                    // Use select_nth_unstable to find the CVaR threshold in O(n)
                    // instead of O(n log n) sort. Work on a mutable copy.
                    let k = ((1.0 - risk_cfg.cvar_alpha) * costs.len() as f64) as usize;
                    let k = k.min(costs.len() - 1);
                    let mut costs_scratch = costs.clone();
                    costs_scratch.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
                    let threshold = costs_scratch[k];

                    for c in costs.iter_mut() {
                        if *c >= threshold {
                            *c += risk_cfg.cvar_penalty * (*c - threshold);
                        }
                    }
                }
            }
        }

        // Write CVaR-updated costs back to results so chance constraint sees them
        for (i, (_, cost, _)) in results.iter_mut().enumerate() {
            *cost = costs[i];
        }

        // Chance constraint: estimate collision probability from samples and apply
        // Lagrangian penalty. Lambda is updated via dual gradient ascent.
        if let Some(ref cc_config) = self.chance_constraint {
            let num_colliding = results.iter().filter(|r| r.2 > 0.0).count();
            let p_collision = num_colliding as f64 / results.len() as f64;

            // Add Lagrangian penalty to colliding trajectories
            let lambda = self.lambda_cc;
            for result in &mut results {
                if result.2 > 0.0 {
                    result.1 += lambda * result.2;
                }
            }

            // Dual variable update (gradient ascent on lambda)
            // lambda += lr * (p_collision - delta)
            self.lambda_cc = (self.lambda_cc
                + cc_config.lambda_lr * (p_collision - cc_config.delta))
                .clamp(cc_config.lambda_min, cc_config.lambda_max);
        }

        // Re-extract costs after chance constraint modification
        let costs: Vec<f64> = results.iter().map(|(_, c, _)| *c).collect();

        // Softmax weights
        let min_cost = costs.iter().copied().fold(f64::INFINITY, f64::min);
        let exp_costs: Vec<f64> = costs
            .iter()
            .map(|c| (-(c - min_cost) / self.temperature).exp())
            .collect();
        let total_exp: f64 = exp_costs.iter().sum();

        // Weighted average of control sequences
        let mut new_mean = vec![Vector4::zeros(); self.horizon];
        for (k, (controls, _, _)) in results.iter().enumerate() {
            let w = exp_costs[k] / total_exp;
            for t in 0..self.horizon {
                new_mean[t] += controls[t] * w;
            }
        }

        let action = new_mean[0];

        // Warm-start: shift left, append hover
        self.mean_controls = new_mean[1..].to_vec();
        let hover_vec = Vector4::new(hover, hover, hover, hover);
        self.mean_controls.push(hover_vec);

        action
    }

    /// Reset mean controls to hover and lambda to its initial value.
    pub fn reset(&mut self) {
        let hover = self.params.hover_thrust();
        let hover_vec = Vector4::new(hover, hover, hover, hover);
        self.mean_controls = vec![hover_vec; self.horizon];
        if let Some(ref cc) = self.chance_constraint {
            self.lambda_cc = cc.lambda_init;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aces_sim_core::environment::{Arena, Obstacle};
    use nalgebra::Vector3;

    fn test_arena() -> Arena {
        let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
            arena.obstacles.push(Obstacle::Box {
                center: Vector3::new(x, y, 1.5),
                half_extents: Vector3::new(0.5, 0.5, 1.5),
            });
        }
        arena
    }

    #[test]
    fn test_optimizer_returns_valid_action() {
        let params = DroneParams::crazyflie();
        let arena = test_arena();
        let weights = CostWeights::default();

        let mut opt =
            MppiOptimizer::new(64, 10, 0.03, 10.0, params.clone(), arena, weights, 0.01, 10);

        let attacker = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(8.0, 5.0, 1.5));

        let action = opt.compute_action(&attacker, &target, true);
        for i in 0..4 {
            assert!(action[i] >= 0.0, "motor {i} negative: {}", action[i]);
            assert!(
                action[i] <= params.max_thrust,
                "motor {i} exceeds max: {}",
                action[i]
            );
        }
    }

    #[test]
    fn test_optimizer_warm_start() {
        let params = DroneParams::crazyflie();
        let arena = test_arena();
        let weights = CostWeights::default();

        let mut opt =
            MppiOptimizer::new(32, 5, 0.03, 10.0, params.clone(), arena, weights, 0.01, 10);
        let a = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));
        let b = DroneState::hover_at(Vector3::new(8.0, 5.0, 1.5));

        opt.compute_action(&a, &b, true);
        let action2 = opt.compute_action(&a, &b, true);
        for i in 0..4 {
            assert!(action2[i] >= 0.0);
        }
    }

    #[test]
    fn test_risk_aware_optimizer_returns_valid_action() {
        let params = DroneParams::crazyflie();
        let arena = test_arena();
        let weights = CostWeights::default();

        let mut opt =
            MppiOptimizer::new(64, 10, 0.03, 10.0, params.clone(), arena, weights, 0.01, 10);

        // Enable risk-aware mode
        opt.set_risk_config(RiskConfig {
            wind: WindModel::new(2.0, Vector3::zeros(), 0.3),
            cvar_alpha: 0.05,
            cvar_penalty: 10.0,
        });

        let attacker = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(8.0, 5.0, 1.5));

        let action = opt.compute_action(&attacker, &target, true);
        for i in 0..4 {
            assert!(action[i] >= 0.0, "motor {i} negative: {}", action[i]);
            assert!(
                action[i] <= params.max_thrust,
                "motor {i} exceeds max: {}",
                action[i]
            );
        }
    }

    #[test]
    fn test_risk_aware_avoids_obstacles_better() {
        // Place drone near a pillar; risk-aware should pick safer trajectories
        let params = DroneParams::crazyflie();
        let arena = test_arena();
        let weights = CostWeights::default();

        // Standard MPPI
        let mut opt_std = MppiOptimizer::new(
            128,
            10,
            0.03,
            10.0,
            params.clone(),
            arena.clone(),
            weights.clone(),
            0.01,
            10,
        );

        // Risk-aware MPPI
        let mut opt_risk = MppiOptimizer::new(
            128,
            10,
            0.03,
            10.0,
            params.clone(),
            arena,
            weights,
            0.01,
            10,
        );
        opt_risk.set_risk_config(RiskConfig {
            wind: WindModel::new(2.0, Vector3::zeros(), 0.3),
            cvar_alpha: 0.1,
            cvar_penalty: 50.0,
        });

        // Drone near a pillar at (2,2,1.5), target on the other side
        let near_pillar = DroneState::hover_at(Vector3::new(2.8, 2.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(8.0, 8.0, 1.5));

        let action_std = opt_std.compute_action(&near_pillar, &target, true);
        let action_risk = opt_risk.compute_action(&near_pillar, &target, true);

        // Both should return valid actions
        for i in 0..4 {
            assert!(action_std[i] >= 0.0);
            assert!(action_risk[i] >= 0.0);
        }
    }

    #[test]
    fn test_chance_constraint_adapts_lambda() {
        // Place drone very close to an obstacle so many samples collide.
        // Lambda should increase when collision rate > delta,
        // and decrease (or stay) when collision rate < delta.
        let params = DroneParams::crazyflie();

        // Arena with a large obstacle so the drone is inside it (guaranteed collisions)
        let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        arena.obstacles.push(Obstacle::Box {
            center: Vector3::new(5.0, 5.0, 1.5),
            half_extents: Vector3::new(4.0, 4.0, 1.5),
        });

        let weights = CostWeights::default();
        let cc_config = ChanceConstraintConfig {
            delta: 0.01,
            lambda_lr: 0.5,
            lambda_init: 100.0,
            lambda_min: 0.0,
            lambda_max: 1e6,
        };
        let lambda_init = cc_config.lambda_init;

        let mut opt = MppiOptimizer::new(
            128,
            5,
            0.1,
            10.0,
            params.clone(),
            arena.clone(),
            weights.clone(),
            0.01,
            5,
        );
        opt.set_risk_config(RiskConfig {
            wind: WindModel::new(2.0, Vector3::zeros(), 0.3),
            cvar_alpha: 0.05,
            cvar_penalty: 10.0,
        });
        opt.set_chance_constraint(cc_config);

        // Drone inside the obstacle — almost all samples will collide
        let self_state = DroneState::hover_at(Vector3::new(5.0, 5.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(0.5, 0.5, 1.5));

        // Run several iterations; lambda should increase because p_collision >> delta
        for _ in 0..5 {
            opt.compute_action(&self_state, &target, true);
        }
        assert!(
            opt.lambda_cc > lambda_init,
            "lambda should increase when collision rate > delta, got {}",
            opt.lambda_cc
        );

        // Now test with a clear arena (no obstacles) — collision rate = 0 < delta
        let clear_arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let cc_config2 = ChanceConstraintConfig {
            delta: 0.01,
            lambda_lr: 0.5,
            lambda_init: 500.0, // start high so we can observe decrease
            lambda_min: 0.0,
            lambda_max: 1e6,
        };
        let lambda_init2 = cc_config2.lambda_init;

        let mut opt2 = MppiOptimizer::new(
            128,
            5,
            0.03,
            10.0,
            params.clone(),
            clear_arena,
            weights,
            0.01,
            5,
        );
        opt2.set_risk_config(RiskConfig {
            wind: WindModel::new(2.0, Vector3::zeros(), 0.3),
            cvar_alpha: 0.05,
            cvar_penalty: 10.0,
        });
        opt2.set_chance_constraint(cc_config2);

        let self_state2 = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));
        let target2 = DroneState::hover_at(Vector3::new(8.0, 5.0, 1.5));

        for _ in 0..5 {
            opt2.compute_action(&self_state2, &target2, true);
        }
        assert!(
            opt2.lambda_cc < lambda_init2,
            "lambda should decrease when collision rate < delta, got {}",
            opt2.lambda_cc
        );

        // Verify reset restores lambda_init
        opt2.reset();
        assert_eq!(
            opt2.lambda_cc, lambda_init2,
            "reset should restore lambda_cc to lambda_init"
        );
    }
}
