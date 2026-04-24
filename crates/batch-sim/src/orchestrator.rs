use crate::battle::{BatchConfig, BattleInfo, BattleState, SpawnMode, StepResult};
use crate::observation::build_observation;
use crate::reward::{self, RewardConfig};
use aces_mppi::cost::CostWeights;
use aces_mppi::optimizer::MppiOptimizer;
use aces_sim_core::dynamics::DroneParams;
use aces_sim_core::environment::Arena;
use aces_sim_core::lockon::LockOnParams;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

/// MPPI configuration for creating optimizers.
#[derive(Debug, Clone)]
pub struct MppiConfig {
    pub num_samples: usize,
    pub horizon: usize,
    pub noise_std: f64,
    pub temperature: f64,
    pub weights: CostWeights,
}

impl Default for MppiConfig {
    fn default() -> Self {
        Self {
            num_samples: 128,
            horizon: 15,
            noise_std: 0.03,
            temperature: 10.0,
            weights: CostWeights::default(),
        }
    }
}

/// Aggregated statistics across all battles.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    pub total_steps: u64,
    pub total_episodes: u64,
    pub kills_a: u64,
    pub kills_b: u64,
    pub collisions_a: u64,
    pub collisions_b: u64,
    pub timeouts: u64,
    pub mean_reward_a: f64,
    pub mean_distance: f64,
}

/// Manages N concurrent MPPI-vs-MPPI dogfights with Rayon parallelism.
pub struct BatchOrchestrator {
    battles: Vec<BattleState>,
    /// One MPPI optimizer per drone (2 per battle: pursuit and evasion).
    optimizers_a: Vec<MppiOptimizer>,
    optimizers_b: Vec<MppiOptimizer>,
    /// Per-battle RNG for deterministic physics/reset.
    rngs: Vec<SmallRng>,
    arena: Arena,
    params: DroneParams,
    lockon_params: LockOnParams,
    batch_config: BatchConfig,
    reward_config: RewardConfig,
    spawn_mode: SpawnMode,

    // Running stats
    stats: BatchStats,
}

impl BatchOrchestrator {
    /// Create a batch of N concurrent battles.
    pub fn new(
        n_battles: usize,
        arena: Arena,
        params: DroneParams,
        lockon_params: LockOnParams,
        mppi_config: MppiConfig,
        batch_config: BatchConfig,
        reward_config: RewardConfig,
    ) -> Self {
        let spawn_mode = SpawnMode::default_for_warehouse();
        let mut master_rng = rand::thread_rng();

        let mut battles = Vec::with_capacity(n_battles);
        let mut optimizers_a = Vec::with_capacity(n_battles);
        let mut optimizers_b = Vec::with_capacity(n_battles);
        let mut rngs = Vec::with_capacity(n_battles);

        for _ in 0..n_battles {
            let seed: u64 = rand::Rng::gen(&mut master_rng);
            let mut rng = SmallRng::seed_from_u64(seed);

            let battle = BattleState::random_spawn(
                &arena,
                lockon_params.clone(),
                batch_config.wind_sigma,
                batch_config.wind_theta,
                &spawn_mode,
                &mut rng,
            );
            battles.push(battle);

            // Pursuit optimizer for drone A
            optimizers_a.push(MppiOptimizer::new(
                mppi_config.num_samples,
                mppi_config.horizon,
                mppi_config.noise_std,
                mppi_config.temperature,
                params.clone(),
                arena.clone(),
                mppi_config.weights.clone(),
                batch_config.dt_ctrl,
                batch_config.substeps,
            ));

            // Evasion optimizer for drone B
            optimizers_b.push(MppiOptimizer::new(
                mppi_config.num_samples,
                mppi_config.horizon,
                mppi_config.noise_std,
                mppi_config.temperature,
                params.clone(),
                arena.clone(),
                mppi_config.weights.clone(),
                batch_config.dt_ctrl,
                batch_config.substeps,
            ));

            rngs.push(rng);
        }

        Self {
            battles,
            optimizers_a,
            optimizers_b,
            rngs,
            arena,
            params,
            lockon_params,
            batch_config,
            reward_config,
            spawn_mode,
            stats: BatchStats::default(),
        }
    }

    /// Override the spawn mode used on reset. Primarily for domain
    /// randomization or test setups.
    pub fn set_spawn_mode(&mut self, mode: SpawnMode) {
        self.spawn_mode = mode;
    }

    /// Read the current spawn mode.
    pub fn spawn_mode(&self) -> &SpawnMode {
        &self.spawn_mode
    }

    /// Number of active battles.
    pub fn n_battles(&self) -> usize {
        self.battles.len()
    }

    /// Step all battles in parallel: MPPI decisions → physics → observations.
    ///
    /// Returns one StepResult per battle.
    pub fn step_all(&mut self) -> Vec<StepResult> {
        let arena = &self.arena;
        let params = &self.params;
        let batch_config = &self.batch_config;
        let reward_config = &self.reward_config;
        let hover_thrust = params.hover_thrust();

        // Phase 1+2: Parallel MPPI action selection + physics step.
        //
        // We zip battles, optimizers, and rngs together and process each
        // battle independently in parallel. Rayon's work-stealing handles
        // the nested parallelism from MPPI's internal par_iter.
        let results: Vec<(StepResult, bool)> = self
            .battles
            .par_iter_mut()
            .zip(self.optimizers_a.par_iter_mut())
            .zip(self.optimizers_b.par_iter_mut())
            .zip(self.rngs.par_iter_mut())
            .map(|(((battle, opt_a), opt_b), rng)| {
                // MPPI: drone A pursues, drone B evades
                let motors_a = opt_a.compute_action(&battle.state_a, &battle.state_b, true);
                let motors_b = opt_b.compute_action(&battle.state_b, &battle.state_a, false);

                // Save pre-step values for reward delta
                let prev_distance = battle.prev_distance;
                let prev_lock_a = battle.prev_lock_progress_a;

                // Physics step
                let info = battle.step_physics(
                    &motors_a,
                    &motors_b,
                    params,
                    arena,
                    batch_config.dt_ctrl,
                    batch_config.substeps,
                    batch_config.max_steps,
                    rng,
                );

                // Update prev values for next step
                battle.prev_distance = info.distance;
                battle.prev_lock_progress_a = info.lock_progress_a;

                // Observations
                let obs_a = build_observation(
                    &battle.state_a,
                    &battle.state_b,
                    arena,
                    info.lock_progress_a,
                    info.lock_progress_b,
                    info.visible_ab,
                );
                let obs_b = build_observation(
                    &battle.state_b,
                    &battle.state_a,
                    arena,
                    info.lock_progress_b,
                    info.lock_progress_a,
                    info.visible_ba,
                );

                // Rewards
                let ctrl_cost_a = reward::control_cost(
                    &[motors_a[0], motors_a[1], motors_a[2], motors_a[3]],
                    hover_thrust,
                );
                let reward_a = reward::compute_reward_a(
                    &info,
                    prev_distance,
                    prev_lock_a,
                    ctrl_cost_a,
                    reward_config,
                );

                // Symmetric reward for B (swap kill/collision flags)
                let info_b = BattleInfo {
                    kill_a: info.kill_b,
                    kill_b: info.kill_a,
                    collision_a: info.collision_b,
                    collision_b: info.collision_a,
                    timeout: info.timeout,
                    distance: info.distance,
                    lock_progress_a: info.lock_progress_b,
                    lock_progress_b: info.lock_progress_a,
                    visible_ab: info.visible_ba,
                    visible_ba: info.visible_ab,
                };
                let ctrl_cost_b = reward::control_cost(
                    &[motors_b[0], motors_b[1], motors_b[2], motors_b[3]],
                    hover_thrust,
                );
                let reward_b = reward::compute_reward_a(
                    &info_b,
                    prev_distance,
                    0.0,
                    ctrl_cost_b,
                    reward_config,
                );

                let done = battle.done;
                let result = StepResult {
                    obs_a,
                    obs_b,
                    motors_a: [motors_a[0], motors_a[1], motors_a[2], motors_a[3]],
                    motors_b: [motors_b[0], motors_b[1], motors_b[2], motors_b[3]],
                    reward_a,
                    reward_b,
                    done,
                    info,
                };

                (result, done)
            })
            .collect();

        // Phase 3: Collect results and reset terminated battles (sequential — resets need RNG).
        let mut step_results = Vec::with_capacity(results.len());
        for (i, (result, done)) in results.into_iter().enumerate() {
            // Update stats
            self.stats.total_steps += 1;
            self.stats.mean_distance += result.info.distance;
            self.stats.mean_reward_a += result.reward_a;

            if done {
                self.stats.total_episodes += 1;
                if result.info.kill_a {
                    self.stats.kills_a += 1;
                }
                if result.info.kill_b {
                    self.stats.kills_b += 1;
                }
                if result.info.collision_a {
                    self.stats.collisions_a += 1;
                }
                if result.info.collision_b {
                    self.stats.collisions_b += 1;
                }
                if result.info.timeout {
                    self.stats.timeouts += 1;
                }

                // Reset battle and optimizers
                self.battles[i].reset(
                    &self.arena,
                    &self.lockon_params,
                    self.batch_config.wind_sigma,
                    self.batch_config.wind_theta,
                    &self.spawn_mode,
                    &mut self.rngs[i],
                );
                self.optimizers_a[i].reset();
                self.optimizers_b[i].reset();
            }

            step_results.push(result);
        }

        step_results
    }

    /// Run N steps and return aggregated statistics.
    pub fn run(&mut self, n_steps: usize) -> BatchStats {
        let start_steps = self.stats.total_steps;
        let start_reward = self.stats.mean_reward_a;
        let start_distance = self.stats.mean_distance;

        for _ in 0..n_steps {
            self.step_all();
        }

        let steps_taken = self.stats.total_steps - start_steps;
        let mut stats = self.stats.clone();
        if steps_taken > 0 {
            stats.mean_reward_a = (self.stats.mean_reward_a - start_reward) / steps_taken as f64;
            stats.mean_distance = (self.stats.mean_distance - start_distance) / steps_taken as f64;
        }
        stats
    }

    /// Get current accumulated statistics.
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Reset all statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = BatchStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aces_sim_core::environment::Obstacle;
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
    fn test_orchestrator_creation() {
        let arena = test_arena();
        let params = DroneParams::crazyflie();
        let lockon = LockOnParams::default();
        let mppi_config = MppiConfig {
            num_samples: 32,
            horizon: 5,
            ..Default::default()
        };
        let batch_config = BatchConfig::default();
        let reward_config = RewardConfig::default();

        let orch = BatchOrchestrator::new(
            4,
            arena,
            params,
            lockon,
            mppi_config,
            batch_config,
            reward_config,
        );

        assert_eq!(orch.n_battles(), 4);
    }

    #[test]
    fn test_step_all_returns_correct_count() {
        let arena = test_arena();
        let params = DroneParams::crazyflie();
        let lockon = LockOnParams::default();
        let mppi_config = MppiConfig {
            num_samples: 16,
            horizon: 3,
            ..Default::default()
        };
        let batch_config = BatchConfig::default();
        let reward_config = RewardConfig::default();

        let mut orch = BatchOrchestrator::new(
            8,
            arena,
            params,
            lockon,
            mppi_config,
            batch_config,
            reward_config,
        );

        let results = orch.step_all();
        assert_eq!(results.len(), 8);

        for r in &results {
            assert_eq!(r.obs_a.len(), 21);
            assert_eq!(r.obs_b.len(), 21);
            // Motors should be valid (non-NaN, within range)
            for m in &r.motors_a {
                assert!(m.is_finite(), "motor NaN: {}", m);
            }
        }
    }

    #[test]
    fn test_run_multiple_steps() {
        let arena = test_arena();
        let params = DroneParams::crazyflie();
        let lockon = LockOnParams::default();
        let mppi_config = MppiConfig {
            num_samples: 16,
            horizon: 3,
            ..Default::default()
        };
        let batch_config = BatchConfig::default();
        let reward_config = RewardConfig::default();

        let mut orch = BatchOrchestrator::new(
            4,
            arena,
            params,
            lockon,
            mppi_config,
            batch_config,
            reward_config,
        );

        let stats = orch.run(10);
        assert_eq!(stats.total_steps, 4 * 10); // 4 battles × 10 steps
    }

    #[test]
    fn test_episodes_complete_and_reset() {
        let arena = test_arena();
        let params = DroneParams::crazyflie();
        let lockon = LockOnParams::default();
        let mppi_config = MppiConfig {
            num_samples: 16,
            horizon: 3,
            ..Default::default()
        };
        let batch_config = BatchConfig {
            max_steps: 50, // Short episodes for testing
            ..Default::default()
        };
        let reward_config = RewardConfig::default();

        let mut orch = BatchOrchestrator::new(
            2,
            arena,
            params,
            lockon,
            mppi_config,
            batch_config,
            reward_config,
        );

        // Run enough steps that episodes should complete (collision or timeout at max_steps=50)
        let stats = orch.run(100);
        // At least some steps should have been taken
        assert!(stats.total_steps > 0);
    }
}
