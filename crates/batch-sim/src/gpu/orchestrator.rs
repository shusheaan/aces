//! GPU counterpart to `BatchOrchestrator`.
//!
//! Runs N concurrent 1v1 battles. For each step, both drones in every battle
//! have their MPPI optimal mean control computed in a single GPU dispatch
//! (shape `[2*N][H][4]`), and then physics / observations / rewards are run
//! in parallel on the CPU (same code path as `BatchOrchestrator`).
//!
//! This module is feature-gated behind `gpu` along with the rest of
//! `crate::gpu`. State packing, noise generation, and warm-start shifting
//! all happen host-side; the GPU does only the rollout + softmax-weighted
//! mean.

use crate::battle::{BatchConfig, BattleInfo, BattleState, SpawnMode, StepResult};
use crate::f32_dynamics::DroneParamsF32;
use crate::f32_sdf::ArenaF32;
use crate::gpu::pipeline::{CostWeightsGpu, GpuBatchMppi, GpuInitError};
use crate::observation::build_observation;
use crate::reward::{self, RewardConfig};
use aces_sim_core::dynamics::DroneParams;
use aces_sim_core::environment::{Arena, Obstacle};
use aces_sim_core::lockon::LockOnParams;
use aces_sim_core::state::DroneState;
use nalgebra::{Vector3, Vector4};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

/// Number of floats per packed drone state.
const STATE_DIM: usize = 13;

/// Default warehouse arena: 10x10x3m with 5 box pillars. Matches the helpers
/// used across the crate (`f32_sdf::tests::warehouse_arena_f64`,
/// `orchestrator::tests::test_arena`).
fn default_warehouse_arena() -> Arena {
    let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
    for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
        arena.obstacles.push(Obstacle::Box {
            center: Vector3::new(x, y, 1.5),
            half_extents: Vector3::new(0.5, 0.5, 1.5),
        });
    }
    arena
}

/// Pack a 13-element `DroneState` into a flat f32 slice.
///
/// Layout: pos3 (0..3), vel3 (3..6), quat_xyzw (6..10), angvel3 (10..13).
/// Matches the unpack in
/// [`crate::gpu::pipeline::compute_batch_actions_cpu_reference`].
///
/// `dst` must have `len >= 13`.
fn pack_state_into(dst: &mut [f32], state: &DroneState) {
    debug_assert!(dst.len() >= STATE_DIM);
    dst[0] = state.position.x as f32;
    dst[1] = state.position.y as f32;
    dst[2] = state.position.z as f32;
    dst[3] = state.velocity.x as f32;
    dst[4] = state.velocity.y as f32;
    dst[5] = state.velocity.z as f32;
    let q = state.attitude.quaternion();
    dst[6] = q.i as f32;
    dst[7] = q.j as f32;
    dst[8] = q.k as f32;
    dst[9] = q.w as f32;
    dst[10] = state.angular_velocity.x as f32;
    dst[11] = state.angular_velocity.y as f32;
    dst[12] = state.angular_velocity.z as f32;
}

/// GPU-backed batch orchestrator.
///
/// Equivalent to [`crate::orchestrator::BatchOrchestrator`] but uses a single
/// GPU MPPI dispatch per step for all `2 * n_battles` drones instead of
/// `2 * n_battles` CPU MPPI optimizers.
pub struct GpuBatchOrchestrator {
    battles: Vec<BattleState>,
    pipeline: GpuBatchMppi,
    /// Per-battle RNG for physics / reset.
    rngs: Vec<SmallRng>,
    /// Dedicated RNG for GPU noise sampling (kept separate so that changing
    /// `mppi_samples`/`horizon` doesn't perturb the physics stream).
    noise_rng: SmallRng,
    lockon_params: LockOnParams,
    spawn_mode: SpawnMode,

    pub params: DroneParams,
    pub params_f32: DroneParamsF32,
    pub arena: Arena,
    pub arena_f32: ArenaF32,
    pub batch_config: BatchConfig,
    pub reward_config: RewardConfig,

    /// MPPI noise standard deviation used when sampling the per-step noise
    /// buffer uploaded to the GPU.
    noise_std: f32,
    /// Warm-start mean control sequence, shape `[2*n_battles][horizon][4]`
    /// flattened row-major. Written back after each GPU dispatch.
    mean_ctrls: Vec<f32>,
}

impl GpuBatchOrchestrator {
    /// Build a new GPU batch orchestrator.
    ///
    /// Mirrors [`crate::orchestrator::BatchOrchestrator::new`] for the CPU
    /// physics state (arena, drone params, spawn, lockon params) and also
    /// stands up the GPU MPPI pipeline for `2 * n_battles` drones.
    ///
    /// `seed` drives both the per-battle physics RNGs (derived) and the
    /// shared noise RNG for MPPI sampling.
    pub fn new(
        n_battles: usize,
        batch_config: BatchConfig,
        reward_config: RewardConfig,
        mppi_samples: usize,
        mppi_horizon: usize,
        noise_std: f32,
        seed: u64,
    ) -> Result<Self, GpuInitError> {
        let params = DroneParams::crazyflie();
        let arena = default_warehouse_arena();
        let params_f32 = DroneParamsF32::crazyflie();
        let arena_f32 = ArenaF32::from_f64(&arena);
        let lockon_params = LockOnParams::default();

        let weights_gpu = CostWeightsGpu::new(
            1.0,
            5.0,
            0.01,
            1000.0,
            0.3,
            params.hover_thrust() as f32,
            [
                arena.bounds.x as f32,
                arena.bounds.y as f32,
                arena.bounds.z as f32,
            ],
        );

        let pipeline = GpuBatchMppi::new(
            2 * n_battles,
            mppi_samples,
            mppi_horizon,
            &params_f32,
            weights_gpu,
            &arena_f32,
        )?;

        // Seed the top-level RNG used to derive every per-battle RNG and the
        // noise RNG. Deterministic given `seed`.
        let mut master = SmallRng::seed_from_u64(seed);
        let noise_rng = SmallRng::seed_from_u64(master.gen());

        let spawn_mode = SpawnMode::default_for_warehouse();
        let mut battles = Vec::with_capacity(n_battles);
        let mut rngs = Vec::with_capacity(n_battles);
        for _ in 0..n_battles {
            let sub_seed: u64 = master.gen();
            let mut rng = SmallRng::seed_from_u64(sub_seed);
            let battle = BattleState::random_spawn(
                &arena,
                lockon_params.clone(),
                batch_config.wind_sigma,
                batch_config.wind_theta,
                &spawn_mode,
                &mut rng,
            );
            battles.push(battle);
            rngs.push(rng);
        }

        let hover = params.hover_thrust() as f32;
        let mean_ctrls = vec![hover; 2 * n_battles * mppi_horizon * 4];

        Ok(Self {
            battles,
            pipeline,
            rngs,
            noise_rng,
            lockon_params,
            spawn_mode,
            params,
            params_f32,
            arena,
            arena_f32,
            batch_config,
            reward_config,
            noise_std,
            mean_ctrls,
        })
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

    /// Reset all battles to fresh random spawns and reset the warm-start
    /// mean-control buffer back to hover. Returns agent-A observations for
    /// every battle as f32, shape `[n_battles][21]`.
    ///
    /// Intended as the VecEnv-style "initial reset" — no physics step is
    /// advanced; observations are built directly from the newly spawned
    /// states with zero lock-on progress and ground-truth visibility.
    pub fn reset(&mut self) -> Vec<[f32; 21]> {
        let horizon = self.pipeline.horizon;
        let drone_stride = horizon * 4;
        let hover_f32 = self.params.hover_thrust() as f32;

        for (battle, rng) in self.battles.iter_mut().zip(self.rngs.iter_mut()) {
            battle.reset(
                &self.arena,
                &self.lockon_params,
                self.batch_config.wind_sigma,
                self.batch_config.wind_theta,
                &self.spawn_mode,
                rng,
            );
        }

        // Warm-start: reset every drone's horizon to hover.
        for v in self.mean_ctrls.iter_mut() {
            *v = hover_f32;
        }
        debug_assert_eq!(self.mean_ctrls.len(), 2 * self.battles.len() * drone_stride);

        self.battles
            .iter()
            .map(|battle| {
                let visible_ab = aces_sim_core::collision::check_line_of_sight(
                    &self.arena,
                    &battle.state_a.position,
                    &battle.state_b.position,
                ) == aces_sim_core::collision::Visibility::Visible;
                let obs = build_observation(
                    &battle.state_a,
                    &battle.state_b,
                    &self.arena,
                    0.0,
                    0.0,
                    visible_ab,
                );
                let mut arr = [0.0f32; 21];
                for (i, v) in obs.iter().enumerate() {
                    arr[i] = *v as f32;
                }
                arr
            })
            .collect()
    }

    /// MPPI horizon (inherited from the GPU pipeline).
    pub fn horizon(&self) -> usize {
        self.pipeline.horizon
    }

    /// MPPI sample count (inherited from the GPU pipeline).
    pub fn n_samples(&self) -> usize {
        self.pipeline.n_samples
    }

    /// Read-only view on the warm-start mean-control buffer. Shape
    /// `[2*n_battles][horizon][4]` flattened row-major.
    pub fn mean_ctrls(&self) -> &[f32] {
        &self.mean_ctrls
    }

    /// Pack states, sample noise, and dispatch the GPU MPPI pipeline.
    ///
    /// Returns the newly optimized mean control sequences for all `2*n_battles`
    /// drones, shape `[2*n_battles][horizon][4]` flattened row-major. Shared
    /// between [`Self::step_all`] and [`Self::step_with_agent_a_actions`] so
    /// the dispatch logic stays in one place.
    fn pack_and_dispatch_gpu_mppi(&mut self) -> Vec<f32> {
        let n_battles = self.battles.len();
        let n_drones = 2 * n_battles;
        let horizon = self.pipeline.horizon;
        let n_samples = self.pipeline.n_samples;

        // 1. Pack current states + paired enemy states into flat f32 buffers.
        let mut states = vec![0.0f32; n_drones * STATE_DIM];
        let mut enemies = vec![0.0f32; n_drones * STATE_DIM];
        for (b, battle) in self.battles.iter().enumerate() {
            let a_idx = 2 * b;
            let b_idx = 2 * b + 1;
            pack_state_into(
                &mut states[a_idx * STATE_DIM..(a_idx + 1) * STATE_DIM],
                &battle.state_a,
            );
            pack_state_into(
                &mut states[b_idx * STATE_DIM..(b_idx + 1) * STATE_DIM],
                &battle.state_b,
            );
            pack_state_into(
                &mut enemies[a_idx * STATE_DIM..(a_idx + 1) * STATE_DIM],
                &battle.state_b,
            );
            pack_state_into(
                &mut enemies[b_idx * STATE_DIM..(b_idx + 1) * STATE_DIM],
                &battle.state_a,
            );
        }

        // 2. Sample Gaussian noise.
        let noise_len = n_drones * n_samples * horizon * 4;
        let normal = Normal::new(0.0f32, self.noise_std).expect("valid noise_std");
        let mut noise = Vec::with_capacity(noise_len);
        for _ in 0..noise_len {
            noise.push(normal.sample(&mut self.noise_rng));
        }

        // 3. Extract current per-drone wind (held constant across the rollout
        //    horizon — see `mppi_rollout.wgsl` header). Drone order matches
        //    state packing: `[battle0_A, battle0_B, battle1_A, battle1_B, ...]`.
        let winds: Vec<[f32; 3]> = self
            .battles
            .iter()
            .flat_map(|b| {
                let wa = b.wind_a.force;
                let wb = b.wind_b.force;
                [
                    [wa.x as f32, wa.y as f32, wa.z as f32],
                    [wb.x as f32, wb.y as f32, wb.z as f32],
                ]
            })
            .collect();
        self.pipeline.set_wind(&winds);

        // 4. One GPU dispatch, producing new optimal mean control sequences.
        self.pipeline
            .compute_batch_actions(&states, &enemies, &self.mean_ctrls, &noise)
    }

    /// Run physics + observations + rewards for every battle in parallel
    /// given per-battle agent-A and agent-B motor commands, then warm-start
    /// shift and reset terminated battles.
    ///
    /// `new_mean` is used for warm-start bookkeeping (the GPU-optimized
    /// sequences). `motors_a_vec` / `motors_b_vec` are the actual motor
    /// commands applied to physics — these may differ from `new_mean[.., 0]`
    /// when an external agent overrides one side (PPO-vs-MPPI mode).
    fn apply_physics_and_warm_start(
        &mut self,
        new_mean: Vec<f32>,
        motors_a_vec: Vec<Vector4<f64>>,
        motors_b_vec: Vec<Vector4<f64>>,
    ) -> Vec<StepResult> {
        let n_battles = self.battles.len();
        let n_drones = 2 * n_battles;
        let horizon = self.pipeline.horizon;
        let drone_stride = horizon * 4;

        let arena = &self.arena;
        let params = &self.params;
        let batch_config = &self.batch_config;
        let reward_config = &self.reward_config;
        let hover_thrust = params.hover_thrust();

        let results: Vec<(StepResult, bool)> = self
            .battles
            .par_iter_mut()
            .zip(self.rngs.par_iter_mut())
            .enumerate()
            .map(|(i, (battle, rng))| {
                let motors_a = motors_a_vec[i];
                let motors_b = motors_b_vec[i];

                let prev_distance = battle.prev_distance;
                let prev_lock_a = battle.prev_lock_progress_a;

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

                battle.prev_distance = info.distance;
                battle.prev_lock_progress_a = info.lock_progress_a;

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

        // Warm-start shift: drop the applied step, append hover. Matches the
        // CPU MPPI behavior in `optimizer.rs`.
        let hover_f32 = hover_thrust as f32;
        for d in 0..n_drones {
            let base = d * drone_stride;
            for h in 0..horizon - 1 {
                let src = base + (h + 1) * 4;
                let dst = base + h * 4;
                self.mean_ctrls[dst] = new_mean[src];
                self.mean_ctrls[dst + 1] = new_mean[src + 1];
                self.mean_ctrls[dst + 2] = new_mean[src + 2];
                self.mean_ctrls[dst + 3] = new_mean[src + 3];
            }
            let last = base + (horizon - 1) * 4;
            self.mean_ctrls[last] = hover_f32;
            self.mean_ctrls[last + 1] = hover_f32;
            self.mean_ctrls[last + 2] = hover_f32;
            self.mean_ctrls[last + 3] = hover_f32;
        }

        // Reset terminated battles and their warm-start slots (sequential —
        // resets need the per-battle RNG).
        let mut step_results = Vec::with_capacity(results.len());
        for (i, (result, done)) in results.into_iter().enumerate() {
            if done {
                self.battles[i].reset(
                    &self.arena,
                    &self.lockon_params,
                    self.batch_config.wind_sigma,
                    self.batch_config.wind_theta,
                    &self.spawn_mode,
                    &mut self.rngs[i],
                );
                for d_offset in 0..2 {
                    let base = (2 * i + d_offset) * drone_stride;
                    for slot in 0..horizon {
                        let s = base + slot * 4;
                        self.mean_ctrls[s] = hover_f32;
                        self.mean_ctrls[s + 1] = hover_f32;
                        self.mean_ctrls[s + 2] = hover_f32;
                        self.mean_ctrls[s + 3] = hover_f32;
                    }
                }
            }
            step_results.push(result);
        }

        step_results
    }

    /// Step all battles once: GPU MPPI action selection → parallel physics →
    /// observations + rewards → per-battle reset on termination.
    ///
    /// Returns one `StepResult` per battle.
    pub fn step_all(&mut self) -> Vec<StepResult> {
        let n_battles = self.battles.len();
        let horizon = self.pipeline.horizon;
        let drone_stride = horizon * 4;

        // Pack states, sample noise, dispatch GPU MPPI.
        let new_mean = self.pack_and_dispatch_gpu_mppi();

        // Pull motor[0] out of each drone's returned horizon. This is the
        // action applied at the current control tick.
        let mut motors_a_vec: Vec<Vector4<f64>> = Vec::with_capacity(n_battles);
        let mut motors_b_vec: Vec<Vector4<f64>> = Vec::with_capacity(n_battles);
        for b in 0..n_battles {
            let base_a = 2 * b * drone_stride;
            let base_b = (2 * b + 1) * drone_stride;
            motors_a_vec.push(Vector4::new(
                new_mean[base_a] as f64,
                new_mean[base_a + 1] as f64,
                new_mean[base_a + 2] as f64,
                new_mean[base_a + 3] as f64,
            ));
            motors_b_vec.push(Vector4::new(
                new_mean[base_b] as f64,
                new_mean[base_b + 1] as f64,
                new_mean[base_b + 2] as f64,
                new_mean[base_b + 3] as f64,
            ));
        }

        self.apply_physics_and_warm_start(new_mean, motors_a_vec, motors_b_vec)
    }

    /// Step all battles using external actions for agent A (learning agent)
    /// and GPU MPPI for agent B (opponent).
    ///
    /// `actions_a` must contain one `[f32; 4]` motor command per battle, in
    /// the same order as the battles vector. Values are expected in
    /// `[0, max_thrust]`; the method clamps into that range before stepping
    /// physics. The full GPU MPPI pipeline still runs (so agent B's plan
    /// conditions on agent A's true state via the enemy buffer), but the
    /// resulting action for agent A is discarded in favor of `actions_a[i]`.
    ///
    /// Panics if `actions_a.len() != self.n_battles()`.
    pub fn step_with_agent_a_actions(&mut self, actions_a: &[[f32; 4]]) -> Vec<StepResult> {
        assert_eq!(
            actions_a.len(),
            self.battles.len(),
            "actions_a len {} != n_battles {}",
            actions_a.len(),
            self.battles.len()
        );

        let n_battles = self.battles.len();
        let horizon = self.pipeline.horizon;
        let drone_stride = horizon * 4;
        let max_thrust = self.params.max_thrust;

        // Pack states, sample noise, dispatch GPU MPPI. Agent A's output
        // from the GPU is discarded below, but we still dispatch for all
        // 2N drones so that agent B's rollouts include agent A's true
        // current state via the enemy buffer.
        let new_mean = self.pack_and_dispatch_gpu_mppi();

        let mut motors_a_vec: Vec<Vector4<f64>> = Vec::with_capacity(n_battles);
        let mut motors_b_vec: Vec<Vector4<f64>> = Vec::with_capacity(n_battles);
        for (b, a) in actions_a.iter().enumerate() {
            let base_b = (2 * b + 1) * drone_stride;
            // Agent A: external action, clamped to [0, max_thrust].
            motors_a_vec.push(Vector4::new(
                (a[0] as f64).clamp(0.0, max_thrust),
                (a[1] as f64).clamp(0.0, max_thrust),
                (a[2] as f64).clamp(0.0, max_thrust),
                (a[3] as f64).clamp(0.0, max_thrust),
            ));
            // Agent B: GPU MPPI output.
            motors_b_vec.push(Vector4::new(
                new_mean[base_b] as f64,
                new_mean[base_b + 1] as f64,
                new_mean[base_b + 2] as f64,
                new_mean[base_b + 3] as f64,
            ));
        }

        self.apply_physics_and_warm_start(new_mean, motors_a_vec, motors_b_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::adapter::probe_gpu;

    fn gpu_available_or_skip(name: &str) -> bool {
        let probe = probe_gpu();
        if !probe.compute_capable {
            eprintln!("[{name}] skipped: no GPU adapter available");
            return false;
        }
        true
    }

    fn small_batch_config() -> BatchConfig {
        BatchConfig {
            max_steps: 500,
            ..Default::default()
        }
    }

    #[test]
    fn test_pack_state_into_identity() {
        // Pure CPU check — no GPU needed.
        let state = DroneState::hover_at(Vector3::new(1.0, 2.0, 3.0));
        let mut dst = [0.0f32; 13];
        pack_state_into(&mut dst, &state);
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], 2.0);
        assert_eq!(dst[2], 3.0);
        for v in &dst[3..6] {
            assert_eq!(*v, 0.0);
        }
        // Identity quaternion: xyzw = (0, 0, 0, 1).
        assert_eq!(dst[6], 0.0);
        assert_eq!(dst[7], 0.0);
        assert_eq!(dst[8], 0.0);
        assert_eq!(dst[9], 1.0);
        for v in &dst[10..13] {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_gpu_orchestrator_constructs() {
        if !gpu_available_or_skip("test_gpu_orchestrator_constructs") {
            return;
        }
        let orch = GpuBatchOrchestrator::new(
            4,
            small_batch_config(),
            RewardConfig::default(),
            32,
            5,
            0.03,
            42,
        )
        .expect("gpu orchestrator builds");
        assert_eq!(orch.n_battles(), 4);
        assert_eq!(orch.horizon(), 5);
        assert_eq!(orch.n_samples(), 32);
        // 2 drones × 4 battles × 5 horizon × 4 motors.
        assert_eq!(orch.mean_ctrls().len(), 2 * 4 * 5 * 4);
    }

    #[test]
    fn test_gpu_orchestrator_runs_one_step() {
        if !gpu_available_or_skip("test_gpu_orchestrator_runs_one_step") {
            return;
        }
        let mut orch = GpuBatchOrchestrator::new(
            4,
            small_batch_config(),
            RewardConfig::default(),
            32,
            5,
            0.03,
            42,
        )
        .expect("gpu orchestrator builds");
        let results = orch.step_all();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert_eq!(r.obs_a.len(), 21);
            assert_eq!(r.obs_b.len(), 21);
            for m in &r.motors_a {
                assert!(m.is_finite(), "motor A non-finite: {m}");
            }
            for m in &r.motors_b {
                assert!(m.is_finite(), "motor B non-finite: {m}");
            }
        }
    }

    #[test]
    fn test_gpu_orchestrator_runs_100_steps() {
        if !gpu_available_or_skip("test_gpu_orchestrator_runs_100_steps") {
            return;
        }
        let mut orch = GpuBatchOrchestrator::new(
            4,
            small_batch_config(),
            RewardConfig::default(),
            32,
            5,
            0.03,
            7,
        )
        .expect("gpu orchestrator builds");

        for step in 0..100 {
            let results = orch.step_all();
            assert_eq!(results.len(), 4, "step {step}");
            for (bi, battle) in orch.battles.iter().enumerate() {
                let pos_a = &battle.state_a.position;
                let pos_b = &battle.state_b.position;
                assert!(
                    pos_a.x.is_finite() && pos_a.y.is_finite() && pos_a.z.is_finite(),
                    "NaN in state_a at step {step}, battle {bi}: {pos_a:?}",
                );
                assert!(
                    pos_b.x.is_finite() && pos_b.y.is_finite() && pos_b.z.is_finite(),
                    "NaN in state_b at step {step}, battle {bi}: {pos_b:?}",
                );
            }
            // mean_ctrls must stay finite and within a very generous bound.
            for &c in orch.mean_ctrls() {
                assert!(c.is_finite(), "mean_ctrl non-finite at step {step}: {c}");
                assert!(c.abs() < 100.0, "mean_ctrl exploded at step {step}: {c}",);
            }
        }
    }

    #[test]
    fn test_gpu_orchestrator_reset_returns_observations() {
        if !gpu_available_or_skip("test_gpu_orchestrator_reset_returns_observations") {
            return;
        }
        let mut orch = GpuBatchOrchestrator::new(
            3,
            small_batch_config(),
            RewardConfig::default(),
            32,
            5,
            0.03,
            99,
        )
        .expect("gpu orchestrator builds");

        // Run a few steps so state + warm-start drift away from hover.
        for _ in 0..5 {
            let _ = orch.step_all();
        }
        let drifted: Vec<f32> = orch.mean_ctrls().to_vec();

        // Reset: should return one obs per battle, all finite, and zero
        // the lock-on + time-since-seen + belief-var slots.
        let obs = orch.reset();
        assert_eq!(obs.len(), 3);
        for o in &obs {
            assert_eq!(o.len(), 21);
            for v in o {
                assert!(v.is_finite(), "reset obs non-finite: {v}");
            }
            // lock_a = obs[16], lock_b = obs[17] must be 0 after reset
            assert_eq!(o[16], 0.0);
            assert_eq!(o[17], 0.0);
            // time_since_last_seen = obs[20] must be 0
            assert_eq!(o[20], 0.0);
        }
        // Warm-start must be hover thrust everywhere after reset.
        let hover = orch.params.hover_thrust() as f32;
        for v in orch.mean_ctrls() {
            assert!((*v - hover).abs() < 1e-6, "warm-start not hover: {v}");
        }
        // Sanity: warm-start actually changed across reset (otherwise the
        // test proves nothing).
        assert_ne!(drifted, orch.mean_ctrls().to_vec());
        // Battles must be live.
        for battle in &orch.battles {
            assert!(!battle.done);
            assert_eq!(battle.step_count, 0);
        }
    }

    #[test]
    fn test_gpu_orchestrator_step_with_agent_a_actions() {
        if !gpu_available_or_skip("test_gpu_orchestrator_step_with_agent_a_actions") {
            return;
        }
        let mut orch = GpuBatchOrchestrator::new(
            4,
            small_batch_config(),
            RewardConfig::default(),
            32,
            5,
            0.03,
            13,
        )
        .expect("gpu orchestrator builds");

        let hover = orch.params.hover_thrust() as f32;
        let actions_a: Vec<[f32; 4]> = vec![[hover; 4]; orch.n_battles()];

        for step in 0..10 {
            let results = orch.step_with_agent_a_actions(&actions_a);
            assert_eq!(results.len(), 4, "step {step}");
            for (bi, r) in results.iter().enumerate() {
                for m in &r.motors_a {
                    assert!(
                        m.is_finite(),
                        "motor A non-finite at step {step}, battle {bi}: {m}"
                    );
                    // Agent A's applied motors must equal the hover input
                    // (up to the f32 -> f64 cast).
                    assert!(
                        (*m - hover as f64).abs() < 1e-5,
                        "motor A not hover at step {step}, battle {bi}: {m}",
                    );
                }
                for m in &r.motors_b {
                    assert!(
                        m.is_finite(),
                        "motor B non-finite at step {step}, battle {bi}: {m}"
                    );
                }
            }
            for (bi, battle) in orch.battles.iter().enumerate() {
                let pos_a = &battle.state_a.position;
                let pos_b = &battle.state_b.position;
                assert!(
                    pos_a.x.is_finite() && pos_a.y.is_finite() && pos_a.z.is_finite(),
                    "NaN in state_a at step {step}, battle {bi}: {pos_a:?}",
                );
                assert!(
                    pos_b.x.is_finite() && pos_b.y.is_finite() && pos_b.z.is_finite(),
                    "NaN in state_b at step {step}, battle {bi}: {pos_b:?}",
                );
            }
        }
    }

    #[test]
    fn test_gpu_orchestrator_panics_on_wrong_actions_a_len() {
        if !gpu_available_or_skip("test_gpu_orchestrator_panics_on_wrong_actions_a_len") {
            return;
        }
        // Can't wrap the orchestrator construction in catch_unwind safely
        // (AssertUnwindSafe on a struct holding a wgpu device is dicey), so
        // build first and then assert that the method call panics.
        let mut orch = GpuBatchOrchestrator::new(
            3,
            small_batch_config(),
            RewardConfig::default(),
            32,
            5,
            0.03,
            17,
        )
        .expect("gpu orchestrator builds");

        let hover = orch.params.hover_thrust() as f32;
        // Wrong length: 2 instead of 3.
        let wrong_actions: Vec<[f32; 4]> = vec![[hover; 4]; 2];

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            orch.step_with_agent_a_actions(&wrong_actions);
        }));
        assert!(
            result.is_err(),
            "expected panic on wrong actions_a length, but call succeeded",
        );
    }

    #[test]
    fn test_gpu_orchestrator_episode_completes() {
        if !gpu_available_or_skip("test_gpu_orchestrator_episode_completes") {
            return;
        }
        // max_steps=50 now drives timeout termination (battle.rs respects
        // BatchConfig::max_steps). Termination may come via either collision
        // (random spawns + MPPI noise) or the configured timeout, whichever
        // fires first.
        let batch_config = BatchConfig {
            max_steps: 50,
            ..Default::default()
        };
        let mut orch =
            GpuBatchOrchestrator::new(4, batch_config, RewardConfig::default(), 32, 5, 0.03, 11)
                .expect("gpu orchestrator builds");

        let mut saw_done = false;
        let mut steps_after_done = 0;
        for _ in 0..500 {
            let results = orch.step_all();
            for r in &results {
                if r.done {
                    saw_done = true;
                }
            }
            if saw_done {
                steps_after_done += 1;
                if steps_after_done >= 5 {
                    break;
                }
            }
        }
        assert!(saw_done, "no battle reached a done condition in 500 steps");

        // After reset, every battle should be live again with finite state.
        for battle in &orch.battles {
            assert!(!battle.done);
            let pos_a = &battle.state_a.position;
            let pos_b = &battle.state_b.position;
            assert!(pos_a.x.is_finite() && pos_a.y.is_finite() && pos_a.z.is_finite());
            assert!(pos_b.x.is_finite() && pos_b.y.is_finite() && pos_b.z.is_finite());
        }
    }
}
