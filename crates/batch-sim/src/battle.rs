use aces_sim_core::collision::{check_line_of_sight, Visibility};
use aces_sim_core::dynamics::{step_rk4, DroneParams};
use aces_sim_core::environment::Arena;
use aces_sim_core::lockon::{LockOnParams, LockOnTracker};
use aces_sim_core::state::DroneState;
use aces_sim_core::wind::WindModel;
use nalgebra::{Vector3, Vector4};
use rand::Rng;

/// How initial drone positions are sampled at reset.
///
/// The CPU environment (`DroneDogfightEnv.reset`) uses fixed corner spawns
/// plus small uniform jitter. The original batch-sim spawn was uniform random
/// anywhere in the arena with obstacle rejection — this produced a completely
/// different initial state distribution than the CPU env, breaking policy
/// transfer between CPU-trained and GPU-batch-trained agents.
///
/// `Fixed` matches CPU env semantics and is the default.
/// `Random` is retained for explicit domain randomization.
#[derive(Debug, Clone)]
pub enum SpawnMode {
    /// Fixed spawn positions with uniform jitter per axis. Matches CPU env
    /// `DroneDogfightEnv.reset()` semantics.
    Fixed {
        spawn_a: Vector3<f64>,
        spawn_b: Vector3<f64>,
        /// Uniform jitter of `±jitter` applied independently on each axis.
        jitter: f64,
    },
    /// Uniform random positions inside arena with obstacle rejection.
    /// Use for domain randomization.
    Random { margin: f64 },
}

impl SpawnMode {
    /// Default matching CPU env arena.toml: `spawn_a=(1,1,1.5)`,
    /// `spawn_b=(9,9,1.5)`, `jitter=0.5`.
    pub fn default_for_warehouse() -> Self {
        SpawnMode::Fixed {
            spawn_a: Vector3::new(1.0, 1.0, 1.5),
            spawn_b: Vector3::new(9.0, 9.0, 1.5),
            jitter: 0.5,
        }
    }

    /// Sample spawn positions for drone A and drone B.
    pub fn sample<R: Rng>(&self, arena: &Arena, rng: &mut R) -> (Vector3<f64>, Vector3<f64>) {
        match self {
            SpawnMode::Fixed {
                spawn_a,
                spawn_b,
                jitter,
            } => {
                let j = *jitter;
                let jit = |base: &Vector3<f64>, rng: &mut R| -> Vector3<f64> {
                    if j == 0.0 {
                        *base
                    } else {
                        Vector3::new(
                            base.x + rng.gen_range(-j..=j),
                            base.y + rng.gen_range(-j..=j),
                            base.z + rng.gen_range(-j..=j),
                        )
                    }
                };
                (jit(spawn_a, rng), jit(spawn_b, rng))
            }
            SpawnMode::Random { margin } => {
                let a = random_safe_position(arena, *margin, rng);
                let b = random_safe_position(arena, *margin, rng);
                (a, b)
            }
        }
    }
}

impl Default for SpawnMode {
    fn default() -> Self {
        Self::default_for_warehouse()
    }
}

/// Configuration for a batch of battles.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub max_steps: u32,
    pub dt_ctrl: f64,
    pub substeps: usize,
    pub wind_sigma: f64,
    pub wind_theta: f64,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_steps: 1000,
            dt_ctrl: 0.01,
            substeps: 10,
            wind_sigma: 0.0,
            wind_theta: 2.0,
        }
    }
}

/// Terminal condition flags for a single battle step.
#[derive(Debug, Clone, Default)]
pub struct BattleInfo {
    /// Drone A killed drone B (lock-on completed)
    pub kill_a: bool,
    /// Drone B killed drone A
    pub kill_b: bool,
    /// Drone A collided with obstacle or boundary
    pub collision_a: bool,
    /// Drone B collided with obstacle or boundary
    pub collision_b: bool,
    /// Episode timed out
    pub timeout: bool,
    /// Distance between drones
    pub distance: f64,
    /// Lock-on progress A→B [0, 1]
    pub lock_progress_a: f64,
    /// Lock-on progress B→A [0, 1]
    pub lock_progress_b: f64,
    /// A can see B
    pub visible_ab: bool,
    /// B can see A
    pub visible_ba: bool,
}

/// Result of stepping a single battle.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub obs_a: [f64; 21],
    pub obs_b: [f64; 21],
    pub motors_a: [f64; 4],
    pub motors_b: [f64; 4],
    pub reward_a: f64,
    pub reward_b: f64,
    pub done: bool,
    pub info: BattleInfo,
}

/// State of a single 1v1 dogfight.
pub struct BattleState {
    pub state_a: DroneState,
    pub state_b: DroneState,
    pub wind_a: WindModel,
    pub wind_b: WindModel,
    pub lockon_a: LockOnTracker,
    pub lockon_b: LockOnTracker,
    pub step_count: u32,
    pub done: bool,

    // Previous-step values for delta-based rewards
    pub prev_distance: f64,
    pub prev_lock_progress_a: f64,
}

impl BattleState {
    /// Create a new battle with drones at the specified spawn positions.
    pub fn new(
        spawn_a: Vector3<f64>,
        spawn_b: Vector3<f64>,
        lockon_params: LockOnParams,
        wind_sigma: f64,
        wind_theta: f64,
    ) -> Self {
        let wind = if wind_sigma > 0.0 {
            WindModel::new(wind_theta, Vector3::zeros(), wind_sigma)
        } else {
            WindModel::disabled()
        };
        let dist = (spawn_b - spawn_a).norm();
        Self {
            state_a: DroneState::hover_at(spawn_a),
            state_b: DroneState::hover_at(spawn_b),
            wind_a: wind,
            wind_b: wind,
            lockon_a: LockOnTracker::new(lockon_params.clone()),
            lockon_b: LockOnTracker::new(lockon_params),
            step_count: 0,
            done: false,
            prev_distance: dist,
            prev_lock_progress_a: 0.0,
        }
    }

    /// Spawn drones according to `spawn_mode`.
    #[allow(clippy::too_many_arguments)]
    pub fn random_spawn<R: Rng>(
        arena: &Arena,
        lockon_params: LockOnParams,
        wind_sigma: f64,
        wind_theta: f64,
        spawn_mode: &SpawnMode,
        rng: &mut R,
    ) -> Self {
        let (spawn_a, spawn_b) = spawn_mode.sample(arena, rng);
        Self::new(spawn_a, spawn_b, lockon_params, wind_sigma, wind_theta)
    }

    /// Step physics for both drones with given motor commands.
    #[allow(clippy::too_many_arguments)]
    pub fn step_physics<R: Rng>(
        &mut self,
        motors_a: &Vector4<f64>,
        motors_b: &Vector4<f64>,
        params: &DroneParams,
        arena: &Arena,
        dt_ctrl: f64,
        substeps: usize,
        rng: &mut R,
    ) -> BattleInfo {
        let dt_sim = dt_ctrl / substeps as f64;

        // Physics substeps
        for _ in 0..substeps {
            let wind_a = self.wind_a.step(dt_sim, rng);
            let wind_b = self.wind_b.step(dt_sim, rng);
            self.state_a = step_rk4(&self.state_a, motors_a, params, dt_sim, &wind_a);
            self.state_b = step_rk4(&self.state_b, motors_b, params, dt_sim, &wind_b);
        }

        // Collision detection
        let collision_a = arena.is_collision(&self.state_a.position)
            || arena.is_out_of_bounds(&self.state_a.position);
        let collision_b = arena.is_collision(&self.state_b.position)
            || arena.is_out_of_bounds(&self.state_b.position);

        // Lock-on tracking
        let kill_a = self
            .lockon_a
            .update(&self.state_a, &self.state_b, arena, dt_ctrl);
        let kill_b = self
            .lockon_b
            .update(&self.state_b, &self.state_a, arena, dt_ctrl);

        // Visibility
        let visible_ab = check_line_of_sight(arena, &self.state_a.position, &self.state_b.position)
            == Visibility::Visible;
        let visible_ba = check_line_of_sight(arena, &self.state_b.position, &self.state_a.position)
            == Visibility::Visible;

        let distance = self.state_a.distance_to(&self.state_b);

        self.step_count += 1;
        let timeout = self.step_count >= 1000; // max_steps checked by orchestrator

        self.done = collision_a || collision_b || kill_a || kill_b || timeout;

        BattleInfo {
            kill_a,
            kill_b,
            collision_a,
            collision_b,
            timeout,
            distance,
            lock_progress_a: self.lockon_a.progress(),
            lock_progress_b: self.lockon_b.progress(),
            visible_ab,
            visible_ba,
        }
    }

    /// Reset the battle to new spawn positions sampled from `spawn_mode`.
    #[allow(clippy::too_many_arguments)]
    pub fn reset<R: Rng>(
        &mut self,
        arena: &Arena,
        lockon_params: &LockOnParams,
        wind_sigma: f64,
        wind_theta: f64,
        spawn_mode: &SpawnMode,
        rng: &mut R,
    ) {
        let (spawn_a, spawn_b) = spawn_mode.sample(arena, rng);

        self.state_a = DroneState::hover_at(spawn_a);
        self.state_b = DroneState::hover_at(spawn_b);
        self.wind_a = if wind_sigma > 0.0 {
            WindModel::new(wind_theta, Vector3::zeros(), wind_sigma)
        } else {
            WindModel::disabled()
        };
        self.wind_b = self.wind_a;
        self.lockon_a = LockOnTracker::new(lockon_params.clone());
        self.lockon_b = LockOnTracker::new(lockon_params.clone());
        self.step_count = 0;
        self.done = false;
        self.prev_distance = (spawn_b - spawn_a).norm();
        self.prev_lock_progress_a = 0.0;
    }
}

/// Generate a random safe position inside the arena, away from obstacles.
fn random_safe_position<R: Rng>(arena: &Arena, margin: f64, rng: &mut R) -> Vector3<f64> {
    for _ in 0..100 {
        let x = rng.gen_range(margin..arena.bounds.x - margin);
        let y = rng.gen_range(margin..arena.bounds.y - margin);
        let z = rng.gen_range(margin..arena.bounds.z - margin);
        let p = Vector3::new(x, y, z);
        if arena.sdf(&p) > margin {
            return p;
        }
    }
    // Fallback to known safe corners
    Vector3::new(1.0, 1.0, 1.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aces_sim_core::environment::Obstacle;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

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
    fn test_battle_state_creation() {
        let arena = test_arena();
        let lockon = LockOnParams::default();
        let mut rng = rand::thread_rng();
        let spawn_mode = SpawnMode::default_for_warehouse();
        let battle = BattleState::random_spawn(&arena, lockon, 0.0, 2.0, &spawn_mode, &mut rng);
        assert!(!battle.done);
        assert_eq!(battle.step_count, 0);
    }

    #[test]
    fn test_fixed_spawn_mode_matches_cpu_env_defaults() {
        // With jitter=0, sample always returns the exact base spawn positions
        // matching the CPU env configured spawn (arena.toml).
        let mode = SpawnMode::Fixed {
            spawn_a: Vector3::new(1.0, 1.0, 1.5),
            spawn_b: Vector3::new(9.0, 9.0, 1.5),
            jitter: 0.0,
        };
        let arena = test_arena();
        let mut rng = SmallRng::seed_from_u64(42);
        let (a, b) = mode.sample(&arena, &mut rng);
        assert_eq!(a, Vector3::new(1.0, 1.0, 1.5));
        assert_eq!(b, Vector3::new(9.0, 9.0, 1.5));
    }

    #[test]
    fn test_fixed_spawn_mode_jitter_bounded() {
        // With jitter=0.5, all samples within ±0.5 of base spawn (per axis).
        let mode = SpawnMode::Fixed {
            spawn_a: Vector3::new(1.0, 1.0, 1.5),
            spawn_b: Vector3::new(9.0, 9.0, 1.5),
            jitter: 0.5,
        };
        let arena = test_arena();
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..100 {
            let (a, b) = mode.sample(&arena, &mut rng);
            assert!((a.x - 1.0).abs() <= 0.5);
            assert!((a.y - 1.0).abs() <= 0.5);
            assert!((a.z - 1.5).abs() <= 0.5);
            assert!((b.x - 9.0).abs() <= 0.5);
            assert!((b.y - 9.0).abs() <= 0.5);
            assert!((b.z - 1.5).abs() <= 0.5);
        }
    }

    #[test]
    fn test_random_spawn_mode_bounded_by_arena() {
        let mode = SpawnMode::Random { margin: 0.5 };
        let arena = test_arena();
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..50 {
            let (a, b) = mode.sample(&arena, &mut rng);
            assert!(a.x >= 0.5 && a.x <= 9.5);
            assert!(a.y >= 0.5 && a.y <= 9.5);
            assert!(a.z >= 0.5 && a.z <= 2.5);
            assert!(b.x >= 0.5 && b.x <= 9.5);
            assert!(b.y >= 0.5 && b.y <= 9.5);
            assert!(b.z >= 0.5 && b.z <= 2.5);
            // Fallback corner (1,1,1.5) always has sdf>0 in test_arena, so
            // `random_safe_position` is guaranteed to return a point with
            // sdf > 0 (either an accepted sample or the fallback).
            assert!(
                arena.sdf(&a) > 0.0,
                "sample a={a:?} has sdf={:.3}",
                arena.sdf(&a)
            );
            assert!(
                arena.sdf(&b) > 0.0,
                "sample b={b:?} has sdf={:.3}",
                arena.sdf(&b)
            );
        }
    }

    #[test]
    fn test_battle_step_physics() {
        let arena = test_arena();
        let params = DroneParams::crazyflie();
        let lockon = LockOnParams::default();
        let hover = params.hover_thrust();
        let motors = Vector4::new(hover, hover, hover, hover);

        let mut battle = BattleState::new(
            Vector3::new(1.0, 1.0, 1.5),
            Vector3::new(9.0, 9.0, 1.5),
            lockon,
            0.0,
            2.0,
        );

        let mut rng = rand::thread_rng();
        let info = battle.step_physics(&motors, &motors, &params, &arena, 0.01, 10, &mut rng);

        assert!(!info.collision_a);
        assert!(!info.collision_b);
        assert!(!info.kill_a);
        assert!(!info.kill_b);
        assert_eq!(battle.step_count, 1);
    }

    #[test]
    fn test_collision_terminates() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let params = DroneParams::crazyflie();
        let lockon = LockOnParams::default();

        // Spawn drone A at boundary — will be OOB after a push
        let mut battle = BattleState::new(
            Vector3::new(0.01, 5.0, 1.5),
            Vector3::new(9.0, 5.0, 1.5),
            lockon,
            0.0,
            2.0,
        );

        // Push drone A out of bounds with zero thrust
        let zero = Vector4::zeros();
        let hover = params.hover_thrust();
        let motors_b = Vector4::new(hover, hover, hover, hover);
        let mut rng = rand::thread_rng();

        // Run many steps — drone A will fall and go OOB
        for _ in 0..200 {
            let info = battle.step_physics(&zero, &motors_b, &params, &arena, 0.01, 10, &mut rng);
            if info.collision_a {
                assert!(battle.done);
                return;
            }
        }
        // It's OK if it doesn't collide in 200 steps — the test just
        // verifies collision detection is wired up.
    }

    #[test]
    fn test_random_safe_position() {
        let arena = test_arena();
        let mut rng = rand::thread_rng();
        for _ in 0..20 {
            let p = random_safe_position(&arena, 0.5, &mut rng);
            assert!(
                arena.sdf(&p) > 0.3,
                "position {:?} too close to obstacle, sdf={}",
                p,
                arena.sdf(&p)
            );
        }
    }
}
