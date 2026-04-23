//! f32 port of `mppi::cost` for GPU (WGSL) parity validation.
//!
//! WGSL compute shaders only support `f32` (no `f64`), so the Phase 2 GPU MPPI
//! rollout must evaluate stage costs in f32. This module mirrors the pursuit
//! and evasion cost functions from `crates/mppi/src/cost.rs` exactly — only
//! the numeric type differs. Belief-weighted variants are out of scope for
//! this port (the batch-sim plan defers particle-filter belief state).
//!
//! Validation tests (`#[cfg(test)]`) compare f32 vs f64 costs across
//! randomized configurations and each branch of the cost function
//! (far-field, near-obstacle, inside-obstacle, close-range evasion, in-FOV
//! evasion).
//!
//! NOTE: keep this file logic-identical to `mppi::cost::{pursuit_cost,
//! evasion_cost}`. Do not add features here that are not in the f64 reference.

use crate::f32_dynamics::DroneStateF32;
use crate::f32_sdf::ArenaF32;
use aces_mppi::cost::CostWeights;
use nalgebra::Vector4;

/// Cost function weights for the MPPI controller (f32 port of `CostWeights`).
#[derive(Debug, Clone, Copy)]
pub struct CostWeightsF32 {
    /// Weight for distance to opponent
    pub w_dist: f32,
    /// Weight for facing opponent
    pub w_face: f32,
    /// Weight for control smoothness
    pub w_ctrl: f32,
    /// Weight for obstacle avoidance
    pub w_obs: f32,
    /// Safe distance from obstacles (meters)
    pub d_safe: f32,
}

impl Default for CostWeightsF32 {
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

impl CostWeightsF32 {
    /// Convert from the f64 reference weights (downcast).
    pub fn from_f64(w: &CostWeights) -> Self {
        Self {
            w_dist: w.w_dist as f32,
            w_face: w.w_face as f32,
            w_ctrl: w.w_ctrl as f32,
            w_obs: w.w_obs as f32,
            d_safe: w.d_safe as f32,
        }
    }
}

/// Compute stage cost for pursuit behavior (f32).
///
/// Mirrors `mppi::cost::pursuit_cost`:
///   - distance² term (`w_dist`)
///   - facing term `w_face * (1 - cos(angle))`
///   - control deviation from hover (`w_ctrl * ||u - hover||²`)
///   - obstacle: sdf ≤ 0 → 1e6; sdf < d_safe → quadratic margin penalty
pub fn pursuit_cost_f32(
    self_state: &DroneStateF32,
    enemy_state: &DroneStateF32,
    control: &Vector4<f32>,
    hover_thrust: f32,
    arena: &ArenaF32,
    weights: &CostWeightsF32,
) -> f32 {
    let mut cost = 0.0f32;

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

/// Compute stage cost for evasion behavior (f32).
///
/// Mirrors `mppi::cost::evasion_cost`:
///   - close-range `(safe_dist - dist)²` penalty when dist < 3m
///   - in-FOV penalty when enemy->self angle < 45°
///   - control deviation from hover
///   - obstacle: sdf ≤ 0 → 1e6; sdf < d_safe → quadratic margin penalty
pub fn evasion_cost_f32(
    self_state: &DroneStateF32,
    enemy_state: &DroneStateF32,
    control: &Vector4<f32>,
    hover_thrust: f32,
    arena: &ArenaF32,
    weights: &CostWeightsF32,
) -> f32 {
    let mut cost = 0.0f32;

    // Reward distance from opponent (penalize being too close)
    let dist = self_state.distance_to(enemy_state);
    let safe_dist = 3.0f32; // meters
    if dist < safe_dist {
        let margin = safe_dist - dist;
        cost += weights.w_dist * margin * margin;
    }

    // Penalize being in enemy's FOV
    let angle_from_enemy = enemy_state.angle_to(self_state);
    let fov_half = std::f32::consts::FRAC_PI_4; // 45 degrees
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

#[cfg(test)]
mod tests {
    use super::*;
    use aces_mppi::cost::{evasion_cost, pursuit_cost, CostWeights};
    use aces_sim_core::environment::{Arena, Obstacle};
    use aces_sim_core::state::DroneState;
    use nalgebra::{Quaternion, UnitQuaternion, Vector3, Vector4};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// Default warehouse arena: 10x10x3m with 5 box pillars (matches
    /// `f32_sdf::tests::warehouse_arena_f64`).
    fn warehouse_arena_f64() -> Arena {
        let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
            arena.obstacles.push(Obstacle::Box {
                center: Vector3::new(x, y, 1.5),
                half_extents: Vector3::new(0.5, 0.5, 1.5),
            });
        }
        arena
    }

    /// Pillar centers (x, y) for probe generation. Must match
    /// `warehouse_arena_f64`.
    const PILLARS: [(f64, f64); 5] = [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)];

    const MAX_THRUST: f64 = 0.15;
    /// Crazyflie hover thrust = mass * g / 4 = 0.027 * 9.81 / 4.
    const HOVER_THRUST: f64 = 0.027 * 9.81 / 4.0;

    /// Sample a position in the open arena volume (roughly avoiding pillars).
    fn sample_position(rng: &mut SmallRng) -> Vector3<f64> {
        Vector3::new(
            rng.gen_range(1.0..9.0),
            rng.gen_range(1.0..9.0),
            rng.gen_range(0.5..2.5),
        )
    }

    /// Sample a random unit quaternion by normalizing a 4-vector of unit
    /// normals (the Marsaglia trick — rejection-free and uniform on SO(3)).
    fn sample_attitude(rng: &mut SmallRng) -> UnitQuaternion<f64> {
        use rand_distr::{Distribution, StandardNormal};
        let n: f64 = StandardNormal.sample(rng);
        let i: f64 = StandardNormal.sample(rng);
        let j: f64 = StandardNormal.sample(rng);
        let k: f64 = StandardNormal.sample(rng);
        let q = Quaternion::new(n, i, j, k);
        let norm = q.norm();
        // Extremely unlikely zero-norm draw — re-roll to nearest valid.
        if norm < 1e-12 {
            UnitQuaternion::identity()
        } else {
            UnitQuaternion::from_quaternion(q / norm)
        }
    }

    /// Sample a control vector uniformly in [0, max_thrust]^4.
    fn sample_control(rng: &mut SmallRng) -> Vector4<f64> {
        Vector4::new(
            rng.gen_range(0.0..MAX_THRUST),
            rng.gen_range(0.0..MAX_THRUST),
            rng.gen_range(0.0..MAX_THRUST),
            rng.gen_range(0.0..MAX_THRUST),
        )
    }

    /// Build an f64 DroneState (velocity and angular velocity zero — cost
    /// functions don't read them).
    fn build_state(position: Vector3<f64>, attitude: UnitQuaternion<f64>) -> DroneState {
        DroneState::new(position, Vector3::zeros(), attitude, Vector3::zeros())
    }

    /// Assert that the f32 cost is close to the f64 cost.
    ///
    /// Uses a combined absolute+relative tolerance so that small costs don't
    /// trigger false positives on the relative check. Tolerated absolute diff
    /// is `rel_tol * max(1.0, |f64_cost|)`.
    #[track_caller]
    fn assert_close(f32_cost: f32, f64_cost: f64, rel_tol: f64, ctx: &str) {
        let abs_diff = (f32_cost as f64 - f64_cost).abs();
        let denom = f64_cost.abs().max(1.0);
        let rel = abs_diff / denom;
        assert!(
            rel < rel_tol,
            "{ctx}: f32={f32_cost} f64={f64_cost} abs_diff={abs_diff} rel={rel} (tol {rel_tol})"
        );
    }

    const REL_TOL: f64 = 1e-4;

    #[test]
    fn test_f32_pursuit_cost_matches_f64_random() {
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);
        let mut rng = SmallRng::seed_from_u64(42);

        let mut max_rel = 0.0f64;
        for i in 0..500 {
            let self_pos = sample_position(&mut rng);
            let self_att = sample_attitude(&mut rng);
            let enemy_pos = sample_position(&mut rng);
            let enemy_att = sample_attitude(&mut rng);
            let control = sample_control(&mut rng);

            let self_f64 = build_state(self_pos, self_att);
            let enemy_f64 = build_state(enemy_pos, enemy_att);
            let self_f32 = DroneStateF32::from_f64(&self_f64);
            let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
            let control_f32 = control.cast::<f32>();

            let c_f64 = pursuit_cost(
                &self_f64,
                &enemy_f64,
                &control,
                HOVER_THRUST,
                &arena_f64,
                &weights_f64,
            );
            let c_f32 = pursuit_cost_f32(
                &self_f32,
                &enemy_f32,
                &control_f32,
                HOVER_THRUST as f32,
                &arena_f32,
                &weights_f32,
            );

            let abs_diff = (c_f32 as f64 - c_f64).abs();
            let denom = c_f64.abs().max(1.0);
            let rel = abs_diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
            assert_close(c_f32, c_f64, REL_TOL, &format!("pursuit random #{i}"));
        }
        // Sanity: should comfortably be below tolerance.
        assert!(
            max_rel < REL_TOL,
            "pursuit random max_rel={max_rel} exceeded tol {REL_TOL}"
        );
    }

    #[test]
    fn test_f32_evasion_cost_matches_f64_random() {
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);
        let mut rng = SmallRng::seed_from_u64(7);

        let mut max_rel = 0.0f64;
        for i in 0..500 {
            let self_pos = sample_position(&mut rng);
            let self_att = sample_attitude(&mut rng);
            let enemy_pos = sample_position(&mut rng);
            let enemy_att = sample_attitude(&mut rng);
            let control = sample_control(&mut rng);

            let self_f64 = build_state(self_pos, self_att);
            let enemy_f64 = build_state(enemy_pos, enemy_att);
            let self_f32 = DroneStateF32::from_f64(&self_f64);
            let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
            let control_f32 = control.cast::<f32>();

            let c_f64 = evasion_cost(
                &self_f64,
                &enemy_f64,
                &control,
                HOVER_THRUST,
                &arena_f64,
                &weights_f64,
            );
            let c_f32 = evasion_cost_f32(
                &self_f32,
                &enemy_f32,
                &control_f32,
                HOVER_THRUST as f32,
                &arena_f32,
                &weights_f32,
            );

            let abs_diff = (c_f32 as f64 - c_f64).abs();
            let denom = c_f64.abs().max(1.0);
            let rel = abs_diff / denom;
            if rel > max_rel {
                max_rel = rel;
            }
            assert_close(c_f32, c_f64, REL_TOL, &format!("evasion random #{i}"));
        }
        assert!(
            max_rel < REL_TOL,
            "evasion random max_rel={max_rel} exceeded tol {REL_TOL}"
        );
    }

    #[test]
    fn test_f32_pursuit_cost_near_obstacle() {
        // Sample 100 positions within 0.5m of the pillar surfaces so the
        // quadratic margin-penalty branch fires (d_safe = 0.3m default).
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);
        let d_safe = weights_f64.d_safe;
        let mut rng = SmallRng::seed_from_u64(11);

        let mut n_accepted = 0usize;
        let mut attempts = 0usize;
        while n_accepted < 100 {
            attempts += 1;
            assert!(
                attempts < 10_000,
                "failed to sample 100 near-obstacle probes"
            );

            // Pick a random pillar and jitter outward within [0.01, 0.5] m on
            // a random face normal.
            let (cx, cy) = PILLARS[rng.gen_range(0..PILLARS.len())];
            let hx = 0.5_f64;
            let hy = 0.5_f64;
            let hz = 1.5_f64;
            let zc = 1.5_f64;
            // Face: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
            let face = rng.gen_range(0..6usize);
            let off = rng.gen_range(0.01..0.5);
            let mut p = Vector3::new(cx, cy, zc);
            // Jitter tangentially within the face (kept inside the face
            // extent to guarantee SDF is the face-normal distance).
            let tangent_jitter = 0.3_f64; // < half-extent
            match face {
                0 => {
                    p.x += hx + off;
                    p.y += rng.gen_range(-tangent_jitter..tangent_jitter);
                    p.z += rng.gen_range(-tangent_jitter..tangent_jitter);
                }
                1 => {
                    p.x -= hx + off;
                    p.y += rng.gen_range(-tangent_jitter..tangent_jitter);
                    p.z += rng.gen_range(-tangent_jitter..tangent_jitter);
                }
                2 => {
                    p.y += hy + off;
                    p.x += rng.gen_range(-tangent_jitter..tangent_jitter);
                    p.z += rng.gen_range(-tangent_jitter..tangent_jitter);
                }
                3 => {
                    p.y -= hy + off;
                    p.x += rng.gen_range(-tangent_jitter..tangent_jitter);
                    p.z += rng.gen_range(-tangent_jitter..tangent_jitter);
                }
                4 => {
                    p.z += hz + off;
                    p.x += rng.gen_range(-tangent_jitter..tangent_jitter);
                    p.y += rng.gen_range(-tangent_jitter..tangent_jitter);
                }
                5 => {
                    // Avoid going below the floor (z < 0).
                    p.z = (zc - hz - off).max(0.0);
                    p.x += rng.gen_range(-tangent_jitter..tangent_jitter);
                    p.y += rng.gen_range(-tangent_jitter..tangent_jitter);
                }
                _ => unreachable!(),
            }

            // Require sdf in (0, d_safe) so the quadratic branch fires.
            let sdf = arena_f64.sdf(&p);
            if !(sdf > 1e-6 && sdf < d_safe) {
                continue;
            }
            n_accepted += 1;

            let self_att = sample_attitude(&mut rng);
            let enemy_pos = sample_position(&mut rng);
            let enemy_att = sample_attitude(&mut rng);
            let control = sample_control(&mut rng);

            let self_f64 = build_state(p, self_att);
            let enemy_f64 = build_state(enemy_pos, enemy_att);
            let self_f32 = DroneStateF32::from_f64(&self_f64);
            let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
            let control_f32 = control.cast::<f32>();

            let c_f64 = pursuit_cost(
                &self_f64,
                &enemy_f64,
                &control,
                HOVER_THRUST,
                &arena_f64,
                &weights_f64,
            );
            let c_f32 = pursuit_cost_f32(
                &self_f32,
                &enemy_f32,
                &control_f32,
                HOVER_THRUST as f32,
                &arena_f32,
                &weights_f32,
            );
            assert_close(
                c_f32,
                c_f64,
                REL_TOL,
                &format!("pursuit near-obstacle (sdf={sdf})"),
            );
        }
    }

    #[test]
    fn test_f32_pursuit_cost_inside_obstacle() {
        // 20 positions inside pillars (sdf < 0, so 1e6 penalty branch fires).
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);
        let mut rng = SmallRng::seed_from_u64(13);

        // Absolute tolerance 1.0 is fine at 1e6 magnitude (rel 1e-6).
        const ABS_TOL: f64 = 1.0;

        for i in 0..20 {
            // Pick a pillar and jitter inside by ±(0..0.4) on each axis
            // (half-extents are 0.5/0.5/1.5, so 0.4 keeps us inside).
            let (cx, cy) = PILLARS[i % PILLARS.len()];
            let p = Vector3::new(
                cx + rng.gen_range(-0.4..0.4),
                cy + rng.gen_range(-0.4..0.4),
                1.5 + rng.gen_range(-1.4..1.4),
            );
            // Sanity: ensure inside.
            let sdf = arena_f64.sdf(&p);
            assert!(
                sdf <= 0.0,
                "sample #{i} was not inside obstacle (sdf={sdf})"
            );

            let self_att = sample_attitude(&mut rng);
            let enemy_pos = sample_position(&mut rng);
            let enemy_att = sample_attitude(&mut rng);
            let control = sample_control(&mut rng);

            let self_f64 = build_state(p, self_att);
            let enemy_f64 = build_state(enemy_pos, enemy_att);
            let self_f32 = DroneStateF32::from_f64(&self_f64);
            let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
            let control_f32 = control.cast::<f32>();

            let c_f64 = pursuit_cost(
                &self_f64,
                &enemy_f64,
                &control,
                HOVER_THRUST,
                &arena_f64,
                &weights_f64,
            );
            let c_f32 = pursuit_cost_f32(
                &self_f32,
                &enemy_f32,
                &control_f32,
                HOVER_THRUST as f32,
                &arena_f32,
                &weights_f32,
            );
            let abs_diff = (c_f32 as f64 - c_f64).abs();
            assert!(
                abs_diff < ABS_TOL,
                "pursuit inside-obstacle #{i}: abs_diff={abs_diff} (tol {ABS_TOL}); f64={c_f64} f32={c_f32}"
            );
            // Also sanity that both hit the 1e6 branch.
            assert!(c_f64 >= 1e6, "expected 1e6 penalty, got f64={c_f64}");
        }
    }

    #[test]
    fn test_f32_evasion_cost_close_proximity() {
        // 50 configurations with dist < 3.0 (close-range evasion branch).
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);
        let mut rng = SmallRng::seed_from_u64(17);

        let mut close_hits = 0usize;
        for i in 0..50 {
            // Anchor self in open space, place enemy within 2.5m on a random
            // unit-sphere direction.
            let self_pos = sample_position(&mut rng);
            let d: f64 = rng.gen_range(0.2..2.5);
            // Random direction.
            use rand_distr::{Distribution, StandardNormal};
            let ex: f64 = StandardNormal.sample(&mut rng);
            let ey: f64 = StandardNormal.sample(&mut rng);
            let ez: f64 = StandardNormal.sample(&mut rng);
            let mut dir = Vector3::new(ex, ey, ez);
            let n = dir.norm();
            if n < 1e-12 {
                dir = Vector3::x();
            } else {
                dir /= n;
            }
            let mut enemy_pos = self_pos + dir * d;
            // Clip into arena to keep sdf sensible.
            enemy_pos.x = enemy_pos.x.clamp(0.2, 9.8);
            enemy_pos.y = enemy_pos.y.clamp(0.2, 9.8);
            enemy_pos.z = enemy_pos.z.clamp(0.1, 2.9);

            let self_att = sample_attitude(&mut rng);
            let enemy_att = sample_attitude(&mut rng);
            let control = sample_control(&mut rng);

            let self_f64 = build_state(self_pos, self_att);
            let enemy_f64 = build_state(enemy_pos, enemy_att);
            // The arena clip above can push the enemy beyond 3m, so count how
            // many samples truly exercise the `dist < 3.0` close-range branch
            // and assert an adequate coverage floor after the loop.
            let actual_dist = self_f64.distance_to(&enemy_f64);
            if actual_dist < 3.0 {
                close_hits += 1;
            }

            let self_f32 = DroneStateF32::from_f64(&self_f64);
            let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
            let control_f32 = control.cast::<f32>();

            let c_f64 = evasion_cost(
                &self_f64,
                &enemy_f64,
                &control,
                HOVER_THRUST,
                &arena_f64,
                &weights_f64,
            );
            let c_f32 = evasion_cost_f32(
                &self_f32,
                &enemy_f32,
                &control_f32,
                HOVER_THRUST as f32,
                &arena_f32,
                &weights_f32,
            );
            assert_close(
                c_f32,
                c_f64,
                REL_TOL,
                &format!("evasion close-proximity #{i}"),
            );
        }
        // Branch-coverage guarantee: the close-range `dist < 3.0` branch must
        // fire on at least 30 of 50 samples. If this fails, the test is no
        // longer exercising the branch it claims to cover.
        assert!(
            close_hits >= 30,
            "close-proximity branch fired on only {close_hits}/50 samples (need >= 30); \
             test is not exercising the dist<3.0 branch — redesign the sampler"
        );
    }

    #[test]
    fn test_f32_evasion_cost_in_enemy_fov() {
        // 50 configurations where self is inside the enemy's 45° FOV cone.
        // To guarantee this: place enemy with identity attitude (forward=+X),
        // then place self along +X offset by a random radial jitter smaller
        // than tan(fov_half)*dist.
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);
        let mut rng = SmallRng::seed_from_u64(19);
        let fov_half = std::f64::consts::FRAC_PI_4;

        for i in 0..50 {
            let enemy_pos = Vector3::new(
                rng.gen_range(2.0..8.0),
                rng.gen_range(2.0..8.0),
                rng.gen_range(1.0..2.0),
            );
            let enemy_att = UnitQuaternion::identity();
            let dist = rng.gen_range(0.5..4.0);
            // Cap lateral offset so angle < fov_half with margin.
            let max_lat = dist * (fov_half * 0.8).tan();
            let dy = rng.gen_range(-max_lat..max_lat);
            let dz = rng.gen_range(-max_lat..max_lat);
            let self_pos = enemy_pos + Vector3::new(dist, dy, dz);

            // Clip into arena so sdf stays finite; very tight FOV invariant
            // depends on relative geometry which clipping won't break here
            // (we verify below).
            let self_att = sample_attitude(&mut rng);
            let control = sample_control(&mut rng);

            let self_f64 = build_state(self_pos, self_att);
            let enemy_f64 = build_state(enemy_pos, enemy_att);

            // Sanity: enemy must see self inside FOV.
            let angle_from_enemy = enemy_f64.angle_to(&self_f64);
            assert!(
                angle_from_enemy < fov_half,
                "#{i}: expected self in enemy FOV, got angle {angle_from_enemy} rad"
            );

            let self_f32 = DroneStateF32::from_f64(&self_f64);
            let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
            let control_f32 = control.cast::<f32>();

            let c_f64 = evasion_cost(
                &self_f64,
                &enemy_f64,
                &control,
                HOVER_THRUST,
                &arena_f64,
                &weights_f64,
            );
            let c_f32 = evasion_cost_f32(
                &self_f32,
                &enemy_f32,
                &control_f32,
                HOVER_THRUST as f32,
                &arena_f32,
                &weights_f32,
            );
            assert_close(c_f32, c_f64, REL_TOL, &format!("evasion in-FOV #{i}"));
        }
    }

    #[test]
    fn test_pursuit_cost_f32_structural_parity() {
        // Identity-attitude state at origin (well in open space), enemy at
        // +X distance d, zero control. Expect:
        //   w_dist * d² + w_face*(1 - cos(0)) [=0] + w_ctrl * ||0 - hover||²
        //     + 0 (no obstacle near)
        // and that f32 matches f64 within the relative tolerance.
        let arena_f64 = warehouse_arena_f64();
        let arena_f32 = ArenaF32::from_f64(&arena_f64);
        let weights_f64 = CostWeights::default();
        let weights_f32 = CostWeightsF32::from_f64(&weights_f64);

        // Use a hover-ish open-space point well away from pillars: center of
        // an empty corner is (1.0, 5.0, 1.5). Far from all pillars.
        let self_pos = Vector3::new(1.0, 5.0, 1.5);
        let d = 2.0_f64;
        let enemy_pos = self_pos + Vector3::new(d, 0.0, 0.0);
        let self_att = UnitQuaternion::identity();
        let enemy_att = UnitQuaternion::identity();
        let control = Vector4::zeros(); // zero thrust

        let self_f64 = build_state(self_pos, self_att);
        let enemy_f64 = build_state(enemy_pos, enemy_att);
        let self_f32 = DroneStateF32::from_f64(&self_f64);
        let enemy_f32 = DroneStateF32::from_f64(&enemy_f64);
        let control_f32 = control.cast::<f32>();

        let c_f64 = pursuit_cost(
            &self_f64,
            &enemy_f64,
            &control,
            HOVER_THRUST,
            &arena_f64,
            &weights_f64,
        );
        let c_f32 = pursuit_cost_f32(
            &self_f32,
            &enemy_f32,
            &control_f32,
            HOVER_THRUST as f32,
            &arena_f32,
            &weights_f32,
        );

        // Analytic expectation. Angle = 0 (enemy directly in front), so
        // face term = 0. No obstacle nearby (sdf > d_safe). Control is zero,
        // so ctrl_diff = -hover (4 entries).
        let expected_dist_term = weights_f64.w_dist * d * d;
        let expected_face_term = 0.0;
        let expected_ctrl_term = weights_f64.w_ctrl * 4.0 * HOVER_THRUST * HOVER_THRUST;
        let expected = expected_dist_term + expected_face_term + expected_ctrl_term;

        // f64 should hit the analytic value very tightly.
        assert!(
            (c_f64 - expected).abs() < 1e-10,
            "f64 structural: {c_f64} vs expected {expected}"
        );

        // f32 vs f64 within tolerance.
        assert_close(c_f32, c_f64, REL_TOL, "pursuit structural");
    }
}
