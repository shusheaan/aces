use crate::collision::{check_line_of_sight, Visibility};
use crate::environment::Arena;
use crate::state::DroneState;

/// Parameters for the lock-on rule system.
#[derive(Debug, Clone)]
pub struct LockOnParams {
    /// Camera field of view in radians (full cone angle)
    pub fov: f64,
    /// Maximum lock-on distance (meters)
    pub lock_distance: f64,
    /// Time required to maintain lock for a kill (seconds)
    pub lock_duration: f64,
}

impl Default for LockOnParams {
    fn default() -> Self {
        Self {
            fov: std::f64::consts::FRAC_PI_2, // 90 degrees
            lock_distance: 2.0,
            lock_duration: 1.5,
        }
    }
}

/// Tracks the lock-on state for one drone targeting another.
#[derive(Debug, Clone)]
pub struct LockOnTracker {
    pub params: LockOnParams,
    /// Accumulated lock time (seconds)
    pub timer: f64,
    /// Whether currently in lock cone
    pub is_locking: bool,
}

impl LockOnTracker {
    pub fn new(params: LockOnParams) -> Self {
        Self {
            params,
            timer: 0.0,
            is_locking: false,
        }
    }

    /// Update lock-on state for one timestep.
    ///
    /// Returns true if kill is confirmed (lock held for full duration).
    pub fn update(
        &mut self,
        attacker: &DroneState,
        target: &DroneState,
        arena: &Arena,
        dt: f64,
    ) -> bool {
        let distance = attacker.distance_to(target);
        let angle = attacker.angle_to(target);
        let half_fov = self.params.fov / 2.0;

        // Check all lock conditions
        let in_fov = angle < half_fov;
        let in_range = distance <= self.params.lock_distance;
        let visible =
            check_line_of_sight(arena, &attacker.position, &target.position) == Visibility::Visible;

        if in_fov && in_range && visible {
            self.is_locking = true;
            self.timer += dt;
        } else {
            self.is_locking = false;
            self.timer = 0.0;
        }

        self.timer >= self.params.lock_duration
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.timer = 0.0;
        self.is_locking = false;
    }

    /// Lock progress as a fraction [0, 1].
    pub fn progress(&self) -> f64 {
        (self.timer / self.params.lock_duration).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_lock_on_facing_and_close() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let mut tracker = LockOnTracker::new(LockOnParams::default());

        let attacker = DroneState::hover_at(Vector3::new(1.0, 5.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5)); // 1m ahead on X

        // Should be locking (within FOV, within range, visible)
        let kill = tracker.update(&attacker, &target, &arena, 0.01);
        assert!(tracker.is_locking);
        assert!(!kill);
    }

    #[test]
    fn test_no_lock_behind() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let mut tracker = LockOnTracker::new(LockOnParams::default());

        let attacker = DroneState::hover_at(Vector3::new(5.0, 5.0, 1.5));
        // Target is behind attacker (negative X direction, attacker faces +X)
        let target = DroneState::hover_at(Vector3::new(4.0, 5.0, 1.5));

        tracker.update(&attacker, &target, &arena, 0.01);
        assert!(!tracker.is_locking);
    }

    #[test]
    fn test_kill_after_duration() {
        let arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        let params = LockOnParams {
            lock_duration: 0.1, // short for testing
            ..Default::default()
        };
        let mut tracker = LockOnTracker::new(params);

        let attacker = DroneState::hover_at(Vector3::new(1.0, 5.0, 1.5));
        let target = DroneState::hover_at(Vector3::new(2.0, 5.0, 1.5));

        // Hold lock for enough timesteps
        for _ in 0..20 {
            let kill = tracker.update(&attacker, &target, &arena, 0.01);
            if kill {
                assert!(tracker.timer >= 0.1);
                return;
            }
        }
        panic!("Kill should have triggered");
    }
}
