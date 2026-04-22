//! Symbolic Finite State Machine for tactical drone decision-making.
//!
//! Mirrors `aces/fsm.py`. Reads semantic features from the Perception NN
//! and selects a tactical mode. Outputs mode, safety margin, and pursuit
//! flag for the MPPI controller.

use std::fmt;

/// Tactical mode for a drone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroneMode {
    Hover,
    Pursue,
    Evade,
    Search,
    Orbit,
}

impl DroneMode {
    /// Lower number = higher priority (overrides immediately).
    fn priority(self) -> u8 {
        match self {
            DroneMode::Hover => 0,
            DroneMode::Evade => 1,
            DroneMode::Pursue => 2,
            DroneMode::Search => 3,
            DroneMode::Orbit => 4,
        }
    }
}

impl fmt::Display for DroneMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DroneMode::Hover => "HOVER",
            DroneMode::Pursue => "PURSUE",
            DroneMode::Evade => "EVADE",
            DroneMode::Search => "SEARCH",
            DroneMode::Orbit => "ORBIT",
        };
        f.write_str(s)
    }
}

/// Semantic features produced by the Perception NN.
#[derive(Debug, Clone, Copy)]
pub struct SemanticFeatures {
    pub threat: f64,
    pub opportunity: f64,
    pub collision_risk: f64,
    pub uncertainty: f64,
    pub opponent_distance: f64,
    /// Predicted opponent intent: 0=approach, 1=flee, 2=patrol.
    /// Available for future FSM rules; not used in current transition logic.
    #[allow(dead_code)]
    pub opponent_intent: u8,
}

/// Output from a single FSM step.
#[derive(Debug, Clone, Copy)]
pub struct FsmOutput {
    pub mode: DroneMode,
    /// Dynamic safety margin for MPPI (meters).
    /// Reserved for future use by the MPPI chance-constraint layer.
    #[allow(dead_code)]
    pub d_safe: f64,
    /// Whether to enable pursuit tracking in MPPI.
    pub pursuit: bool,
}

/// Priority-based FSM with hysteresis.
///
/// Transition rules (checked in priority order):
///   1. `collision_risk > 0.7`                               → Hover
///   2. `threat > 0.7`                                       → Evade
///   3. `opportunity > 0.6 && opponent_distance < 3`         → Pursue
///   4. `uncertainty > 0.5`                                  → Search
///   5. else                                                  → Orbit
///
/// Hysteresis: a lower-priority mode must be requested for
/// `hysteresis_ticks` consecutive ticks before the FSM switches.
/// Higher-priority modes override immediately.
pub struct SymbolicFsm {
    pub mode: DroneMode,
    hysteresis_ticks: u32,
    ticks_requesting: u32,
    requested_mode: DroneMode,
}

impl SymbolicFsm {
    pub fn new(hysteresis_ticks: u32) -> Self {
        Self {
            mode: DroneMode::Orbit,
            hysteresis_ticks,
            ticks_requesting: 0,
            requested_mode: DroneMode::Orbit,
        }
    }

    pub fn reset(&mut self) {
        self.mode = DroneMode::Orbit;
        self.ticks_requesting = 0;
        self.requested_mode = DroneMode::Orbit;
    }

    pub fn step(&mut self, features: &SemanticFeatures) -> FsmOutput {
        let desired = self.evaluate(features);

        if desired == self.mode {
            self.ticks_requesting = 0;
            self.requested_mode = desired;
        } else if desired.priority() < self.mode.priority() {
            // Higher-priority mode — override immediately.
            self.mode = desired;
            self.ticks_requesting = 0;
            self.requested_mode = desired;
        } else if desired == self.requested_mode {
            self.ticks_requesting += 1;
            if self.ticks_requesting >= self.hysteresis_ticks {
                self.mode = desired;
                self.ticks_requesting = 0;
            }
        } else {
            self.requested_mode = desired;
            self.ticks_requesting = 1;
        }

        let d_safe = 0.3 + 0.3 * features.collision_risk;
        let pursuit = matches!(self.mode, DroneMode::Pursue | DroneMode::Search);

        FsmOutput {
            mode: self.mode,
            d_safe,
            pursuit,
        }
    }

    fn evaluate(&self, f: &SemanticFeatures) -> DroneMode {
        if f.collision_risk > 0.7 {
            return DroneMode::Hover;
        }
        if f.threat > 0.7 {
            return DroneMode::Evade;
        }
        if f.opportunity > 0.6 && f.opponent_distance < 3.0 {
            return DroneMode::Pursue;
        }
        if f.uncertainty > 0.5 {
            return DroneMode::Search;
        }
        DroneMode::Orbit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(
        threat: f64,
        opportunity: f64,
        collision_risk: f64,
        uncertainty: f64,
        opponent_distance: f64,
    ) -> SemanticFeatures {
        SemanticFeatures {
            threat,
            opportunity,
            collision_risk,
            uncertainty,
            opponent_distance,
            opponent_intent: 0,
        }
    }

    #[test]
    fn default_orbit() {
        let mut fsm = SymbolicFsm::new(10);
        let out = fsm.step(&feats(0.0, 0.0, 0.0, 0.0, 10.0));
        assert_eq!(out.mode, DroneMode::Orbit);
        assert!(!out.pursuit);
    }

    #[test]
    fn collision_overrides_all() {
        let mut fsm = SymbolicFsm::new(10);
        // Start in Pursue to confirm high-priority collision overrides
        let f_pursue = feats(0.0, 0.9, 0.0, 0.0, 1.0);
        // Accumulate hysteresis ticks for Pursue
        for _ in 0..10 {
            fsm.step(&f_pursue);
        }
        // Now trigger collision_risk > 0.7
        let out = fsm.step(&feats(0.0, 0.9, 0.9, 0.0, 1.0));
        assert_eq!(out.mode, DroneMode::Hover);
    }

    #[test]
    fn pursue_flag() {
        let mut fsm = SymbolicFsm::new(2);
        let f = feats(0.0, 0.9, 0.0, 0.0, 1.0);
        // Need hysteresis_ticks=2 consecutive ticks to switch
        fsm.step(&f);
        let out = fsm.step(&f);
        assert_eq!(out.mode, DroneMode::Pursue);
        assert!(out.pursuit);
    }
}
