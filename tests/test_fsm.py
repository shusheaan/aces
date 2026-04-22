"""Tests for aces.fsm — Symbolic Finite State Machine."""

from aces.fsm import DroneMode, SymbolicFSM


def _features(**overrides: float) -> dict[str, float]:
    """Default safe features with overrides."""
    defaults = {
        "threat": 0.0,
        "opportunity": 0.0,
        "collision_risk": 0.0,
        "uncertainty": 0.0,
        "opponent_distance": 5.0,
        "opponent_intent": 2,  # patrol
    }
    defaults.update(overrides)
    return defaults


def test_default_state_is_orbit() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features())
    assert out.mode == DroneMode.ORBIT


def test_high_collision_risk_triggers_hover() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(collision_risk=0.8))
    assert out.mode == DroneMode.HOVER


def test_high_threat_triggers_evade() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(threat=0.8))
    assert out.mode == DroneMode.EVADE


def test_high_opportunity_close_triggers_pursue() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert out.mode == DroneMode.PURSUE


def test_high_uncertainty_triggers_search() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(uncertainty=0.6))
    assert out.mode == DroneMode.SEARCH


def test_priority_collision_over_threat() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(collision_risk=0.8, threat=0.9))
    assert out.mode == DroneMode.HOVER


def test_priority_threat_over_opportunity() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(threat=0.8, opportunity=0.9, opponent_distance=1.0))
    assert out.mode == DroneMode.EVADE


def test_hysteresis_prevents_chatter() -> None:
    fsm = SymbolicFSM(hysteresis_ticks=10)
    fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert fsm.mode == DroneMode.PURSUE
    for _ in range(5):
        out = fsm.step(_features(opportunity=0.5, opponent_distance=2.0))
    assert out.mode == DroneMode.PURSUE


def test_hysteresis_allows_transition_after_enough_ticks() -> None:
    fsm = SymbolicFSM(hysteresis_ticks=3)
    fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert fsm.mode == DroneMode.PURSUE
    for _ in range(4):
        out = fsm.step(_features())
    assert out.mode == DroneMode.ORBIT


def test_high_priority_overrides_hysteresis() -> None:
    fsm = SymbolicFSM(hysteresis_ticks=100)
    fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert fsm.mode == DroneMode.PURSUE
    out = fsm.step(_features(collision_risk=0.8))
    assert out.mode == DroneMode.HOVER


def test_d_safe_scales_with_collision_risk() -> None:
    fsm = SymbolicFSM()
    out_safe = fsm.step(_features(collision_risk=0.0))
    fsm2 = SymbolicFSM()
    out_risky = fsm2.step(_features(collision_risk=1.0))
    assert out_risky.d_safe > out_safe.d_safe


def test_pursuit_flag() -> None:
    fsm = SymbolicFSM()
    out = fsm.step(_features(opportunity=0.7, opponent_distance=2.0))
    assert out.pursuit is True
    fsm2 = SymbolicFSM()
    out2 = fsm2.step(_features(threat=0.8))
    assert out2.pursuit is False
