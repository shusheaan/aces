"""Tests for aces.god_oracle — ground truth label computation."""

import numpy as np

from aces.perception import GodOracle, extract_oracle_inputs


def test_threat_max_when_being_locked_and_close() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=1.0,
        distance=0.5,
        opponent_facing_me=1.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["threat"] > 0.9


def test_threat_zero_when_safe() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=10.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["threat"] < 0.1


def test_opportunity_high_when_locked_and_close() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=1.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.8,
        a_sees_b=True,
        i_face_opponent=1.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["opportunity"] > 0.7


def test_collision_risk_high_near_wall_fast() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=10.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=0.1,
        speed=3.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["collision_risk"] > 0.7


def test_collision_risk_zero_far_from_walls() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=10.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=False,
        i_face_opponent=0.0,
        nearest_obs_dist=2.0,
        speed=1.0,
        belief_var=0.0,
        opponent_closing_speed=0.0,
    )
    assert labels["collision_risk"] < 0.01


def test_opponent_intent_approach() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=5.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=True,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=2.0,
    )
    assert labels["opponent_intent"] == 0


def test_opponent_intent_flee() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.0,
        distance=5.0,
        opponent_facing_me=0.0,
        lock_a_progress=0.0,
        a_sees_b=True,
        i_face_opponent=0.0,
        nearest_obs_dist=5.0,
        speed=0.0,
        belief_var=0.0,
        opponent_closing_speed=-2.0,
    )
    assert labels["opponent_intent"] == 1


def test_all_labels_in_range() -> None:
    oracle = GodOracle()
    labels = oracle.compute(
        lock_b_progress=0.5,
        distance=3.0,
        opponent_facing_me=0.5,
        lock_a_progress=0.5,
        a_sees_b=True,
        i_face_opponent=0.5,
        nearest_obs_dist=0.3,
        speed=2.0,
        belief_var=2.0,
        opponent_closing_speed=0.0,
    )
    assert 0.0 <= labels["threat"] <= 1.0
    assert 0.0 <= labels["opportunity"] <= 1.0
    assert 0.0 <= labels["collision_risk"] <= 1.0
    assert 0.0 <= labels["uncertainty"] <= 1.0
    assert labels["opponent_distance"] >= 0.0
    assert labels["opponent_intent"] in (0, 1, 2)


# --- extract_oracle_inputs tests ---


def test_extract_oracle_inputs_returns_all_keys() -> None:
    obs = np.zeros(21, dtype=np.float32)
    obs[6] = 3.0  # rel_pos x
    obs[0] = 1.0  # own_vel x
    info: dict[str, object] = {}
    result = extract_oracle_inputs(obs, info)
    expected_keys = {
        "lock_b_progress",
        "distance",
        "opponent_facing_me",
        "lock_a_progress",
        "a_sees_b",
        "i_face_opponent",
        "nearest_obs_dist",
        "speed",
        "belief_var",
        "opponent_closing_speed",
    }
    assert set(result.keys()) == expected_keys


def test_extract_oracle_inputs_compatible_with_compute() -> None:
    """Extracted inputs can be passed directly to oracle.compute()."""
    obs = np.zeros(21, dtype=np.float32)
    obs[6] = 2.0  # rel_pos x → distance ~2
    obs[0] = 1.0  # own_vel x
    info: dict[str, object] = {"distance": 2.0, "lock_progress": 0.5}
    inputs = extract_oracle_inputs(obs, info)
    oracle = GodOracle()
    labels = oracle.compute(**inputs)
    assert 0.0 <= labels["threat"] <= 1.0
    assert labels["opponent_distance"] >= 0.0


def test_extract_oracle_inputs_zero_distance_safe() -> None:
    """When rel_pos is zero, should not crash."""
    obs = np.zeros(21, dtype=np.float32)
    info: dict[str, object] = {}
    inputs = extract_oracle_inputs(obs, info)
    assert inputs["opponent_facing_me"] == 0.0
    assert inputs["i_face_opponent"] == 0.0
    assert inputs["opponent_closing_speed"] == 0.0
