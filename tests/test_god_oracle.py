"""Tests for aces.god_oracle — ground truth label computation."""

from aces.god_oracle import GodOracle


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
