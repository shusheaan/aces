"""Tests for DroneDogfightEnv opponent method shims (Bug #26).

Verifies that set_opponent_weights and set_opponent_policy are directly
callable on DroneDogfightEnv, so that callbacks work when n_envs=1 and
no BatchedOpponentVecEnv is in the wrapper chain.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aces.env.dogfight import DroneDogfightEnv


def _make_env() -> DroneDogfightEnv:
    """Create a DroneDogfightEnv using default configs."""
    return DroneDogfightEnv()


def test_set_opponent_policy_exists():
    """DroneDogfightEnv.set_opponent_policy must exist."""
    assert callable(getattr(DroneDogfightEnv, "set_opponent_policy", None))


def test_set_opponent_weights_exists():
    """DroneDogfightEnv.set_opponent_weights must exist."""
    assert callable(getattr(DroneDogfightEnv, "set_opponent_weights", None))


def test_set_opponent_policy_sets_mode(tmp_path):
    """set_opponent_policy stores the policy and switches to 'policy' mode."""
    env = _make_env()
    try:
        mock_policy = MagicMock()
        env.set_opponent_policy(mock_policy)
        assert env._opponent_policy is mock_policy
        assert env._opponent_mode == "policy"
    finally:
        env.close()


def test_set_opponent_weights_delegates(tmp_path):
    """set_opponent_weights calls load_state_dict on the stored policy."""
    env = _make_env()
    try:
        mock_policy = MagicMock()
        env.set_opponent_policy(mock_policy)

        fake_state = {"fc.weight": MagicMock()}
        env.set_opponent_weights(fake_state)
        mock_policy.load_state_dict.assert_called_once_with(fake_state)
    finally:
        env.close()


def test_set_opponent_weights_noop_without_policy():
    """set_opponent_weights is a no-op when no policy is set (no crash)."""
    env = _make_env()
    try:
        # Should not raise even if _opponent_policy is None
        env.set_opponent_weights({"dummy": None})
    finally:
        env.close()
