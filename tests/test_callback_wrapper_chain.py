"""Tests for _find_opponent_env wrapper-chain traversal (Bug #22)."""

from __future__ import annotations


from aces.training.callbacks import _find_opponent_env


class _FakeBatchedOpponentVecEnv:
    """Minimal stand-in for BatchedOpponentVecEnv — has the marker attributes."""

    has_opponent = False

    def set_opponent_policy(self, policy):
        self.has_opponent = True

    def set_opponent_weights(self, state_dict):
        pass


class _FakeVecNormalize:
    """Minimal stand-in for VecNormalize — has venv but NOT set_opponent_policy."""

    def __init__(self, inner):
        self.venv = inner


class _FakePlainVecEnv:
    """Plain VecEnv with no venv and no opponent attributes."""

    pass


def test_find_opponent_env_direct():
    """If the top-level env has set_opponent_policy, return it directly."""
    env = _FakeBatchedOpponentVecEnv()
    result = _find_opponent_env(env)
    assert result is env


def test_find_opponent_env_through_vec_normalize():
    """Walking through a VecNormalize wrapper finds the inner BatchedOpponentVecEnv."""
    inner = _FakeBatchedOpponentVecEnv()
    wrapped = _FakeVecNormalize(inner)
    result = _find_opponent_env(wrapped)
    assert result is inner


def test_find_opponent_env_nested_wrappers():
    """Works through two layers of VecNormalize-like wrappers."""
    inner = _FakeBatchedOpponentVecEnv()
    mid = _FakeVecNormalize(inner)
    outer = _FakeVecNormalize(mid)
    result = _find_opponent_env(outer)
    assert result is inner


def test_find_opponent_env_no_opponent_env():
    """Returns None gracefully when no env in the chain has set_opponent_policy."""
    plain = _FakePlainVecEnv()
    result = _find_opponent_env(plain)
    assert result is None


def test_find_opponent_env_vec_normalize_over_plain():
    """VecNormalize wrapping a plain VecEnv (no opponent) → None."""
    plain = _FakePlainVecEnv()
    wrapped = _FakeVecNormalize(plain)
    result = _find_opponent_env(wrapped)
    assert result is None
