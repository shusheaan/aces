"""Shared pytest fixtures for ACES test suite.

Two capability probes are exposed as module-scoped fixtures:
  - `core_available` -> bool: True iff aces._core (the Rust extension) can be imported.
  - `gpu_available`  -> bool: True iff a GpuVecEnv instance can actually be constructed
                              (implies both aces._core AND a working GPU adapter).

Tests that require the Rust extension or GPU should accept one of these
fixtures as a parameter and pytest.skip(...) when it is False.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def core_available() -> bool:
    try:
        import aces._core  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gpu_available(core_available: bool) -> bool:
    if not core_available:
        return False
    try:
        from aces.training.gpu_vec_env import GpuVecEnv

        env = GpuVecEnv(n_envs=1, mppi_samples=8, mppi_horizon=4)
        env.close()
        return True
    except Exception:
        return False
