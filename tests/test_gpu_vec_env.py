"""Smoke tests for GpuVecEnv SB3 wrapper.

These tests require (a) the Rust extension built with --features gpu,
and (b) a GPU adapter available at runtime. If either is missing, tests skip.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gpu_env(gpu_available: bool):
    if not gpu_available:
        pytest.skip("GPU not available")
    from aces.training.gpu_vec_env import GpuVecEnv

    env = GpuVecEnv(n_envs=4, mppi_samples=32, mppi_horizon=10)
    yield env
    env.close()


def test_observation_space(gpu_env):
    assert gpu_env.observation_space.shape == (21,)
    assert gpu_env.observation_space.dtype == np.float32


def test_action_space(gpu_env):
    assert gpu_env.action_space.shape == (4,)
    assert gpu_env.action_space.dtype == np.float32
    assert np.allclose(gpu_env.action_space.low, -1.0)
    assert np.allclose(gpu_env.action_space.high, 1.0)


def test_reset_shape(gpu_env):
    obs = gpu_env.reset()
    assert obs.shape == (4, 21)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_step_shape(gpu_env):
    gpu_env.reset()
    actions = np.zeros((4, 4), dtype=np.float32)
    gpu_env.step_async(actions)
    obs, rewards, dones, infos = gpu_env.step_wait()

    assert obs.shape == (4, 21)
    assert obs.dtype == np.float32
    assert rewards.shape == (4,)
    assert rewards.dtype == np.float32
    assert dones.shape == (4,)
    assert dones.dtype == bool
    assert len(infos) == 4
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(rewards))


def test_step_many(gpu_env):
    gpu_env.reset()
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        actions = rng.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float32)
        gpu_env.step_async(actions)
        obs, rewards, dones, infos = gpu_env.step_wait()
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
