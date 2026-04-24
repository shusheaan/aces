"""Verify GpuVecEnv picks up wind_sigma / wind_theta from rules.toml by default."""

import pytest


def test_load_wind_from_rules_returns_expected_keys():
    from aces.training.gpu_vec_env import _load_wind_from_rules

    w = _load_wind_from_rules()
    assert "wind_sigma" in w
    assert "wind_theta" in w
    assert isinstance(w["wind_sigma"], float)
    assert isinstance(w["wind_theta"], float)


def test_rules_toml_has_wind_fields():
    from aces.config import load_configs

    cfg = load_configs()
    noise = cfg.rules.noise
    assert hasattr(noise, "wind_sigma")
    assert hasattr(noise, "wind_theta")
    # Default rules.toml expects nonzero wind
    assert noise.wind_sigma > 0.0, (
        "rules.toml [noise] wind_sigma should be nonzero default"
    )


@pytest.fixture
def gpu_available(core_available):
    if not core_available:
        return False
    try:
        from aces.training.gpu_vec_env import GpuVecEnv

        env = GpuVecEnv(n_envs=1, mppi_samples=8, mppi_horizon=4)
        env.close()
        return True
    except Exception:
        return False


def test_gpu_vec_env_default_uses_rules_toml_wind(gpu_available):
    if not gpu_available:
        pytest.skip("GPU/extension unavailable")
    from aces.training.gpu_vec_env import GpuVecEnv

    # No explicit wind_sigma → reads rules.toml
    env = GpuVecEnv(n_envs=2, mppi_samples=8, mppi_horizon=4)
    env.close()


def test_gpu_vec_env_explicit_wind_overrides_rules_toml(gpu_available):
    if not gpu_available:
        pytest.skip("GPU/extension unavailable")
    from aces.training.gpu_vec_env import GpuVecEnv

    env = GpuVecEnv(
        n_envs=2,
        mppi_samples=8,
        mppi_horizon=4,
        wind_sigma=0.7,
        wind_theta=1.5,
    )
    env.close()
