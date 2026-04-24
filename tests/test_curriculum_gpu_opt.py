"""Smoke test: CurriculumTrainer with ``use_gpu_env=True`` instantiates correctly.

The CPU path must remain unaffected when the flag is off. The GPU path only
runs if the Rust extension was built with ``--features gpu`` and a GPU adapter
is available at runtime; otherwise the GPU-specific test skips cleanly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def core_available() -> bool:
    try:
        import aces._core  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture(scope="module")
def gpu_available() -> bool:
    try:
        from aces.training.gpu_vec_env import GpuVecEnv
    except (ImportError, RuntimeError):
        return False
    try:
        env = GpuVecEnv(n_envs=1, mppi_samples=8, mppi_horizon=4)
        env.close()
        return True
    except Exception:
        return False


def _minimal_stages() -> list[dict]:
    """Tiny two-stage curriculum for construction-only smoke tests."""
    return [
        {"task": "dogfight", "timesteps": 256},
        {"task": "dogfight", "timesteps": 256},
    ]


def test_curriculum_trainer_accepts_gpu_flag_default_off(core_available: bool):
    """Flag defaults off; existing CPU path untouched."""
    if not core_available:
        pytest.skip("aces._core not built (rebuild with: poetry run maturin develop)")
    from aces.training.curriculum_trainer import CurriculumTrainer

    trainer = CurriculumTrainer(
        stages=_minimal_stages(),
        n_envs=1,
    )
    assert trainer is not None
    assert trainer._use_gpu_env is False


def test_curriculum_trainer_stores_gpu_params(core_available: bool):
    """GPU kwargs are stored on the trainer even when the flag is off."""
    if not core_available:
        pytest.skip("aces._core not built (rebuild with: poetry run maturin develop)")
    from aces.training.curriculum_trainer import CurriculumTrainer

    trainer = CurriculumTrainer(
        stages=_minimal_stages(),
        n_envs=1,
        use_gpu_env=False,
        gpu_mppi_samples=64,
        gpu_mppi_horizon=8,
        gpu_noise_std=0.05,
    )
    assert trainer._gpu_mppi_samples == 64
    assert trainer._gpu_mppi_horizon == 8
    assert trainer._gpu_noise_std == 0.05


def test_curriculum_trainer_gpu_env_factory(gpu_available: bool):
    """With ``use_gpu_env=True`` and a GPU, ``_make_vec_env`` returns a
    :class:`VecNormalize`-wrapped :class:`GpuVecEnv`."""
    if not gpu_available:
        pytest.skip("GPU not available")

    from stable_baselines3.common.vec_env import VecNormalize

    from aces.training.curriculum_trainer import CurriculumTrainer

    trainer = CurriculumTrainer(
        stages=_minimal_stages(),
        n_envs=2,
        use_gpu_env=True,
        gpu_mppi_samples=16,
        gpu_mppi_horizon=5,
    )

    vec_env = trainer._make_vec_env("dogfight")
    try:
        assert isinstance(vec_env, VecNormalize)
        assert vec_env.num_envs == 2
        obs = vec_env.reset()
        assert obs.shape == (2, 21)  # type: ignore[union-attr]
    finally:
        vec_env.close()
