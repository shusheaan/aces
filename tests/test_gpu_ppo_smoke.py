"""Pytest for GPU PPO smoke test.

Runs the script with minimal settings. Skips if GpuVecEnv unavailable or
if GPU can't be acquired at runtime.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_gpu_ppo.py"


def test_smoke_script_exists():
    assert SMOKE_SCRIPT.exists(), f"missing {SMOKE_SCRIPT}"


@pytest.fixture(scope="module")
def gpu_available() -> bool:
    """Check if aces._core.GpuVecEnv + GPU adapter both available."""
    try:
        from aces._core import GpuVecEnv  # noqa: F401
    except ImportError:
        return False
    try:
        from aces.training.gpu_vec_env import GpuVecEnv

        env = GpuVecEnv(n_envs=1, mppi_samples=8, mppi_horizon=4)
        env.close()
        return True
    except Exception:
        return False


def test_ppo_smoke_runs(gpu_available):
    if not gpu_available:
        pytest.skip("GpuVecEnv/GPU not available")

    # n_steps * n_envs = 32 * 4 = 128 rollout buffer; 256 timesteps = 2
    # rollout iterations so PPO actually triggers gradient updates.
    result = subprocess.run(
        [
            sys.executable,
            str(SMOKE_SCRIPT),
            "--timesteps",
            "256",
            "--n-envs",
            "4",
            "--mppi-samples",
            "16",
            "--mppi-horizon",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min upper bound
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, (
        f"smoke script failed (code {result.returncode})\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "Training completed" in result.stdout
    assert "Final policy produces finite actions" in result.stdout
