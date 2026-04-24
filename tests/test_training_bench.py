"""Smoke test for the training throughput benchmark script."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "bench_training_throughput.py"


def test_script_exists():
    assert SCRIPT.exists()


@pytest.fixture
def core_available() -> bool:
    try:
        import aces._core  # noqa: F401

        return True
    except ImportError:
        return False


def test_script_imports_and_runs_help(core_available):
    # Runs `--help` which doesn't need GPU
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=REPO_ROOT,
    )
    # Should print help and exit 0
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    assert "throughput" in result.stdout.lower() or "benchmark" in result.stdout.lower()


def test_script_runs_small_config(core_available):
    if not core_available:
        pytest.skip("aces._core not built")

    # Might skip internally if GPU adapter unavailable
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--configs", "small"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    # Exit code: 0 if GPU present and ran, 1 if GPU unavailable
    # Both are acceptable - we just want no crashes
    assert result.returncode in (0, 1), (
        f"unexpected exit code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
