#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTEST_CORE_FILES=(
    tests/test_config.py
    tests/test_obs_layout.py
    tests/test_denormalize_action.py
    tests/test_action_normalization_consistency.py
    tests/test_reward_consistency.py
    tests/test_dynamics.py
    tests/test_env.py
    tests/test_observation_consistency.py
    tests/test_gpu_wind_plumbing.py
    tests/test_gpu_reward_plumbing.py
    tests/test_curriculum_gpu_opt.py
    tests/test_gpu_vec_env.py
)

PYTEST_CORE_K="not fpv and not pursuit_evasive"

if [[ -x ".venv/bin/python" ]]; then
    exec .venv/bin/python -m pytest -q "${PYTEST_CORE_FILES[@]}" -k "$PYTEST_CORE_K"
fi

if command -v poetry >/dev/null 2>&1; then
    exec poetry run python -m pytest -q "${PYTEST_CORE_FILES[@]}" -k "$PYTEST_CORE_K"
fi

exec python3 -m pytest -q "${PYTEST_CORE_FILES[@]}" -k "$PYTEST_CORE_K"
