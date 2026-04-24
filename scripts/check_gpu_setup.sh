#!/usr/bin/env bash
# check_gpu_setup.sh — end-to-end GPU MPPI setup validation.
#
# Walks through: GPU hardware probe, Rust tests, Python extension load,
# Python smoke tests. Each step reports OK / FAIL / SKIP with a short
# diagnostic. Exits 0 if everything that CAN work does; exits 1 if any
# required stage fails.
#
# Usage:
#   bash scripts/check_gpu_setup.sh
#
# Idempotent — can be re-run after fixing issues.

set -uo pipefail  # NOTE: no -e; we want to continue past failures and report them

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Colors (skip if not a tty)
if [ -t 1 ]; then
    GREEN=$'\033[0;32m'
    RED=$'\033[0;31m'
    YELLOW=$'\033[0;33m'
    BLUE=$'\033[0;34m'
    NC=$'\033[0m'
else
    GREEN=""
    RED=""
    YELLOW=""
    BLUE=""
    NC=""
fi

TOTAL=0
PASS=0
FAIL=0
SKIP=0

section() {
    TOTAL=$((TOTAL + 1))
    echo
    echo "${BLUE}[$TOTAL] $1${NC}"
}

report_ok() {
    PASS=$((PASS + 1))
    echo "    ${GREEN}OK${NC}   $1"
}

report_fail() {
    FAIL=$((FAIL + 1))
    echo "    ${RED}FAIL${NC} $1"
}

report_skip() {
    SKIP=$((SKIP + 1))
    echo "    ${YELLOW}SKIP${NC} $1"
}

echo "=== ACES GPU MPPI Setup Check ==="
echo "Repo: $REPO_ROOT"

# ---------- Stage 1: cargo + rustc available ----------
section "Rust toolchain"
if command -v cargo >/dev/null 2>&1; then
    RUSTV=$(rustc --version 2>&1 || echo "unknown")
    report_ok "cargo/rustc present ($RUSTV)"
else
    report_fail "cargo not on PATH. Install Rust: https://rustup.rs/"
    echo
    echo "=== Halting: cargo is required ==="
    exit 1
fi

# ---------- Stage 2: GPU adapter probe ----------
section "GPU adapter probe"
PROBE_OUT=$(cargo run -p aces-batch-sim --features gpu --example gpu_probe --release 2>&1)
PROBE_RC=$?
if [ $PROBE_RC -ne 0 ]; then
    report_fail "gpu_probe build/run failed (exit $PROBE_RC). Tail:"
    echo "$PROBE_OUT" | tail -20 | sed 's/^/      /'
elif echo "$PROBE_OUT" | grep -q "No GPU adapters found"; then
    report_skip "No GPU adapter detected by wgpu."
    echo "      (Apple Silicon: should auto-detect Metal. Check your wgpu/driver setup.)"
else
    ADAPTER=$(echo "$PROBE_OUT" | grep -E "Name:" | head -1 | sed 's/.*Name: *//')
    BACKEND=$(echo "$PROBE_OUT" | grep -E "Backend:" | head -1 | sed 's/.*Backend: *//')
    report_ok "Adapter: ${ADAPTER:-unknown} (backend: ${BACKEND:-unknown})"
fi

# ---------- Stage 3: Rust tests (lib) ----------
section "Rust tests (aces-batch-sim --features gpu)"
TEST_OUT=$(cargo test -p aces-batch-sim --features gpu --release 2>&1)
TEST_RC=$?
if [ $TEST_RC -ne 0 ] || echo "$TEST_OUT" | grep -q "test result:.*FAILED"; then
    report_fail "Some Rust tests failed:"
    echo "$TEST_OUT" | grep -E "test result|FAILED" | head -10 | sed 's/^/      /'
else
    COUNTS=$(echo "$TEST_OUT" | grep -c "test result: ok")
    report_ok "$COUNTS test binaries passed"
fi

# ---------- Stage 4: Python extension ----------
section "Python extension (aces._core)"
if ! command -v poetry >/dev/null 2>&1; then
    report_skip "poetry not on PATH — Python layer unavailable."
else
    PY_OUT=$(poetry run python -c "
import importlib.util
import sys

try:
    spec = importlib.util.find_spec('aces._core')
except ImportError as e:
    # Parent package (aces/__init__.py) failed to import — almost always a
    # missing transitive dep (numpy, gymnasium, ...), not a missing extension.
    print(f'IMPORT_ERR {e}')
    sys.exit(0)

if spec is None:
    print('MISSING_CORE')
    sys.exit(0)

try:
    import aces._core  # noqa: F401
    print('OK')
except ImportError as e:
    print(f'IMPORT_ERR {e}')
    sys.exit(0)
" 2>&1)
    PY_RC=$?
    PY_LAST=$(echo "$PY_OUT" | tail -1)
    if [ $PY_RC -ne 0 ]; then
        report_fail "python probe crashed (exit $PY_RC):"
        echo "$PY_OUT" | tail -10 | sed 's/^/      /'
    elif [ "$PY_LAST" = "OK" ]; then
        report_ok "aces._core importable"
    elif [ "$PY_LAST" = "MISSING_CORE" ]; then
        report_fail "aces._core extension not built"
        echo "      Hint: poetry run maturin develop --features gpu --release"
    elif [[ "$PY_LAST" == IMPORT_ERR* ]]; then
        report_fail "aces._core import failed (likely missing Python dep)"
        echo "      Hint: poetry install  (ensure the poetry env is complete)"
        echo "      Error: ${PY_LAST#IMPORT_ERR }"
    else
        report_fail "unexpected python probe output:"
        echo "$PY_OUT" | tail -10 | sed 's/^/      /'
    fi
fi

# ---------- Stage 5: Python smoke tests ----------
section "Python smoke tests"
if ! command -v poetry >/dev/null 2>&1; then
    report_skip "poetry not on PATH"
else
    SMOKE_OUT=$(poetry run pytest tests/test_denormalize_action.py tests/test_gpu_vec_env.py tests/test_gpu_ppo_smoke.py -q 2>&1)
    SMOKE_RC=$?
    if [ $SMOKE_RC -ne 0 ] || echo "$SMOKE_OUT" | grep -qE "^(FAILED|ERROR)"; then
        report_fail "Some Python smoke tests failed:"
        echo "$SMOKE_OUT" | tail -15 | sed 's/^/      /'
    else
        SUMMARY=$(echo "$SMOKE_OUT" | grep -E "[0-9]+ (passed|skipped)" | tail -1)
        report_ok "${SUMMARY:-all smoke tests passed/skipped cleanly}"
    fi
fi

# ---------- Summary ----------
echo
echo "=== Summary ==="
echo "  Total: $TOTAL stages"
echo "  ${GREEN}OK:   $PASS${NC}"
echo "  ${RED}FAIL: $FAIL${NC}"
echo "  ${YELLOW}SKIP: $SKIP${NC}"
echo

if [ $FAIL -gt 0 ]; then
    echo "${RED}Setup has failures — see diagnostics above.${NC}"
    exit 1
fi
if [ $SKIP -gt 0 ]; then
    echo "${YELLOW}Setup works as far as it can; some stages skipped (see above).${NC}"
    exit 0
fi
echo "${GREEN}Full GPU MPPI stack is working.${NC}"
exit 0
