#!/usr/bin/env bash
set -e

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

failed=0

step() {
    printf "${YELLOW}▶ %s${NC}\n" "$1"
}

pass() {
    printf "${GREEN}✓ %s${NC}\n" "$1"
}

fail() {
    printf "${RED}✗ %s${NC}\n" "$1"
    failed=1
}

# ── Rust ──────────────────────────────────────────────

step "cargo fmt --check"
if cargo fmt --all -- --check 2>/dev/null; then
    pass "cargo fmt"
else
    fail "cargo fmt — run 'cargo fmt --all' to fix"
fi

step "cargo clippy"
if cargo clippy --workspace --all-targets -- -D warnings 2>/dev/null; then
    pass "cargo clippy"
else
    fail "cargo clippy"
fi

step "cargo test core"
if cargo test -p aces-sim-core -p aces-batch-sim -p aces-mppi -p aces-estimator --lib --quiet 2>/dev/null; then
    pass "cargo test core"
else
    fail "cargo test core"
fi

# ── Python ────────────────────────────────────────────

step "ruff format --check"
if poetry run ruff format --check aces/ tests/ scripts/ 2>/dev/null; then
    pass "ruff format"
else
    fail "ruff format — run 'poetry run ruff format aces/ tests/ scripts/' to fix"
fi

step "ruff check"
if poetry run ruff check aces/ tests/ scripts/ 2>/dev/null; then
    pass "ruff check"
else
    fail "ruff check — run 'poetry run ruff check --fix aces/ tests/ scripts/' to fix"
fi

step "pytest core"
if bash scripts/test-python-core.sh 2>/dev/null; then
    pass "pytest core"
else
    fail "pytest core"
fi

# ── Result ────────────────────────────────────────────

echo ""
if [ $failed -ne 0 ]; then
    printf "${RED}Pre-commit checks failed. Commit aborted.${NC}\n"
    exit 1
else
    printf "${GREEN}All pre-commit checks passed.${NC}\n"
fi
