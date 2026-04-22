#!/usr/bin/env bash
# Install git pre-commit hook
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK_PATH" << 'HOOK'
#!/usr/bin/env bash
exec "$(git rev-parse --show-toplevel)/scripts/pre-commit.sh"
HOOK

chmod +x "$HOOK_PATH"
chmod +x "$REPO_ROOT/scripts/pre-commit.sh"

echo "Pre-commit hook installed successfully."
