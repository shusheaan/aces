#!/usr/bin/env bash

set -euo pipefail

log() {
    printf '[runpod-start] %s\n' "$*"
}

WORKSPACE_ROOT="${ACES_WORKSPACE_ROOT:-/workspace}"
PROJECT_DIR="${ACES_PROJECT_DIR:-${WORKSPACE_ROOT}/aces}"
IMAGE_SOURCE_DIR="${ACES_IMAGE_SOURCE_DIR:-/opt/aces/image-src}"
STATE_DIR="${ACES_STATE_DIR:-${WORKSPACE_ROOT}/.dev-home}"
DOTFILES_DIR="${ACES_DOTFILES_DIR:-}"
DOTFILES_REPO="${ACES_DOTFILES_REPO:-}"
DOTFILES_REF="${ACES_DOTFILES_REF:-main}"
REPO_URL="${ACES_REPO_URL:-}"
REPO_REF="${ACES_REPO_REF:-}"
REPO_REMOTE="${ACES_REPO_REMOTE:-origin}"
BOOTSTRAP_FROM_IMAGE="${ACES_BOOTSTRAP_FROM_IMAGE:-0}"
BOOTSTRAP_PROJECT="${ACES_BOOTSTRAP_PROJECT:-0}"
FORCE_BOOTSTRAP="${ACES_FORCE_BOOTSTRAP:-0}"
AUTO_PULL="${ACES_GIT_AUTO_PULL:-0}"
RUN_GPU_CHECK="${ACES_RUN_GPU_CHECK:-0}"
HOME_DIR="${HOME:-/root}"
BOOTSTRAP_MARKER=".aces-runpod-ready"

persist_path() {
    local rel="$1"
    local kind="${2:-file}"
    local src="${HOME_DIR}/${rel}"
    local dst="${STATE_DIR}/${rel}"

    mkdir -p "${HOME_DIR}"
    mkdir -p "$(dirname "${dst}")"

    if [[ -L "${src}" ]]; then
        return 0
    fi

    if [[ -e "${src}" || -d "${src}" ]]; then
        if [[ ! -e "${dst}" && ! -d "${dst}" ]]; then
            cp -a "${src}" "${dst}"
        fi
        rm -rf "${src}"
    else
        if [[ "${kind}" == "dir" ]]; then
            mkdir -p "${dst}"
        else
            touch "${dst}"
        fi
    fi

    ln -s "${dst}" "${src}"

    if [[ "${rel}" == ".ssh" ]]; then
        chmod 700 "${dst}"
    fi
}

ensure_persistent_home() {
    mkdir -p "${WORKSPACE_ROOT}" "${STATE_DIR}"
    persist_path .claude dir
    persist_path .claude.json
    persist_path .config dir
    persist_path .bash_history
    persist_path .gitconfig
    persist_path .git-credentials
    persist_path .ssh dir
    persist_path .zsh_history
}

clone_dotfiles_if_requested() {
    if [[ -z "${DOTFILES_REPO}" ]]; then
        return 0
    fi

    if [[ -z "${DOTFILES_DIR}" ]]; then
        DOTFILES_DIR="${WORKSPACE_ROOT}/dotfiles"
    fi

    if [[ ! -d "${DOTFILES_DIR}/.git" ]]; then
        log "cloning dotfiles repo ${DOTFILES_REPO}"
        git clone --depth 1 --branch "${DOTFILES_REF}" "${DOTFILES_REPO}" "${DOTFILES_DIR}"
    fi
}

apply_dotfiles_if_present() {
    if [[ -n "${DOTFILES_DIR}" && -d "${DOTFILES_DIR}" ]]; then
        /opt/aces/bin/apply-dotfiles.sh "${DOTFILES_DIR}" "${HOME_DIR}"
    fi
}

bootstrap_project_from_git() {
    if [[ -z "${REPO_URL}" || -e "${PROJECT_DIR}" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "${PROJECT_DIR}")"
    log "cloning project ${REPO_URL} -> ${PROJECT_DIR}"
    git clone "${REPO_URL}" "${PROJECT_DIR}"
}

bootstrap_project_from_image() {
    if [[ "${BOOTSTRAP_FROM_IMAGE}" != "1" || -e "${PROJECT_DIR}" || ! -d "${IMAGE_SOURCE_DIR}" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "${PROJECT_DIR}")"
    log "copying baked project snapshot -> ${PROJECT_DIR}"
    cp -a "${IMAGE_SOURCE_DIR}" "${PROJECT_DIR}"
}

checkout_requested_ref() {
    if [[ -z "${REPO_REF}" || ! -d "${PROJECT_DIR}/.git" ]]; then
        return 0
    fi

    if git -C "${PROJECT_DIR}" show-ref --verify --quiet "refs/heads/${REPO_REF}"; then
        git -C "${PROJECT_DIR}" checkout "${REPO_REF}"
        return 0
    fi

    if git -C "${PROJECT_DIR}" ls-remote --exit-code --heads "${REPO_REMOTE}" "${REPO_REF}" >/dev/null 2>&1; then
        log "checking out branch ${REPO_REMOTE}/${REPO_REF}"
        git -C "${PROJECT_DIR}" fetch "${REPO_REMOTE}" "${REPO_REF}"
        git -C "${PROJECT_DIR}" checkout -B "${REPO_REF}" --track "${REPO_REMOTE}/${REPO_REF}"
        return 0
    fi

    if git -C "${PROJECT_DIR}" rev-parse --verify "${REPO_REF}" >/dev/null 2>&1; then
        git -C "${PROJECT_DIR}" checkout "${REPO_REF}"
        return 0
    fi

    log "fetching ref ${REPO_REF}"
    git -C "${PROJECT_DIR}" fetch "${REPO_REMOTE}" "${REPO_REF}"
    git -C "${PROJECT_DIR}" checkout FETCH_HEAD
}

sync_project_repo() {
    local branch=""

    if [[ "${AUTO_PULL}" != "1" || ! -d "${PROJECT_DIR}/.git" ]]; then
        return 0
    fi

    if [[ -n "$(git -C "${PROJECT_DIR}" status --porcelain)" ]]; then
        log "skipping git pull because the worktree has local changes"
        return 0
    fi

    if ! branch="$(git -C "${PROJECT_DIR}" symbolic-ref --quiet --short HEAD 2>/dev/null)"; then
        log "skipping git pull because HEAD is detached"
        return 0
    fi

    log "pulling latest changes for ${branch}"
    if git -C "${PROJECT_DIR}" rev-parse --verify "refs/remotes/${REPO_REMOTE}/${branch}" >/dev/null 2>&1; then
        git -C "${PROJECT_DIR}" pull --ff-only "${REPO_REMOTE}" "${branch}"
        return 0
    fi

    git -C "${PROJECT_DIR}" pull --ff-only
}

bootstrap_project_env() {
    local marker="${PROJECT_DIR}/${BOOTSTRAP_MARKER}"
    local current_revision=""

    if [[ "${BOOTSTRAP_PROJECT}" != "1" || ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
        return 0
    fi

    if [[ -d "${PROJECT_DIR}/.git" ]]; then
        current_revision="$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || true)"
    fi

    if [[ -f "${marker}" && "${FORCE_BOOTSTRAP}" != "1" ]]; then
        if [[ -n "${current_revision}" ]] && grep -Fxq "${current_revision}" "${marker}"; then
            log "project bootstrap already completed for ${current_revision}"
            return 0
        fi

        if [[ -z "${current_revision}" ]]; then
            log "project bootstrap already completed"
            return 0
        fi
    fi

    log "installing Python dependencies"
    (
        cd "${PROJECT_DIR}"
        export POETRY_VIRTUALENVS_CREATE="${POETRY_VIRTUALENVS_CREATE:-true}"
        export POETRY_VIRTUALENVS_IN_PROJECT="${POETRY_VIRTUALENVS_IN_PROJECT:-true}"
        if command -v python3.12 >/dev/null 2>&1; then
            poetry env use python3.12
        else
            poetry env use python3
        fi
        poetry install --with dev --no-interaction --no-ansi
        log "building GPU-enabled Rust extension"
        poetry run maturin develop --release --features gpu
        if [[ "${RUN_GPU_CHECK}" == "1" && -x scripts/check_gpu_setup.sh ]]; then
            log "running GPU validation"
            bash scripts/check_gpu_setup.sh
        fi
        if [[ -n "${current_revision}" ]]; then
            printf '%s\n' "${current_revision}" > "${marker}"
        else
            date -u +"%Y-%m-%dT%H:%M:%SZ" > "${marker}"
        fi
    )
}

main() {
    ensure_persistent_home
    clone_dotfiles_if_requested
    apply_dotfiles_if_present
    bootstrap_project_from_git
    bootstrap_project_from_image
    checkout_requested_ref
    sync_project_repo
    bootstrap_project_env

    if [[ $# -gt 0 ]]; then
        exec "$@"
    fi

    if [[ -x /start.sh ]]; then
        exec /start.sh
    fi

    exec sleep infinity
}

main "$@"
