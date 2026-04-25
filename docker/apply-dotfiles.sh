#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="${1:-}"
TARGET_HOME="${2:-${HOME:-/root}}"

log() {
    printf '[dotfiles] %s\n' "$*"
}

copy_file_if_present() {
    local rel="$1"
    local src="${SOURCE_DIR}/${rel}"
    local dst="${TARGET_HOME}/${rel}"

    if [[ ! -e "${src}" && ! -L "${src}" ]]; then
        return 0
    fi

    mkdir -p "$(dirname "${dst}")"
    cp -a "${src}" "${dst}"
    log "copied ${rel}"
}

merge_dir_if_present() {
    local rel="$1"
    local src="${SOURCE_DIR}/${rel}"
    local dst="${TARGET_HOME}/${rel}"

    if [[ ! -d "${src}" ]]; then
        return 0
    fi

    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
    log "merged ${rel}/"
}

if [[ -z "${SOURCE_DIR}" || ! -d "${SOURCE_DIR}" ]]; then
    log "no dotfiles directory provided; skipping"
    exit 0
fi

mkdir -p "${TARGET_HOME}"

copy_file_if_present .aliases
copy_file_if_present .bash_profile
copy_file_if_present .bashrc
copy_file_if_present .gitconfig
copy_file_if_present .gitignore_global
copy_file_if_present .inputrc
copy_file_if_present .profile
copy_file_if_present .p10k.zsh
copy_file_if_present .tmux.conf
copy_file_if_present .vimrc
copy_file_if_present .zprofile
copy_file_if_present .zshenv
copy_file_if_present .zshrc

merge_dir_if_present .config
merge_dir_if_present .local/bin

if [[ "${DOTFILES_RUN_INSTALL:-0}" == "1" && -x "${SOURCE_DIR}/install.sh" ]]; then
    log "running install.sh"
    HOME="${TARGET_HOME}" "${SOURCE_DIR}/install.sh"
fi
