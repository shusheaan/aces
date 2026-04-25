# Runpod GPU Development Setup

All Dockerfiles now live under `docker/`:

- `docker/Dockerfile.dev-base`
  - Runpod-compatible base image for cloud development.
  - Installs Rust, Poetry, Maturin, Vulkan userspace libraries, `tmux`, `zsh`,
    `ripgrep`, Node `25.9.0`, npm `11.13.0`, `openssh-client`, and `Claude Code`.
- `docker/Dockerfile.aces`
  - Thin ACES workspace image on top of the dev base.
  - Does not bake the repo into the image.
  - Defaults to the startup-script flow: clone ACES into `/workspace/aces`,
    then bootstrap dependencies inside the Pod.
- `docker/Dockerfile.train`
  - Standalone CPU training image for headless local or batch runs.

## Why this layout

The cloud-dev case and the training-image case are different:

- `Dockerfile.dev-base` is your reusable workstation environment.
  - Shell, dotfiles, git/ssh tools, editors, language runtimes, Claude Code.
- `Dockerfile.aces` is the ACES-flavored workstation image.
  - It keeps the environment opinionated for this repo, but still clones the
    repo at Pod startup so the cloud machine can `pull`, create branches, and
    `push` like a normal remote dev box.
- `Dockerfile.train` is for packaging a reproducible runtime image.
  - It bakes code into the image on purpose because it targets batch training,
    not interactive cloud development.

## Base image and toolchain versions

Pinned as of `2026-04-24`:

- Runpod base image:
  - `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Node:
  - `25.9.0`
- npm:
  - `11.13.0`

If you want the safer LTS lane instead of the latest current release, rebuild
the dev image with:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.dev-base \
  --build-arg NODE_VERSION=24.15.0 \
  --build-arg NPM_VERSION=11.13.0 \
  -t YOUR_REGISTRY/aces-dev-base:lts \
  --push .
```

## Build and push

Build the reusable development image first:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.dev-base \
  -t YOUR_REGISTRY/aces-dev-base:latest \
  --push .
```

Then build the ACES workspace image on top of it:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.aces \
  --build-arg DEV_BASE_IMAGE=YOUR_REGISTRY/aces-dev-base:latest \
  -t YOUR_REGISTRY/aces-runpod:latest \
  --push .
```

Use the headless training image for local or batch jobs:

```bash
docker build -f docker/Dockerfile.train -t aces-train .
```

`linux/amd64` is important for Runpod GPU Pods.

## Dotfiles / personal environment

You mentioned reusing dotfiles from another project. There are two supported
ways to manage that now:

1. Bake dotfiles into the base image:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.dev-base \
  --build-arg DOTFILES_REPO=https://github.com/YOU/gral-dotfiles.git \
  --build-arg DOTFILES_REF=main \
  -t YOUR_REGISTRY/aces-dev-base:latest \
  --push .
```

2. Apply dotfiles at Pod startup:

- Set `ACES_DOTFILES_REPO` to a git URL, or
- Mount or clone your dotfiles into a directory and set `ACES_DOTFILES_DIR`.

The helper script `docker/apply-dotfiles.sh` copies common top-level shell/git
files and merges `.config/` plus `.local/bin/` into the container home.

## Runpod startup behavior

The image entrypoint is `docker/runpod-start.sh`. On Pod boot it can:

- Persist Claude config, git config, SSH config or keys, and shell history into
  `/workspace/.dev-home`.
- Clone your dotfiles repo if `ACES_DOTFILES_REPO` is set.
- Clone the project repo if `ACES_REPO_URL` is set.
- Checkout a requested branch or ref if `ACES_REPO_REF` is set.
- Optionally run a safe startup `git pull --ff-only` if `ACES_GIT_AUTO_PULL=1`.
- Run `poetry install` and rebuild the GPU extension if
  `ACES_BOOTSTRAP_PROJECT=1`. The bootstrap marker is tied to the current git
  commit, so a newly pulled commit will trigger a rebuild automatically.

Useful environment variables:

- `ACES_PROJECT_DIR=/workspace/aces`
- `ACES_REPO_URL=https://github.com/YOU/aces.git`
- `ACES_REPO_REF=main`
- `ACES_REPO_REMOTE=origin`
- `ACES_GIT_AUTO_PULL=1`
- `ACES_BOOTSTRAP_PROJECT=1`
- `ACES_FORCE_BOOTSTRAP=1`
- `ACES_RUN_GPU_CHECK=1`
- `POETRY_VIRTUALENVS_CREATE=true`
- `POETRY_VIRTUALENVS_IN_PROJECT=true`
- `ACES_DOTFILES_REPO=https://github.com/YOU/gral-dotfiles.git`
- `ACES_DOTFILES_REF=main`
- `ACES_DOTFILES_DIR=/workspace/dotfiles`

## Recommended Runpod flow

### Option A: use the ACES workspace image and clone on boot

Use this when you want the cloud Pod to behave like a normal remote dev box.

1. Build and push `YOUR_REGISTRY/aces-runpod:latest`.
2. In Runpod, create a custom Pod template with:
   - Image: `YOUR_REGISTRY/aces-runpod:latest`
   - Volume mount path: `/workspace`
   - Ports: `22/tcp`, `8888/http`
3. Set environment variables:
   - `ACES_REPO_URL=https://github.com/YOU/aces.git`
   - `ACES_REPO_REF=main`
   - `ACES_PROJECT_DIR=/workspace/aces`
   - `ACES_BOOTSTRAP_PROJECT=1`
   - `ACES_GIT_AUTO_PULL=1`
   - `POETRY_VIRTUALENVS_CREATE=true`
   - `POETRY_VIRTUALENVS_IN_PROJECT=true`
4. Start the Pod.
5. Open the terminal or SSH in.
6. `cd /workspace/aces`
7. `claude`

If you want GPU validation during startup, also set:

- `ACES_RUN_GPU_CHECK=1`

### Option B: use the dev-base image and clone manually

Use this when you want one general-purpose cloud workstation.

1. Build and push `YOUR_REGISTRY/aces-dev-base:latest`.
2. Create a Runpod template pointing to that image.
3. Start a new Pod with `/workspace` as the persistent volume.
4. SSH in or open the terminal.
5. Clone the repo:

```bash
cd /workspace
git clone YOUR_REPO_URL aces
cd aces
poetry install --with dev
poetry run maturin develop --release --features gpu
bash scripts/check_gpu_setup.sh
claude
```

## Running ACES on GPU inside the Pod

For the GPU-backed curriculum path in this repo:

```bash
cd /workspace/aces
poetry run python scripts/run.py --mode curriculum --use-gpu-env --n-envs 16
```

The simpler headless server entrypoint is still available:

```bash
poetry run python scripts/train_server.py --n-envs 8 --device cuda
```

## Operational notes

- The project's GPU path uses `wgpu` and expects Linux + NVIDIA through
  Vulkan, not just a CUDA-only Python environment.
- The dev base image installs `libvulkan1` and `vulkan-tools` for that reason.
- `Claude Code` state is persisted under `/workspace/.dev-home`, so restarting
  the Pod does not wipe your login and history.
- Git state under `~/.gitconfig`, `~/.git-credentials`, and `~/.ssh` is also
  persisted there, which is what makes remote `pull` and `push` practical.
- If you update the repo significantly, rerun:

```bash
poetry install --with dev
poetry run maturin develop --release --features gpu
```

- If bootstrapping fails while uninstalling a system package such as
  `pyparsing`, Poetry is modifying the Runpod base image's system Python. Keep
  the project isolated by setting `POETRY_VIRTUALENVS_CREATE=true` and
  `POETRY_VIRTUALENVS_IN_PROJECT=true`, then restart with
  `ACES_FORCE_BOOTSTRAP=1`.
