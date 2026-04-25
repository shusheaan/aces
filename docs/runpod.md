# Runpod GPU Development Setup

This repo now includes two Dockerfiles under `docker/` so you can separate the
cloud development environment from the ACES project image:

- `docker/Dockerfile.dev-base`
  - Runpod-compatible base image for cloud development.
  - Installs Rust, Poetry, Maturin, Vulkan userspace libraries, `tmux`, `zsh`,
    `ripgrep`, Node `25.9.0`, npm `11.13.0`, and `Claude Code`.
- `docker/Dockerfile.aces`
  - Extends the dev base image.
  - Installs ACES Python dependencies.
  - Builds the GPU-enabled Rust extension with `maturin develop --features gpu`.
  - Bakes the current repo snapshot into `/opt/aces/image-src`.

## Why two Dockerfiles

Keep the layers separate:

- Personal environment changes belong in `Dockerfile.dev-base`.
  - Shell, dotfiles, `Claude Code`, git tools, editors, language runtimes.
- Project-specific dependencies belong in `Dockerfile.aces`.
  - ACES source, Poetry dependencies, Rust extension build, test helpers.

That gives you two deployment patterns:

- Reusable dev image:
  - Start a Pod, clone any repo you want, and work normally.
- Project image:
  - Start a Pod with ACES already baked in and ready to bootstrap into
    `/workspace/aces`.

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

Then build the ACES project image on top of it:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.aces \
  --build-arg DEV_BASE_IMAGE=YOUR_REGISTRY/aces-dev-base:latest \
  -t YOUR_REGISTRY/aces-runpod:latest \
  --push .
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
- Mount/clone your dotfiles into a directory and set `ACES_DOTFILES_DIR`.

The helper script `docker/apply-dotfiles.sh` copies common top-level shell/git
files and merges `.config/` plus `.local/bin/` into the container home.

## Runpod startup behavior

The image entrypoint is `docker/runpod-start.sh`. On Pod boot it can:

- Persist Claude config and shell history into `/workspace/.dev-home`.
- Clone your dotfiles repo if `ACES_DOTFILES_REPO` is set.
- Clone the project repo if `ACES_REPO_URL` is set.
- Or copy the baked image snapshot from `/opt/aces/image-src` into
  `/workspace/aces` if `ACES_BOOTSTRAP_FROM_IMAGE=1`.
- Optionally run `poetry install` and rebuild the GPU extension if
  `ACES_BOOTSTRAP_PROJECT=1`.

Useful environment variables:

- `ACES_PROJECT_DIR=/workspace/aces`
- `ACES_REPO_URL=https://github.com/YOU/aces.git`
- `ACES_REPO_REF=main`
- `ACES_BOOTSTRAP_FROM_IMAGE=1`
- `ACES_BOOTSTRAP_PROJECT=1`
- `ACES_RUN_GPU_CHECK=1`
- `ACES_DOTFILES_REPO=https://github.com/YOU/gral-dotfiles.git`
- `ACES_DOTFILES_REF=main`
- `ACES_DOTFILES_DIR=/workspace/dotfiles`

## Recommended Runpod flow

### Option A: use the ACES image directly

Use this when you want the fastest first boot for this repo.

1. Build and push `YOUR_REGISTRY/aces-runpod:latest`.
2. In Runpod, create a custom Pod template with:
   - Image: `YOUR_REGISTRY/aces-runpod:latest`
   - Volume mount path: `/workspace`
   - Ports: `22/tcp`, `8888/http`
3. Set environment variables:
   - `ACES_BOOTSTRAP_FROM_IMAGE=1`
   - `ACES_PROJECT_DIR=/workspace/aces`
4. Start the Pod.
5. Open the terminal or SSH in.
6. `cd /workspace/aces`
7. `claude`

If you want a fresh rebuild on first boot, also set:

- `ACES_BOOTSTRAP_PROJECT=1`
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
- If you update the repo significantly, rerun:

```bash
poetry install --with dev
poetry run maturin develop --release --features gpu
```
