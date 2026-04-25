# Docker / Runpod Quickstart

This folder contains the container definitions used for local training images
and for the Runpod cloud development workflow.

For the fuller background and design notes, see [`../docs/runpod.md`](../docs/runpod.md).

## Files

- `Dockerfile.dev-base`
  - Reusable Runpod development base image.
  - Installs Rust, Poetry, Maturin, git, SSH client, Claude Code, and common CLI tools.
- `Dockerfile.aces`
  - ACES Runpod workspace image.
  - Intended for remote development.
  - The Pod clones the repo at startup instead of baking the repo into the image.
- `Dockerfile.train`
  - Standalone CPU training image for local or batch jobs.
- `runpod-start.sh`
  - Pod startup script.
  - Clones the repo, checks out the requested ref, optionally auto-pulls, and bootstraps the project.
- `apply-dotfiles.sh`
  - Optional helper to apply dotfiles into the Pod home directory.

## Which Image To Use

For Runpod cloud development, use `Dockerfile.aces`.

`Dockerfile.dev-base` is only the base layer used to build `Dockerfile.aces`.

`Dockerfile.train` is a separate training/runtime image and is not the one you
should point Runpod at for interactive development.

## Build And Push

Run these commands from the repository root, not from the `docker/` directory.

```bash
cd /path/to/aces
docker login

docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.dev-base \
  -t YOUR_REGISTRY/aces-dev-base:latest \
  --push .

docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.aces \
  --build-arg DEV_BASE_IMAGE=YOUR_REGISTRY/aces-dev-base:latest \
  -t YOUR_REGISTRY/aces-runpod:latest \
  --push .
```

Examples:

- Docker Hub:
  - `YOUR_REGISTRY=your-dockerhub-name`
- GHCR:
  - `YOUR_REGISTRY=ghcr.io/your-github-name`

The final image to use in Runpod is:

```text
YOUR_REGISTRY/aces-runpod:latest
```

## Runpod Pod Settings

Recommended Pod template values:

- Image:
  - `YOUR_REGISTRY/aces-runpod:latest`
- Volume mount path:
  - `/workspace`
- Ports:
  - `22/tcp`
  - `8888/http`

If you do not mount a persistent volume at `/workspace`, repo state and
bootstrap outputs will not survive Pod restarts.

## Runpod Environment Variables

In the Runpod UI, the environment-variable editor is under:

1. Create Pod
2. `Edit Template`
3. `Environment Variables`
4. `Add Environment Variable`

Set these values:

```text
ACES_REPO_URL=https://github.com/YOU/aces.git
ACES_REPO_REF=main
ACES_PROJECT_DIR=/workspace/aces
ACES_BOOTSTRAP_PROJECT=1
ACES_GIT_AUTO_PULL=1
POETRY_VIRTUALENVS_CREATE=true
POETRY_VIRTUALENVS_IN_PROJECT=true
```

Optional values:

```text
ACES_RUN_GPU_CHECK=1
ACES_FORCE_BOOTSTRAP=1
ACES_REPO_REMOTE=origin
ACES_DOTFILES_REPO=https://github.com/YOU/dotfiles.git
ACES_DOTFILES_REF=main
ACES_DOTFILES_DIR=/workspace/dotfiles
```

Notes:

- `ACES_BOOTSTRAP_PROJECT=1`
  - Runs `poetry install --with dev` and `poetry run maturin develop --release --features gpu`.
- `POETRY_VIRTUALENVS_CREATE=true` and `POETRY_VIRTUALENVS_IN_PROJECT=true`
  - Keeps Poetry inside `/workspace/aces/.venv` instead of modifying the Runpod base image's system Python.
- `ACES_GIT_AUTO_PULL=1`
  - Runs `git pull --ff-only` on startup when the worktree is clean.
  - If the Pod already has local changes, startup skips the pull on purpose.
- `ACES_REPO_REF=main`
  - Can be a branch name, tag, or commit.
  - Branch names are checked out as real local branches that track the remote.
- `ACES_FORCE_BOOTSTRAP=1`
  - Forces a rebuild even if the current commit was already bootstrapped before.

## What Happens On First Boot

When the Pod starts:

1. The startup script persists shell history, Claude state, git config, and SSH state under `/workspace/.dev-home`.
2. If `ACES_REPO_URL` is set and `/workspace/aces` does not exist yet, it clones the repo.
3. If `ACES_REPO_REF` is set, it checks out that branch or ref.
4. If `ACES_GIT_AUTO_PULL=1`, it runs a safe fast-forward pull when possible.
5. If `ACES_BOOTSTRAP_PROJECT=1`, it installs Python dependencies and builds the GPU extension.

The initial bootstrap can take a while. That is expected.

## After The Pod Starts

SSH in or open the web terminal, then run:

```bash
cd /workspace/aces
git status
git branch --show-current
env | grep '^ACES_'
```

If the repo cloned and the bootstrap completed, you can work normally:

```bash
cd /workspace/aces
git checkout -b your-branch
claude
```

For a quick GPU-path validation:

```bash
cd /workspace/aces
bash scripts/check_gpu_setup.sh
```

To start ACES training or curriculum runs, use the normal project commands from
inside `/workspace/aces`.

## Git / Push Requirements

`git clone` from a public repo works without extra setup, but `git push` does not.

To push from the Pod, make sure one of these is available inside the Pod:

- SSH auth:
  - mount or copy your SSH key and known hosts into `~/.ssh`
- HTTPS auth:
  - configure a credential helper or token in `~/.git-credentials`

The startup script persists `~/.ssh`, `~/.gitconfig`, and `~/.git-credentials`
into `/workspace/.dev-home`, so they survive Pod restarts when `/workspace` is persistent.

## Common Pitfalls

- Build from the repo root:
  - correct: `docker buildx build -f docker/Dockerfile.dev-base ... .`
  - wrong: building with `docker/` as the context
- Do not point Runpod at `aces-dev-base` unless you want a generic workstation image.
  - For ACES development, use `aces-runpod`.
- The first boot is slower than later boots.
  - Dependency install and `maturin develop` happen there.
- If startup fails while uninstalling a package such as `pyparsing`, Poetry is
  trying to modify system Python. Set `POETRY_VIRTUALENVS_CREATE=true` and
  `POETRY_VIRTUALENVS_IN_PROJECT=true`, then restart with
  `ACES_FORCE_BOOTSTRAP=1`.
- Updating Pod environment variables causes a Pod restart.
  - Only data under `/workspace` is preserved.
- `ACES_GIT_AUTO_PULL=1` is intentionally conservative.
  - If the worktree is dirty, startup leaves your local edits alone.

## About `.dockerignore`

Keep `.dockerignore` at the repository root for the current build commands.

Reason:

- The build context is the final `.` in commands like:
  - `docker buildx build -f docker/Dockerfile.dev-base ... .`
- Docker looks for `.dockerignore` in the root of the build context.

Do not simply move `.dockerignore` into `docker/` unless you also change the
build strategy.

If you ever want per-Dockerfile ignore rules, Docker supports Dockerfile-specific
ignore files with names like:

- `docker/Dockerfile.dev-base.dockerignore`
- `docker/Dockerfile.aces.dockerignore`
- `docker/Dockerfile.train.dockerignore`

Those can live next to the Dockerfiles and take precedence for that Dockerfile.
