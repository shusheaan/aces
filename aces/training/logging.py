"""Centralized logging configuration for ACES training and evaluation."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path


_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_dir: Path | str | None = None,
) -> logging.Logger:
    """Configure root ACES logger with console + optional file output.

    Args:
        level: Logging level (default INFO).
        log_dir: If provided, also write logs to ``<log_dir>/training.log``.

    Returns:
        The ``aces`` logger instance.
    """
    logger = logging.getLogger("aces")
    logger.setLevel(level)

    # Add console handler only once
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        logger.addHandler(console)

    # Add file handler if requested and not already present for this dir
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_file = log_dir / "training.log"
        already_has = any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename) == log_file.resolve()
            for h in logger.handlers
        )
        if not already_has:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            logger.addHandler(fh)

    return logger


def create_run_dir(prefix: str = "train", base: str = "logs") -> Path:
    """Create a timestamped run directory under ``base/``.

    Returns:
        Path to the new run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config_snapshot(run_dir: Path, config_dir: str | None = None) -> None:
    """Copy all TOML configs into the run directory for reproducibility."""
    if config_dir is None:
        config_dir = str(Path(__file__).parent.parent.parent / "configs")
    src = Path(config_dir)
    dst = run_dir / "config_snapshot"
    if src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.toml"):
            shutil.copy2(f, dst / f.name)


def save_run_metadata(
    run_dir: Path,
    *,
    task: str = "",
    timesteps: int = 0,
    fpv: bool = False,
    wind_sigma: float | None = None,
    obs_noise_std: float | None = None,
    extra: dict | None = None,
) -> None:
    """Write a JSON metadata file summarizing the run configuration."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "timesteps": timesteps,
        "fpv": fpv,
        "wind_sigma": wind_sigma,
        "obs_noise_std": obs_noise_std,
    }
    if extra:
        meta.update(extra)
    with open(run_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
