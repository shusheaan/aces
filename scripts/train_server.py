#!/usr/bin/env python3
"""Headless server training script for ACES.

Usage:
    # Full curriculum from scratch (8 parallel envs)
    python scripts/train_server.py --n-envs 8

    # Resume from checkpoint
    python scripts/train_server.py --resume checkpoints/pursuit_linear_step_100000/

    # Custom curriculum config
    python scripts/train_server.py --curriculum configs/curriculum.toml --n-envs 16

    # Single-stage training (backward compat)
    python scripts/train_server.py --task dogfight --timesteps 500000
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ACES Headless Training")
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).parent.parent / "configs"),
    )
    parser.add_argument(
        "--curriculum",
        default=None,
        help="Path to curriculum.toml (default: <config-dir>/curriculum.toml)",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint dir to resume from"
    )
    parser.add_argument("--fpv", action="store_true")
    parser.add_argument("--save-path", default="aces_model_final")
    parser.add_argument(
        "--task",
        default=None,
        choices=["pursuit_linear", "pursuit_evasive", "search_pursuit", "dogfight"],
        help="Single task (skips curriculum, uses SelfPlayTrainer)",
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, etc.",
    )

    args = parser.parse_args()

    # Single-task mode (backward compat)
    if args.task:
        from aces.trainer import SelfPlayTrainer

        print(
            f"[ACES] Single-task training: {args.task} for {args.timesteps} steps "
            f"({args.n_envs} envs)"
        )
        trainer = SelfPlayTrainer(
            config_dir=args.config_dir,
            total_timesteps=args.timesteps,
            fpv=args.fpv,
            task=args.task,
            device=args.device,
        )
        trainer.train()
        trainer.save(args.save_path)
        print(f"[ACES] Model saved to {args.save_path}")
        return

    # Curriculum mode
    from aces.curriculum import load_curriculum
    from aces.trainer import CurriculumTrainer

    curriculum_path = args.curriculum or str(Path(args.config_dir) / "curriculum.toml")
    phases = load_curriculum(curriculum_path)
    print(f"[ACES] Loaded {len(phases)} curriculum phases from {curriculum_path}")
    for i, p in enumerate(phases):
        print(f"  Phase {i}: {p.name} ({p.task}, {p.max_timesteps} steps)")

    trainer = CurriculumTrainer(
        phases=phases,
        config_dir=args.config_dir,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        checkpoint_interval=args.checkpoint_interval,
        fpv=args.fpv,
        device=args.device,
    )

    if args.resume:
        print(f"[ACES] Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Graceful shutdown handler
    shutdown_requested = False

    def shutdown_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n[ACES] Forced exit.")
            sys.exit(1)
        shutdown_requested = True
        print("\n[ACES] Shutdown requested, saving checkpoint after current step...")
        trainer.save_checkpoint(str(Path(args.save_dir) / "interrupted"))
        print(f"[ACES] Checkpoint saved to {args.save_dir}/interrupted/")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"[ACES] Starting curriculum training with {args.n_envs} parallel envs")
    print(f"[ACES] Checkpoints: {args.save_dir}/")
    print(f"[ACES] TensorBoard: tensorboard --logdir {args.save_dir}/")

    model = trainer.train()

    model.save(args.save_path)
    print(f"[ACES] Final model saved to {args.save_path}")


if __name__ == "__main__":
    main()
