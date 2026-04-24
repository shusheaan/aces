"""ACES -- Main entry point for training, evaluation, and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path


from aces._core import MppiController, Simulation
from aces.config import load_configs


def run_mppi_vs_mppi(args):
    """Run MPPI pursuer vs MPPI evader."""
    cfg = load_configs(args.config_dir)

    # Common physical parameters for Sim and MPPI
    common = dict(
        bounds=list(cfg.arena.bounds),
        obstacles=list(cfg.arena.obstacles),
        mass=cfg.drone.mass,
        arm_length=cfg.drone.arm_length,
        inertia=cfg.drone.inertia,
        max_thrust=cfg.drone.max_motor_thrust,
        torque_coeff=cfg.drone.torque_coefficient,
        drag_coeff=cfg.drone.drag_coefficient,
        dt_ctrl=cfg.drone.dt_ctrl,
        substeps=cfg.drone.substeps,
        drone_radius=cfg.arena.collision_radius,
    )

    camera_enabled = args.fpv

    sim = Simulation(
        **common,
        fov=cfg.rules.lockon.fov_radians,
        lock_distance=cfg.rules.lockon.lock_distance,
        lock_duration=cfg.rules.lockon.lock_duration,
        wind_theta=cfg.rules.noise.wind_theta,
        wind_mu=list(cfg.rules.noise.wind_mu),
        wind_sigma=cfg.rules.noise.wind_sigma,
        obs_noise_std=cfg.rules.noise.obs_noise_std,
        camera_enabled=camera_enabled,
        camera_width=cfg.rules.camera.width,
        camera_height=cfg.rules.camera.height,
        camera_fov_deg=cfg.rules.camera.fov_deg,
        camera_max_depth=cfg.rules.camera.max_depth,
        camera_render_hz=cfg.rules.camera.render_hz,
        camera_min_conf_dist=cfg.rules.detection.min_confidence_distance,
    )

    mppi_common = dict(
        **common,
        num_samples=args.mppi_samples,
        horizon=args.mppi_horizon,
        noise_std=cfg.rules.mppi.noise_std,
        temperature=cfg.rules.mppi.temperature,
        w_dist=cfg.rules.mppi.weights.w_dist,
        w_face=cfg.rules.mppi.weights.w_face,
        w_ctrl=cfg.rules.mppi.weights.w_ctrl,
        w_obs=cfg.rules.mppi.weights.w_obs,
        d_safe=cfg.rules.mppi.weights.d_safe,
        risk_wind_theta=cfg.rules.mppi.risk.wind_theta,
        risk_wind_sigma=cfg.rules.mppi.risk.wind_sigma,
        risk_cvar_alpha=cfg.rules.mppi.risk.cvar_alpha,
        risk_cvar_penalty=cfg.rules.mppi.risk.cvar_penalty,
    )

    pursuer = MppiController(**mppi_common)
    evader = MppiController(**mppi_common)

    vis = None
    if not args.no_vis:
        from aces.viz import AcesVisualizer

        vis = AcesVisualizer(config_dir=args.config_dir, recording_id="aces_mppi")

    spawn_a = list(cfg.arena.spawn_a)
    spawn_b = list(cfg.arena.spawn_b)
    sim.reset(spawn_a, spawn_b)
    pursuer.reset()
    evader.reset()

    mode_str = "FPV" if args.fpv else "omniscient"
    print(f"[ACES] MPPI vs MPPI ({mode_str}) -- {args.max_steps} steps")

    for step in range(args.max_steps):
        state_a = sim.drone_a_state()
        state_b = sim.drone_b_state()

        motors_a = pursuer.compute_action(list(state_a), list(state_b), pursuit=True)
        motors_b = evader.compute_action(list(state_b), list(state_a), pursuit=False)

        result = sim.step(list(motors_a), list(motors_b))

        if vis is not None:
            vis.log_step(step, result, sim=sim)
            if args.fpv:
                vis.log_camera(step, result, drone="a")
                vis.log_camera(step, result, drone="b")

        if result.drone_a_collision or result.drone_a_oob:
            print(f"[ACES] Drone A crashed at step {step}")
            break
        if result.drone_b_collision or result.drone_b_oob:
            print(f"[ACES] Drone B crashed at step {step}")
            break
        if result.kill_a:
            print(f"[ACES] Drone A locked on Drone B at step {step} -- KILL!")
            break
        if result.kill_b:
            print(f"[ACES] Drone B locked on Drone A at step {step} -- KILL!")
            break

        if step % 100 == 0:
            print(
                f"  step {step}: dist={result.distance:.3f}m  "
                f"lock_a={result.lock_a_progress:.2f}  lock_b={result.lock_b_progress:.2f}"
            )

    print("[ACES] Done.")


def _resolve_noise(args) -> tuple[float | None, float | None]:
    """Resolve noise overrides from CLI args."""
    if args.no_noise:
        return 0.0, 0.0
    wind = args.wind_sigma if args.wind_sigma is not None else None
    obs = args.obs_noise if args.obs_noise is not None else None
    return wind, obs


def run_train(args):
    """Train RL agent with self-play."""
    from aces.training import SelfPlayTrainer

    wind_sigma, obs_noise_std = _resolve_noise(args)
    timesteps = int(str(args.timesteps).split(",")[0])

    noise_desc = []
    if wind_sigma is not None:
        noise_desc.append(f"wind={wind_sigma}")
    if obs_noise_std is not None:
        noise_desc.append(f"obs_noise={obs_noise_std}")
    noise_str = ", ".join(noise_desc) if noise_desc else "from config"

    mode_str = "FPV" if args.fpv else "vector"

    trainer = SelfPlayTrainer(
        config_dir=args.config_dir,
        total_timesteps=timesteps,
        wind_sigma=wind_sigma,
        obs_noise_std=obs_noise_std,
        fpv=args.fpv,
        task=args.task,
    )

    if args.resume:
        print(f"[ACES] Resuming from {args.resume}...")
        trainer.load(args.resume)

    print(
        f"[ACES] Training for {timesteps} timesteps "
        f"(obs: {mode_str}, noise: {noise_str})..."
    )
    trainer.train()
    trainer.save(args.save_path)
    print(f"[ACES] Model saved to {args.save_path}")

    if trainer.stats:
        s = trainer.stats
        print(
            f"[ACES] Stats: {s['episodes']} episodes, "
            f"mean_reward={s['mean_reward']:.2f}, "
            f"kills={s['kills']}, deaths={s['deaths']}, "
            f"kill_rate={s['kill_rate']:.2%}"
        )


def run_evaluate(args):
    """Evaluate a trained model against an opponent."""
    from aces.training import evaluate

    wind_sigma, obs_noise_std = _resolve_noise(args)

    noise_desc = []
    if wind_sigma is not None:
        noise_desc.append(f"wind={wind_sigma}")
    if obs_noise_std is not None:
        noise_desc.append(f"obs_noise={obs_noise_std}")
    noise_str = ", ".join(noise_desc) if noise_desc else "from config"

    print(
        f"[ACES] Evaluating {args.model_path} vs {args.opponent} "
        f"({args.n_episodes} episodes, noise: {noise_str})..."
    )

    results = evaluate(
        model_path=args.model_path,
        config_dir=args.config_dir,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_steps,
        opponent=args.opponent,
        wind_sigma=wind_sigma,
        obs_noise_std=obs_noise_std,
        mppi_samples=args.mppi_samples,
        mppi_horizon=args.mppi_horizon,
        fpv=args.fpv,
    )

    print(f"\n[ACES] Evaluation Results ({args.n_episodes} episodes):")
    print(f"  Win rate:          {results['win_rate']:.1%} ({results['wins']} wins)")
    print(
        f"  Loss rate:         {results['loss_rate']:.1%} ({results['losses']} losses)"
    )
    print(
        f"  Crash rate:        {results['crash_rate']:.1%} ({results['crashes']} crashes)"
    )
    print(f"  Timeouts:          {results['timeouts']}")
    print(f"  Avg kill time:     {results['avg_kill_time']:.1f} steps")
    print(f"  Avg survival time: {results['avg_survival_time']:.1f} steps")
    print(f"  Mean reward:       {results['mean_reward']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="ACES: Air Combat Engagement Simulation"
    )
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).parent.parent / "configs"),
    )
    parser.add_argument(
        "--mode",
        choices=["mppi-vs-mppi", "train", "evaluate", "export", "curriculum"],
        default="mppi-vs-mppi",
    )
    parser.add_argument("--timesteps", default="500000")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--mppi-samples", type=int, default=1024)
    parser.add_argument("--mppi-horizon", type=int, default=50)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument(
        "--fpv",
        action="store_true",
        help="Enable first-person vision (depth camera + CNN policy)",
    )
    parser.add_argument(
        "--task",
        default="dogfight",
        choices=[
            "hover",
            "pursuit_linear",
            "pursuit_evasive",
            "search_pursuit",
            "dogfight",
        ],
        help="Curriculum task / difficulty stage",
    )
    parser.add_argument("--save-path", default="aces_model")

    # Noise control
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable all noise (wind + observation) regardless of config",
    )
    parser.add_argument(
        "--wind-sigma",
        type=float,
        default=None,
        help="Override wind sigma (0 to disable)",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=None,
        help="Override observation noise std (0 to disable)",
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from a saved model path (e.g., aces_model)",
    )

    # Curriculum GPU opt-in
    parser.add_argument(
        "--use-gpu-env",
        action="store_true",
        help=(
            "Use GPU-batched MPPI opponent env for --mode curriculum "
            "(requires aces-py-bridge built with --features gpu). "
            "Overrides phase task/opponent — runs PPO-vs-GPU-MPPI dogfight "
            "for every phase."
        ),
    )
    parser.add_argument(
        "--gpu-mppi-samples",
        type=int,
        default=128,
        help="MPPI sample count per drone for GPU env (default: 128)",
    )
    parser.add_argument(
        "--gpu-mppi-horizon",
        type=int,
        default=15,
        help="MPPI rollout horizon for GPU env (default: 15)",
    )
    parser.add_argument(
        "--gpu-noise-std",
        type=float,
        default=0.03,
        help="MPPI noise std for GPU env (default: 0.03)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel envs (applies to both CPU SubprocVecEnv and GPU VecEnv paths; default: 8)",
    )

    # Evaluate mode
    parser.add_argument("--model-path", default="aces_model")
    parser.add_argument(
        "--opponent",
        default="mppi",
        choices=["mppi", "random"],
        help="Opponent type for evaluation",
    )
    parser.add_argument("--n-episodes", type=int, default=100)

    args = parser.parse_args()

    if args.mode == "mppi-vs-mppi":
        run_mppi_vs_mppi(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "export":
        from aces.policy import export_mlp_policy

        export_mlp_policy(args.model_path, args.save_path.rstrip("/") + ".bin")
    elif args.mode == "curriculum":
        from aces.training import CurriculumTrainer

        wind_sigma, obs_noise_std = _resolve_noise(args)
        ts_parts = [int(x) for x in str(args.timesteps).split(",")]
        tasks = [
            "hover",
            "pursuit_linear",
            "pursuit_evasive",
            "search_pursuit",
            "dogfight",
        ]
        while len(ts_parts) < len(tasks):
            ts_parts.append(ts_parts[-1])
        stages = [{"task": t, "timesteps": s} for t, s in zip(tasks, ts_parts)]
        print(f"[ACES] Curriculum training: {len(stages)} stages")
        for s in stages:
            print(f"  {s['task']}: {s['timesteps']} steps")
        trainer = CurriculumTrainer(
            stages=stages,
            config_dir=args.config_dir,
            fpv=args.fpv,
            save_dir=args.save_path,
            wind_sigma=wind_sigma,
            obs_noise_std=obs_noise_std,
            n_envs=args.n_envs,
            use_gpu_env=args.use_gpu_env,
            gpu_mppi_samples=args.gpu_mppi_samples,
            gpu_mppi_horizon=args.gpu_mppi_horizon,
            gpu_noise_std=args.gpu_noise_std,
        )
        model = trainer.train()
        model.save(args.save_path + "_final")
        print(f"[ACES] Final model saved to {args.save_path}_final")


if __name__ == "__main__":
    main()
