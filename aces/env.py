"""Gymnasium environment for ACES 1v1 quadrotor dogfight."""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np

from aces._core import MppiController, Simulation
from aces.config import load_configs
from aces.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DroneDogfightEnv(gym.Env):
    """1v1 quadrotor dogfight environment.

    Supports two observation modes:

    **Vector mode** (default, ``fpv=False``):
        Observation space (21-dim): velocity, angular velocity, relative
        opponent state, attitude, obstacle distance, lock progress,
        visibility, belief uncertainty, time since last seen.

    **FPV mode** (``fpv=True``):
        Dict observation with depth image + IMU vector:
        - "image": (1, 60, 80) normalized depth image
        - "vector": (12,) proprioceptive state (no opponent info)

    Action space (4-dim, continuous [-1, 1]):
        Thrust delta for each motor, mapped to:
        motors = hover + action * (max_thrust - hover), clipped to [0, max_thrust]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_dir: str | None = None,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        opponent: str = "random",
        mppi_samples: int | None = None,
        mppi_horizon: int | None = None,
        wind_sigma: float | None = None,
        obs_noise_std: float | None = None,
        fpv: bool = False,
        task: str = "dogfight",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._fpv = fpv

        # Load configs
        cfg = load_configs(config_dir)

        self._reward_cfg = dataclasses.asdict(cfg.rules.reward)

        self._task = task

        # Task-specific reward overrides
        _TASK_REWARD_OVERRIDES = {
            "pursuit_linear": {
                "approach_reward": 0.2,
                "info_gain_reward": 0.0,
                "lost_contact_penalty": 0.0,
            },
            "pursuit_evasive": {},
            "search_pursuit": {
                "info_gain_reward": 0.1,
                "lost_contact_penalty": 0.02,
            },
            "dogfight": {},
        }
        for key, val in _TASK_REWARD_OVERRIDES.get(task, {}).items():
            self._reward_cfg[key] = val

        # Trajectory state for pursuit_linear task
        self._traj_fn = None

        # Arena parameters
        self._bounds = list(cfg.arena.bounds)
        self._obstacles = list(cfg.arena.obstacles)
        self._spawn_a = list(cfg.arena.spawn_a)
        self._spawn_b = list(cfg.arena.spawn_b)
        self._drone_radius = cfg.arena.collision_radius

        # Physical parameters
        self._mass = cfg.drone.mass
        self._arm_length = cfg.drone.arm_length
        self._max_thrust = cfg.drone.max_motor_thrust
        self._torque_coeff = cfg.drone.torque_coefficient
        self._drag_coeff = cfg.drone.drag_coefficient
        self._inertia = cfg.drone.inertia

        # Simulation parameters
        self._dt_ctrl = cfg.drone.dt_ctrl
        self._substeps = cfg.drone.substeps

        # Camera parameters (Level 4 FPV)
        cam = cfg.rules.camera
        det = cfg.rules.detection
        self._camera_enabled = fpv or cam.enabled
        self._camera_width = cam.width
        self._camera_height = cam.height
        self._camera_fov_deg = cam.fov_deg
        self._camera_max_depth = cam.max_depth
        self._camera_render_hz = cam.render_hz
        self._policy_width = cam.policy_width
        self._policy_height = cam.policy_height
        self._camera_min_conf_dist = det.min_confidence_distance

        # Lock-on parameters
        self._fov = cfg.rules.lockon.fov_radians
        self._lock_distance = cfg.rules.lockon.lock_distance
        self._lock_duration = cfg.rules.lockon.lock_duration

        # Noise parameters (Level 2) -- constructor args override config
        noise = cfg.rules.noise
        self._wind_theta = noise.wind_theta
        self._wind_mu = list(noise.wind_mu)
        self._wind_sigma = noise.wind_sigma if wind_sigma is None else wind_sigma
        self._obs_noise_std = (
            noise.obs_noise_std if obs_noise_std is None else obs_noise_std
        )

        # Build Rust simulation
        self._sim = self._build_sim()

        # Hover / max thrust from sim
        self._hover_thrust = self._sim.hover_thrust()

        # Opponent configuration
        self._opponent_mode = opponent
        self._opponent_policy = None
        self._mppi: MppiController | None = None

        mppi_cfg = cfg.rules.mppi
        if opponent == "mppi":
            n_samples = (
                mppi_samples if mppi_samples is not None else mppi_cfg.num_samples
            )
            n_horizon = mppi_horizon if mppi_horizon is not None else mppi_cfg.horizon
            self._mppi = MppiController(
                bounds=self._bounds,
                obstacles=self._obstacles,
                num_samples=n_samples,
                horizon=n_horizon,
                noise_std=mppi_cfg.noise_std,
                temperature=mppi_cfg.temperature,
                mass=self._mass,
                arm_length=self._arm_length,
                inertia=self._inertia,
                max_thrust=self._max_thrust,
                torque_coeff=self._torque_coeff,
                drag_coeff=self._drag_coeff,
                dt_ctrl=self._dt_ctrl,
                substeps=self._substeps,
                drone_radius=self._drone_radius,
                w_dist=mppi_cfg.weights.w_dist,
                w_face=mppi_cfg.weights.w_face,
                w_ctrl=mppi_cfg.weights.w_ctrl,
                w_obs=mppi_cfg.weights.w_obs,
                d_safe=mppi_cfg.weights.d_safe,
                risk_wind_theta=mppi_cfg.risk.wind_theta,
                risk_wind_sigma=mppi_cfg.risk.wind_sigma,
                risk_cvar_alpha=mppi_cfg.risk.cvar_alpha,
                risk_cvar_penalty=mppi_cfg.risk.cvar_penalty,
            )

        # Curriculum tasks that need MPPI evasion
        if task in ("pursuit_evasive", "search_pursuit") and self._mppi is None:
            self._mppi = MppiController(
                bounds=self._bounds,
                obstacles=self._obstacles,
                num_samples=mppi_cfg.num_samples,
                horizon=mppi_cfg.horizon,
                noise_std=mppi_cfg.noise_std,
                temperature=mppi_cfg.temperature,
                mass=self._mass,
                arm_length=self._arm_length,
                inertia=self._inertia,
                max_thrust=self._max_thrust,
                torque_coeff=self._torque_coeff,
                drag_coeff=self._drag_coeff,
                dt_ctrl=self._dt_ctrl,
                substeps=self._substeps,
                drone_radius=self._drone_radius,
                w_dist=mppi_cfg.weights.w_dist,
                w_face=mppi_cfg.weights.w_face,
                w_ctrl=mppi_cfg.weights.w_ctrl,
                w_obs=mppi_cfg.weights.w_obs,
                d_safe=mppi_cfg.weights.d_safe,
            )

        # Spaces
        if self._fpv:
            self.observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(1, self._policy_height, self._policy_width),
                        dtype=np.float32,
                    ),
                    "vector": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
                    ),
                }
            )
        else:
            # Legacy 21-dim vector observation
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
            )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Episode state
        self._step_count = 0
        self._prev_lock_progress: float = 0.0
        self._prev_distance: float = 0.0
        self._prev_belief_var: float = 0.0
        # Cached depth frame for FPV (reused between camera renders)
        self._last_depth: np.ndarray | None = None

    def _build_sim(self) -> Simulation:
        """Create a fresh Simulation instance from config."""
        return Simulation(
            bounds=self._bounds,
            obstacles=self._obstacles,
            mass=self._mass,
            arm_length=self._arm_length,
            inertia=self._inertia,
            max_thrust=self._max_thrust,
            torque_coeff=self._torque_coeff,
            drag_coeff=self._drag_coeff,
            fov=self._fov,
            lock_distance=self._lock_distance,
            lock_duration=self._lock_duration,
            dt_ctrl=self._dt_ctrl,
            substeps=self._substeps,
            drone_radius=self._drone_radius,
            wind_theta=self._wind_theta,
            wind_mu=self._wind_mu,
            wind_sigma=self._wind_sigma,
            obs_noise_std=self._obs_noise_std,
            camera_enabled=self._camera_enabled,
            camera_width=self._camera_width,
            camera_height=self._camera_height,
            camera_fov_deg=self._camera_fov_deg,
            camera_max_depth=self._camera_max_depth,
            camera_render_hz=self._camera_render_hz,
            camera_min_conf_dist=self._camera_min_conf_dist,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_opponent_policy(self, policy: Any) -> None:
        """Set a trained SB3 policy as the opponent (for 'policy' mode)."""
        self._opponent_policy = policy
        self._opponent_mode = "policy"

    def set_opponent_weights(self, state_dict: dict) -> None:
        """Update opponent policy weights. Works across SubprocVecEnv processes."""
        if self._opponent_policy is not None:
            self._opponent_policy.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        own_state: tuple | list,
        opp_state: tuple | list,
        nearest_obs_dist: float,
        lock_progress: float,
        being_locked_progress: float,
        opponent_visible: float = 1.0,
        belief_uncertainty: float = 0.0,
        time_since_last_seen: float = 0.0,
        euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Build the 21-dim observation vector."""
        own = np.array(own_state, dtype=np.float64)
        opp = np.array(opp_state, dtype=np.float64)

        own_vel = own[3:6]
        own_angvel = own[10:13]
        rel_pos = opp[0:3] - own[0:3]
        rel_vel = opp[3:6] - own[3:6]
        roll, pitch, yaw = euler

        obs = np.array(
            [
                *own_vel,
                *own_angvel,
                *rel_pos,
                *rel_vel,
                roll,
                pitch,
                yaw,
                nearest_obs_dist,
                lock_progress,
                being_locked_progress,
                opponent_visible,
                belief_uncertainty,
                time_since_last_seen,
            ],
            dtype=np.float32,
        )
        return obs

    def _build_fpv_obs(
        self,
        own_state: tuple | list,
        nearest_obs_dist: float,
        lock_progress: float,
        being_locked_progress: float,
        depth_image: np.ndarray | None,
        euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        """Build FPV Dict observation (image + 12-dim vector)."""
        own = np.array(own_state, dtype=np.float64)

        own_vel = own[3:6]
        own_angvel = own[10:13]
        roll, pitch, yaw = euler

        vector = np.array(
            [
                *own_vel,
                *own_angvel,
                roll,
                pitch,
                yaw,
                lock_progress,
                being_locked_progress,
                nearest_obs_dist,
            ],
            dtype=np.float32,
        )

        # Depth image: downsample from full res to policy res and normalize
        if depth_image is not None:
            self._last_depth = depth_image
        if self._last_depth is not None:
            img_full = self._last_depth.reshape(self._camera_height, self._camera_width)
            # 4x4 average pooling via reshape + mean
            img_ds = img_full.reshape(
                self._policy_height,
                self._camera_height // self._policy_height,
                self._policy_width,
                self._camera_width // self._policy_width,
            ).mean(axis=(1, 3))
            # Normalize to [0, 1]
            img_ds = np.clip(img_ds / self._camera_max_depth, 0.0, 1.0).astype(
                np.float32
            )
            image = img_ds[np.newaxis, :, :]  # (1, H, W)
        else:
            # No frame yet (first step before first render) -- return max depth
            image = np.ones(
                (1, self._policy_height, self._policy_width), dtype=np.float32
            )

        return {"image": image, "vector": vector}

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _map_action(self, action: np.ndarray) -> list[float]:
        """Map [-1, 1] action to motor thrusts in [0, max_thrust]."""
        hover = self._hover_thrust
        motors = hover + action * (self._max_thrust - hover)
        motors = np.clip(motors, 0.0, self._max_thrust)
        return motors.tolist()

    # ------------------------------------------------------------------
    # Opponent action
    # ------------------------------------------------------------------

    def _opponent_action(self) -> list[float]:
        """Compute opponent motor thrusts based on current task/mode."""
        if self._task == "pursuit_linear":
            return self._trajectory_action()
        elif self._task in ("pursuit_evasive", "search_pursuit"):
            return self._mppi_evasion_action()
        # dogfight: use existing opponent logic
        elif self._opponent_mode == "random":
            raw = self.np_random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
            return self._map_action(raw)
        elif self._opponent_mode == "mppi" and self._mppi is not None:
            state_b = list(self._sim.drone_b_state())
            state_a = list(self._sim.drone_a_state())
            action = self._mppi.compute_action(state_b, state_a, True)
            return list(action)
        elif self._opponent_mode == "policy" and self._opponent_policy is not None:
            state_b = list(self._sim.drone_b_state())
            state_a = list(self._sim.drone_a_state())
            if self._fpv:
                opp_obs = self._build_fpv_obs(
                    state_b,
                    nearest_obs_dist=0.0,
                    lock_progress=0.0,
                    being_locked_progress=0.0,
                    depth_image=None,
                )
            else:
                opp_obs = self._build_obs(state_b, state_a, 0.0, 0.0, 0.0)
            raw, _ = self._opponent_policy.predict(opp_obs, deterministic=True)
            return self._map_action(np.array(raw, dtype=np.float32))
        else:
            return [self._hover_thrust] * 4

    def _mppi_evasion_action(self) -> list[float]:
        """MPPI opponent in evasion mode (pursuit=False)."""
        state_b = list(self._sim.drone_b_state())
        state_a = list(self._sim.drone_a_state())
        action = self._mppi.compute_action(state_b, state_a, False)
        return list(action)

    def _trajectory_action(self) -> list[float]:
        """PD controller tracking the current trajectory waypoint."""
        target = self._traj_fn(t=self._step_count * self._dt_ctrl)
        state_b = self._sim.drone_b_state()
        pos = np.array(state_b[:3], dtype=np.float64)
        vel = np.array(state_b[3:6], dtype=np.float64)

        error = target - pos
        kp, kd = 2.0, 1.0
        accel_cmd = kp * error - kd * vel
        hover = self._hover_thrust

        thrust = hover + (accel_cmd[2] * self._mass / 4.0)
        lateral_mag = float(np.linalg.norm(accel_cmd[:2])) * self._mass * 0.25
        motors = [
            max(0.0, min(self._max_thrust, thrust + lateral_mag)),
            max(0.0, min(self._max_thrust, thrust + lateral_mag)),
            max(0.0, min(self._max_thrust, thrust - lateral_mag)),
            max(0.0, min(self._max_thrust, thrust - lateral_mag)),
        ]
        return motors

    # ------------------------------------------------------------------
    # Spawn helpers
    # ------------------------------------------------------------------

    def _occluded_spawn(self) -> tuple[list[float], list[float]]:
        """Find spawn positions where drones can't see each other.

        Places drone A at its configured spawn + jitter, then tries
        positions on the far side of obstacles for drone B.
        """
        jitter_a = self.np_random.uniform(-0.3, 0.3, size=(3,))
        pos_a = [self._spawn_a[i] + jitter_a[i] for i in range(3)]

        obstacles = self._obstacles
        if not obstacles:
            pos_b = [
                self._spawn_b[i] + self.np_random.uniform(-0.3, 0.3) for i in range(3)
            ]
            return pos_a, pos_b

        for _ in range(50):
            obs_center, obs_half = obstacles[self.np_random.integers(len(obstacles))]
            dir_a_to_obs = np.array(obs_center[:2]) - np.array(pos_a[:2])
            dist = float(np.linalg.norm(dir_a_to_obs))
            if dist < 0.1:
                continue
            dir_norm = dir_a_to_obs / dist
            offset = max(obs_half[0], obs_half[1]) + self.np_random.uniform(0.5, 1.5)
            bx = obs_center[0] + dir_norm[0] * offset
            by = obs_center[1] + dir_norm[1] * offset
            bz = self.np_random.uniform(0.8, self._bounds[2] - 0.5)
            margin = 0.5
            bx = float(np.clip(bx, margin, self._bounds[0] - margin))
            by = float(np.clip(by, margin, self._bounds[1] - margin))
            pos_b = [bx, by, bz]
            if not self._sim.check_los(pos_a, pos_b):
                return pos_a, pos_b

        # Fallback: maximum distance corners
        pos_b = [
            self._bounds[0] - pos_a[0],
            self._bounds[1] - pos_a[1],
            self.np_random.uniform(0.8, self._bounds[2] - 0.5),
        ]
        return pos_a, pos_b

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_count = 0

        # Set up trajectory for pursuit_linear
        if self._task == "pursuit_linear":
            traj_type, traj_kwargs = Trajectory.random_trajectory(
                self._bounds, self.np_random
            )
            self._traj_fn = lambda t, _type=traj_type, _kw=traj_kwargs: getattr(
                Trajectory, _type
            )(t=t, **_kw)

        # Spawn positions
        if self._task == "search_pursuit":
            pos_a, pos_b = self._occluded_spawn()
        else:
            jitter_a = self.np_random.uniform(-0.5, 0.5, size=(3,))
            jitter_b = self.np_random.uniform(-0.5, 0.5, size=(3,))
            pos_a = [self._spawn_a[i] + jitter_a[i] for i in range(3)]
            pos_b = [self._spawn_b[i] + jitter_b[i] for i in range(3)]

        state_a, state_b = self._sim.reset(pos_a, pos_b)

        # Reset MPPI warm-start if applicable
        if self._mppi is not None:
            self._mppi.reset()

        # Initialize tracking for reward shaping
        self._prev_lock_progress = 0.0
        self._prev_distance = float(
            np.linalg.norm(np.array(state_a[:3]) - np.array(state_b[:3]))
        )
        self._prev_belief_var = 0.0
        self._last_depth = None

        # At reset drones start at hover with identity rotation
        initial_euler = (0.0, 0.0, 0.0)

        if self._fpv:
            obs = self._build_fpv_obs(
                state_a,
                nearest_obs_dist=10.0,
                lock_progress=0.0,
                being_locked_progress=0.0,
                depth_image=None,
                euler=initial_euler,
            )
        else:
            initial_visible = 1.0 if self._sim.check_los(pos_a, pos_b) else 0.0
            obs = self._build_obs(
                state_a,
                state_b,
                nearest_obs_dist=10.0,
                lock_progress=0.0,
                being_locked_progress=0.0,
                opponent_visible=initial_visible,
                euler=initial_euler,
            )

        info: dict[str, Any] = {
            "agent_pos": np.array(state_a[:3], dtype=np.float32),
            "opponent_pos": np.array(state_b[:3], dtype=np.float32),
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Map agent action to motor thrusts
        motors_a = self._map_action(action)

        # Compute opponent action
        motors_b = self._opponent_action()

        # Step simulation
        result = self._sim.step(motors_a, motors_b)

        # Extract states
        state_a = list(result.drone_a_state)
        state_b = list(result.drone_b_state)

        # Get euler angles from Rust (avoids Python quaternion conversion)
        euler_a = tuple(result.drone_a_euler)

        # Build observation
        if self._fpv:
            # FPV mode: depth image + IMU vector
            depth_img = None
            if result.depth_image_a is not None:
                depth_img = np.array(result.depth_image_a, dtype=np.float32)
            obs = self._build_fpv_obs(
                state_a,
                nearest_obs_dist=result.nearest_obs_dist_a,
                lock_progress=result.lock_a_progress,
                being_locked_progress=result.lock_b_progress,
                depth_image=depth_img,
                euler=euler_a,
            )
        else:
            # Vector mode: belief state
            visible = result.a_sees_b
            if visible:
                if self._obs_noise_std > 0.0:
                    ekf_pos = list(result.ekf_b_pos_from_a)
                    ekf_vel = list(result.ekf_b_vel_from_a)
                    opp_for_obs = ekf_pos + ekf_vel + list(state_b[6:])
                else:
                    opp_for_obs = state_b
            else:
                belief_pos = list(result.belief_b_pos_from_a)
                opp_for_obs = belief_pos + [0.0, 0.0, 0.0] + list(state_b[6:])

            obs = self._build_obs(
                state_a,
                opp_for_obs,
                nearest_obs_dist=result.nearest_obs_dist_a,
                lock_progress=result.lock_a_progress,
                being_locked_progress=result.lock_b_progress,
                opponent_visible=1.0 if visible else 0.0,
                belief_uncertainty=result.belief_b_var_from_a,
                time_since_last_seen=result.time_since_a_saw_b,
                euler=euler_a,
            )

        # ---- Reward computation ----
        rcfg = self._reward_cfg
        reward = 0.0
        terminated = False

        # Terminal conditions (agent collision / OOB)
        if result.drone_a_collision or result.drone_a_oob:
            reward = float(rcfg["collision_penalty"])
            terminated = True
        # Opponent kills agent
        elif result.kill_b:
            reward = float(rcfg["killed_penalty"])
            terminated = True
        # Agent kills opponent
        elif result.kill_a:
            reward = float(rcfg["kill_reward"])
            terminated = True
        # Opponent crashes
        elif result.drone_b_collision or result.drone_b_oob:
            reward = float(rcfg["kill_reward"]) * 0.5
            terminated = True
        else:
            # Shaping rewards
            reward += float(rcfg["survival_bonus"])

            # Lock progress delta
            delta_lock = result.lock_a_progress - self._prev_lock_progress
            reward += float(rcfg["lock_progress_reward"]) * delta_lock

            # Approach reward (distance decrease is positive reward)
            delta_dist = self._prev_distance - result.distance
            reward += float(rcfg["approach_reward"]) * delta_dist

            # Control penalty
            motors_arr = np.array(motors_a, dtype=np.float64)
            hover_arr = np.full(4, self._hover_thrust, dtype=np.float64)
            ctrl_cost = float(np.sum((motors_arr - hover_arr) ** 2))
            reward -= float(rcfg["control_penalty"]) * ctrl_cost

            # Information-theoretic rewards (Level 3)
            info_gain_w = float(rcfg.get("info_gain_reward", 0.0))
            lost_contact_w = float(rcfg.get("lost_contact_penalty", 0.0))

            if info_gain_w > 0.0:
                # Reward for reducing belief uncertainty
                delta_var = self._prev_belief_var - result.belief_b_var_from_a
                if delta_var > 0:
                    reward += info_gain_w * delta_var

            if lost_contact_w > 0.0:
                # Penalty for prolonged loss of visual contact
                reward -= lost_contact_w * result.time_since_a_saw_b

        # Update tracking state
        self._prev_lock_progress = result.lock_a_progress
        self._prev_distance = result.distance
        self._prev_belief_var = result.belief_b_var_from_a

        # Truncation
        truncated = self._step_count >= self.max_episode_steps

        # Info dict
        info: dict[str, Any] = {
            "agent_pos": np.array(state_a[:3], dtype=np.float32),
            "opponent_pos": np.array(state_b[:3], dtype=np.float32),
            "distance": result.distance,
            "lock_a_progress": result.lock_a_progress,
            "lock_b_progress": result.lock_b_progress,
            "kill_a": result.kill_a,
            "kill_b": result.kill_b,
            "ekf_opponent_pos": np.array(result.ekf_b_pos_from_a, dtype=np.float32),
            "wind_force": np.array(result.wind_force_a, dtype=np.float32),
            "opponent_visible": result.a_sees_b,
            "belief_pos": np.array(result.belief_b_pos_from_a, dtype=np.float32),
            "belief_var": result.belief_b_var_from_a,
            "time_since_last_seen": result.time_since_a_saw_b,
            "camera_rendered": result.camera_rendered_a,
            "detection": {
                "detected": result.det_a_detected,
                "bbox": list(result.det_a_bbox),
                "confidence": result.det_a_confidence,
                "depth": result.det_a_depth,
                "pixel_center": list(result.det_a_pixel_center),
            },
        }

        return obs, float(reward), bool(terminated), bool(truncated), info
