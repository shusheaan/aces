"""Unified TOML config loading for ACES.

Loads drone.toml, arena.toml, and rules.toml into typed dataclasses.
Single source of truth for all configuration across the project.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DroneConfig:
    mass: float
    arm_length: float
    max_motor_thrust: float
    torque_coefficient: float
    drag_coefficient: float
    gravity: float
    ixx: float
    iyy: float
    izz: float
    dt_sim: float
    dt_ctrl: float
    substeps: int

    @property
    def inertia(self) -> list[float]:
        return [self.ixx, self.iyy, self.izz]


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArenaConfig:
    bounds: list[float]
    collision_radius: float
    spawn_a: list[float]
    spawn_b: list[float]
    obstacles: list[tuple[list[float], list[float]]]


# ---------------------------------------------------------------------------
# Rules sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockOnConfig:
    fov_degrees: float
    lock_distance: float
    lock_duration: float

    @property
    def fov_radians(self) -> float:
        return math.radians(self.fov_degrees)


@dataclass(frozen=True)
class MppiWeightsConfig:
    w_dist: float
    w_face: float
    w_ctrl: float
    w_obs: float
    d_safe: float


@dataclass(frozen=True)
class MppiRiskConfig:
    wind_theta: float
    wind_sigma: float
    cvar_alpha: float
    cvar_penalty: float


@dataclass(frozen=True)
class MppiConfig:
    num_samples: int
    horizon: int
    temperature: float
    noise_std: float
    weights: MppiWeightsConfig
    risk: MppiRiskConfig


@dataclass(frozen=True)
class NoiseConfig:
    wind_theta: float
    wind_mu: list[float]
    wind_sigma: float
    obs_noise_std: float


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool
    width: int
    height: int
    fov_deg: float
    max_depth: float
    render_hz: int
    policy_width: int
    policy_height: int


@dataclass(frozen=True)
class DetectionConfig:
    drone_radius: float
    min_confidence_distance: float


@dataclass(frozen=True)
class RewardConfig:
    kill_reward: float
    killed_penalty: float
    collision_penalty: float
    lock_progress_reward: float
    control_penalty: float
    approach_reward: float
    survival_bonus: float
    info_gain_reward: float
    lost_contact_penalty: float


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int
    learning_rate: float
    batch_size: int
    n_steps: int
    gamma: float
    gae_lambda: float
    clip_range: float
    n_epochs: int
    opponent_update_interval: int
    max_episode_steps: int


@dataclass(frozen=True)
class RulesConfig:
    lockon: LockOnConfig
    mppi: MppiConfig
    noise: NoiseConfig
    camera: CameraConfig
    detection: DetectionConfig
    reward: RewardConfig
    training: TrainingConfig


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcesConfig:
    drone: DroneConfig
    arena: ArenaConfig
    rules: RulesConfig


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root (directory containing 'configs/')."""
    return Path(__file__).resolve().parent.parent


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _parse_drone(data: dict) -> DroneConfig:
    phys = data["physical"]
    iner = data["inertia"]
    sim = data["simulation"]
    return DroneConfig(
        mass=phys["mass"],
        arm_length=phys["arm_length"],
        max_motor_thrust=phys["max_motor_thrust"],
        torque_coefficient=phys["torque_coefficient"],
        drag_coefficient=phys["drag_coefficient"],
        gravity=phys["gravity"],
        ixx=iner["ixx"],
        iyy=iner["iyy"],
        izz=iner["izz"],
        dt_sim=sim["dt_sim"],
        dt_ctrl=sim["dt_ctrl"],
        substeps=sim["substeps"],
    )


def _parse_arena(data: dict) -> ArenaConfig:
    b = data["bounds"]
    bounds = [b["x"], b["y"], b["z"]]
    obstacles: list[tuple[list[float], list[float]]] = [
        (obs["center"], obs["half_extents"]) for obs in data.get("obstacles", [])
    ]
    return ArenaConfig(
        bounds=bounds,
        collision_radius=data["drone"]["collision_radius"],
        spawn_a=data["spawn"]["drone_a"],
        spawn_b=data["spawn"]["drone_b"],
        obstacles=obstacles,
    )


def _parse_rules(data: dict) -> RulesConfig:
    lo = data["lockon"]
    lockon = LockOnConfig(
        fov_degrees=lo["fov_degrees"],
        lock_distance=lo["lock_distance"],
        lock_duration=lo["lock_duration"],
    )

    mppi_raw = data["mppi"]
    weights_raw = mppi_raw["weights"]
    weights = MppiWeightsConfig(
        w_dist=weights_raw["w_dist"],
        w_face=weights_raw["w_face"],
        w_ctrl=weights_raw["w_ctrl"],
        w_obs=weights_raw["w_obs"],
        d_safe=weights_raw["d_safe"],
    )
    risk_raw = mppi_raw["risk"]
    risk = MppiRiskConfig(
        wind_theta=risk_raw["wind_theta"],
        wind_sigma=risk_raw["wind_sigma"],
        cvar_alpha=risk_raw["cvar_alpha"],
        cvar_penalty=risk_raw["cvar_penalty"],
    )
    mppi = MppiConfig(
        num_samples=mppi_raw["num_samples"],
        horizon=mppi_raw["horizon"],
        temperature=mppi_raw["temperature"],
        noise_std=mppi_raw["noise_std"],
        weights=weights,
        risk=risk,
    )

    noise_raw = data["noise"]
    noise = NoiseConfig(
        wind_theta=noise_raw["wind_theta"],
        wind_mu=noise_raw["wind_mu"],
        wind_sigma=noise_raw["wind_sigma"],
        obs_noise_std=noise_raw["obs_noise_std"],
    )

    cam_raw = data["camera"]
    camera = CameraConfig(
        enabled=cam_raw["enabled"],
        width=cam_raw["width"],
        height=cam_raw["height"],
        fov_deg=cam_raw["fov_deg"],
        max_depth=cam_raw["max_depth"],
        render_hz=cam_raw["render_hz"],
        policy_width=cam_raw["policy_width"],
        policy_height=cam_raw["policy_height"],
    )

    det_raw = data["detection"]
    detection = DetectionConfig(
        drone_radius=det_raw["drone_radius"],
        min_confidence_distance=det_raw["min_confidence_distance"],
    )

    rew_raw = data["reward"]
    reward = RewardConfig(
        kill_reward=rew_raw["kill_reward"],
        killed_penalty=rew_raw["killed_penalty"],
        collision_penalty=rew_raw["collision_penalty"],
        lock_progress_reward=rew_raw["lock_progress_reward"],
        control_penalty=rew_raw["control_penalty"],
        approach_reward=rew_raw["approach_reward"],
        survival_bonus=rew_raw["survival_bonus"],
        info_gain_reward=rew_raw["info_gain_reward"],
        lost_contact_penalty=rew_raw["lost_contact_penalty"],
    )

    tr_raw = data["training"]
    training = TrainingConfig(
        total_timesteps=tr_raw["total_timesteps"],
        learning_rate=tr_raw["learning_rate"],
        batch_size=tr_raw["batch_size"],
        n_steps=tr_raw["n_steps"],
        gamma=tr_raw["gamma"],
        gae_lambda=tr_raw["gae_lambda"],
        clip_range=tr_raw["clip_range"],
        n_epochs=tr_raw["n_epochs"],
        opponent_update_interval=tr_raw["opponent_update_interval"],
        max_episode_steps=tr_raw["max_episode_steps"],
    )

    return RulesConfig(
        lockon=lockon,
        mppi=mppi,
        noise=noise,
        camera=camera,
        detection=detection,
        reward=reward,
        training=training,
    )


def load_configs(config_dir: str | Path | None = None) -> AcesConfig:
    """Load all TOML config files and return a typed AcesConfig.

    Parameters
    ----------
    config_dir:
        Path to the directory containing drone.toml, arena.toml, and
        rules.toml.  Defaults to ``<project_root>/configs/``.
    """
    if config_dir is None:
        config_dir = _project_root() / "configs"
    else:
        config_dir = Path(config_dir)

    drone_data = _load_toml(config_dir / "drone.toml")
    arena_data = _load_toml(config_dir / "arena.toml")
    rules_data = _load_toml(config_dir / "rules.toml")

    return AcesConfig(
        drone=_parse_drone(drone_data),
        arena=_parse_arena(arena_data),
        rules=_parse_rules(rules_data),
    )
