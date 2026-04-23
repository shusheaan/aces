"""Tests for aces.config — unified TOML config loading."""

import math

import pytest

from aces.config import AcesConfig, load_configs


@pytest.fixture(scope="module")
def cfg() -> AcesConfig:
    """Load configs once for all tests in this module."""
    return load_configs()


def test_load_configs_returns_dataclass(cfg: AcesConfig) -> None:
    assert isinstance(cfg, AcesConfig)


def test_drone_config_fields(cfg: AcesConfig) -> None:
    d = cfg.drone
    assert d.mass == 0.027
    assert d.max_motor_thrust == 0.15
    assert d.dt_ctrl == 0.01
    assert d.substeps == 10


def test_arena_config_fields(cfg: AcesConfig) -> None:
    a = cfg.arena
    assert a.bounds == [10, 10, 3]
    assert len(a.obstacles) == 5
    assert a.collision_radius == 0.05


def test_rules_config_fields(cfg: AcesConfig) -> None:
    r = cfg.rules
    assert r.lockon.fov_degrees == 90.0
    assert r.lockon.lock_distance == 2.0
    assert r.reward.kill_reward == 100.0
    assert r.mppi.num_samples == 1024


def test_training_config_fields(cfg: AcesConfig) -> None:
    t = cfg.rules.training
    assert t.total_timesteps == 500_000
    assert t.learning_rate == 3e-4
    assert t.n_steps == 2048


def test_load_configs_default_dir() -> None:
    """Calling load_configs() with no argument should succeed."""
    cfg = load_configs()
    assert isinstance(cfg, AcesConfig)
    assert cfg.drone.mass > 0


def test_obstacles_parsed_as_tuples(cfg: AcesConfig) -> None:
    for center, half_extents in cfg.arena.obstacles:
        assert len(center) == 3
        assert len(half_extents) == 3


def test_fov_radians_property(cfg: AcesConfig) -> None:
    expected = math.radians(90.0)
    assert cfg.rules.lockon.fov_radians == pytest.approx(expected)


def test_drone_inertia_property(cfg: AcesConfig) -> None:
    d = cfg.drone
    assert d.inertia == (d.ixx, d.iyy, d.izz)
    assert len(d.inertia) == 3
