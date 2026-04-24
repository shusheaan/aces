import numpy as np
import pytest

from aces.env.obs_layout import (
    OBS_DIM,
    OBS_LAYOUT,
    describe_obs,
    get_field,
    verify_layout,
)


def test_verify_layout_passes():
    verify_layout()  # no exception


def test_obs_dim_is_21():
    assert OBS_DIM == 21


def test_layout_total_matches_dim():
    assert sum(f.length for f in OBS_LAYOUT) == OBS_DIM


def test_layout_field_names_unique():
    names = [f.name for f in OBS_LAYOUT]
    assert len(names) == len(set(names))


def test_get_field_1d():
    obs = np.arange(OBS_DIM, dtype=np.float32)
    vel = get_field(obs, "own_velocity")
    assert vel.shape == (3,)
    np.testing.assert_array_equal(vel, [0, 1, 2])


def test_get_field_scalar():
    obs = np.arange(OBS_DIM, dtype=np.float32)
    sdf = get_field(obs, "nearest_sdf")
    assert isinstance(sdf, float)
    assert sdf == 15.0


def test_get_field_2d_batch():
    batch = np.arange(3 * OBS_DIM, dtype=np.float32).reshape(3, OBS_DIM)
    vel = get_field(batch, "own_velocity")
    assert vel.shape == (3, 3)
    np.testing.assert_array_equal(vel[0], [0, 1, 2])
    np.testing.assert_array_equal(vel[1], [21, 22, 23])


def test_get_field_unknown_name():
    obs = np.zeros(OBS_DIM)
    with pytest.raises(KeyError):
        get_field(obs, "nonexistent_field")


def test_get_field_wrong_length():
    with pytest.raises(ValueError):
        get_field(np.zeros(10), "own_velocity")


def test_describe_obs_1d():
    obs = np.arange(OBS_DIM, dtype=np.float32)
    text = describe_obs(obs)
    # Every field name should appear
    for f in OBS_LAYOUT:
        assert f.name in text
    # Header line
    assert f"Observation ({OBS_DIM}-dim)" in text


def test_describe_obs_2d_picks_env():
    batch = np.stack(
        [
            np.arange(OBS_DIM) * 0.0,
            np.arange(OBS_DIM) * 1.0,
        ]
    ).astype(np.float32)
    text0 = describe_obs(batch, env_index=0)
    text1 = describe_obs(batch, env_index=1)
    assert text0 != text1  # different content


def test_describe_obs_bad_env_index():
    batch = np.zeros((2, OBS_DIM), dtype=np.float32)
    with pytest.raises(IndexError):
        describe_obs(batch, env_index=5)


def test_describe_obs_wrong_shape():
    with pytest.raises(ValueError):
        describe_obs(np.zeros(10))
