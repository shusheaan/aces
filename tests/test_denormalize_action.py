"""Unit tests for the action denormalization math.

These tests do NOT require aces._core, GPU, or maturin — they only exercise
the pure NumPy function and are safe to run in any Python environment.

The mapping must match the CPU env (`DroneDogfightEnv._map_action`):

    motor = hover + action * (max_thrust - hover), clamped to [0, max_thrust]
"""

import numpy as np

from aces.training.gpu_vec_env import (
    HOVER_THRUST_PER_MOTOR,
    MAX_THRUST_PER_MOTOR,
    denormalize_action,
)


def test_minus_one_clamps_to_zero():
    # action = -1 -> hover - (max - hover) = 2*hover - max.
    # For Crazyflie defaults this is negative, so clips to 0.
    actions = np.array([[-1.0, -1.0, -1.0, -1.0]], dtype=np.float32)
    out = denormalize_action(actions)
    assert out.shape == (1, 4)
    assert out.dtype == np.float32
    # 2*0.0662 - 0.15 = -0.0176 -> clamped to 0.0
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_plus_one_maps_to_max_thrust():
    actions = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    out = denormalize_action(actions)
    np.testing.assert_allclose(out, MAX_THRUST_PER_MOTOR, atol=1e-6)


def test_zero_maps_to_hover_thrust():
    actions = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out = denormalize_action(actions)
    np.testing.assert_allclose(out, HOVER_THRUST_PER_MOTOR, atol=1e-6)


def test_above_plus_one_clamps():
    actions = np.array([[2.0, 3.0, 10.0, 1.5]], dtype=np.float32)
    out = denormalize_action(actions)
    np.testing.assert_allclose(out, MAX_THRUST_PER_MOTOR, atol=1e-6)
    assert np.all(out <= MAX_THRUST_PER_MOTOR + 1e-9)


def test_below_minus_one_clamps():
    actions = np.array([[-2.0, -5.0, -1.5, -10.0]], dtype=np.float32)
    out = denormalize_action(actions)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)
    assert np.all(out >= -1e-9)


def test_preserves_shape():
    for shape in [(4,), (1, 4), (16, 4), (4, 8, 4), (2, 3, 4)]:
        actions = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        out = denormalize_action(actions)
        assert out.shape == shape
        assert out.dtype == np.float32


def test_per_motor_independence():
    # Each motor slot is denormalized independently using the hover-centered map.
    actions = np.array([[-1.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    out = denormalize_action(actions)
    # action = -1 -> clipped to 0; 0 -> hover; 0.5 -> hover + 0.5*(max-hover);
    # 1 -> max.
    expected = np.array(
        [
            [
                0.0,
                HOVER_THRUST_PER_MOTOR,
                HOVER_THRUST_PER_MOTOR
                + 0.5 * (MAX_THRUST_PER_MOTOR - HOVER_THRUST_PER_MOTOR),
                MAX_THRUST_PER_MOTOR,
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_float64_input_converted_to_float32():
    actions = np.array([[-0.5, 0.5, -0.5, 0.5]], dtype=np.float64)
    out = denormalize_action(actions)
    assert out.dtype == np.float32


def test_linearity_mid_range():
    # Away from clamp saturation, the map is strictly affine:
    # motor = hover + x * (max - hover).
    # Use a range that stays above the lower clamp (i.e. action >= a_crit where
    # hover + a_crit*(max-hover) = 0 -> a_crit = -hover/(max-hover)).
    a_crit = -HOVER_THRUST_PER_MOTOR / (MAX_THRUST_PER_MOTOR - HOVER_THRUST_PER_MOTOR)
    lo = a_crit + 1e-3  # strictly inside the unclamped region
    actions = np.linspace(lo, 1.0, 21, dtype=np.float32).reshape(-1, 1)
    out = denormalize_action(actions)
    expected = HOVER_THRUST_PER_MOTOR + actions * (
        MAX_THRUST_PER_MOTOR - HOVER_THRUST_PER_MOTOR
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)
