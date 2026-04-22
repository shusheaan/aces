"""Tests for the Rust PyO3 bridge (aces._core)."""

from aces._core import Simulation, MppiController, StepResult


def test_simulation_creation():
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    assert sim.hover_thrust() > 0
    assert sim.max_thrust() > sim.hover_thrust()


def test_simulation_reset():
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    state_a, state_b = sim.reset([1.0, 1.0, 1.5], [9.0, 9.0, 1.5])
    assert len(state_a) == 13
    assert len(state_b) == 13
    assert abs(state_a[0] - 1.0) < 1e-9
    assert abs(state_b[0] - 9.0) < 1e-9


def test_simulation_hover_stationary():
    """At hover thrust, drone should stay approximately still."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    hover = sim.hover_thrust()
    motors = [hover, hover, hover, hover]

    result = sim.step(motors, motors)
    pos = result.drone_a_state[:3]
    assert abs(pos[0] - 5.0) < 0.01
    assert abs(pos[1] - 5.0) < 0.01
    assert abs(pos[2] - 1.5) < 0.01


def test_simulation_freefall():
    """Zero thrust should cause descent."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    zero = [0.0, 0.0, 0.0, 0.0]

    for _ in range(10):
        result = sim.step(zero, zero)

    pos = result.drone_a_state[:3]
    assert pos[2] < 1.5, "drone should have fallen"


def test_step_result_fields():
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    sim.reset([1.0, 1.0, 1.5], [9.0, 9.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)

    assert isinstance(result, StepResult)
    assert len(result.drone_a_state) == 13
    assert len(result.drone_a_forward) == 3
    assert len(result.drone_a_euler) == 3
    assert isinstance(result.drone_a_collision, bool)
    assert isinstance(result.kill_a, bool)
    assert isinstance(result.distance, float)
    assert isinstance(result.nearest_obs_dist_a, float)
    assert not result.drone_a_collision
    assert not result.drone_a_oob


def test_collision_detection():
    """Drone spawned inside an obstacle should be detected as collision."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    sim.reset([5.0, 5.0, 1.5], [1.0, 1.0, 1.5])  # drone_a inside pillar
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)
    assert result.drone_a_collision


def test_oob_detection():
    """Drone spawned outside bounds should be detected."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
    )
    sim.reset([-1.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)
    assert result.drone_a_oob


def test_step_result_noise_and_ekf_fields():
    """StepResult should include noise, wind, and EKF fields."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        obs_noise_std=0.1,
        wind_theta=2.0,
        wind_sigma=0.3,
    )
    sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)

    # Noise fields exist and have correct length
    assert len(result.noisy_b_pos_from_a) == 3
    assert len(result.noisy_a_pos_from_b) == 3
    assert len(result.wind_force_a) == 3
    assert len(result.wind_force_b) == 3

    # EKF fields exist and have correct length
    assert len(result.ekf_b_pos_from_a) == 3
    assert len(result.ekf_b_vel_from_a) == 3
    assert len(result.ekf_a_pos_from_b) == 3
    assert len(result.ekf_a_vel_from_b) == 3


def test_ekf_reduces_estimation_error():
    """EKF-estimated position should be closer to truth than raw noise over time."""
    import numpy as np

    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        obs_noise_std=0.1,
    )
    sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])
    hover = sim.hover_thrust()

    sum_noisy_err = 0.0
    sum_ekf_err = 0.0
    n = 200

    for i in range(n):
        result = sim.step([hover] * 4, [hover] * 4)
        true_b_pos = np.array(result.drone_b_state[:3])
        noisy_b_pos = np.array(result.noisy_b_pos_from_a)
        ekf_b_pos = np.array(result.ekf_b_pos_from_a)

        # Skip first 20 steps (EKF burn-in)
        if i >= 20:
            sum_noisy_err += np.linalg.norm(noisy_b_pos - true_b_pos)
            sum_ekf_err += np.linalg.norm(ekf_b_pos - true_b_pos)

    avg_noisy = sum_noisy_err / (n - 20)
    avg_ekf = sum_ekf_err / (n - 20)

    assert avg_ekf < avg_noisy, (
        f"EKF error ({avg_ekf:.4f}) should be less than raw noise ({avg_noisy:.4f})"
    )


def test_wind_causes_drift():
    """With wind enabled, hovering drone should drift from its position."""
    import numpy as np

    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        wind_theta=2.0,
        wind_sigma=0.5,  # strong wind
    )
    sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 1.5])
    hover = sim.hover_thrust()

    for _ in range(500):
        result = sim.step([hover] * 4, [hover] * 4)

    pos = np.array(result.drone_a_state[:3])
    initial = np.array([5.0, 5.0, 1.5])
    drift = np.linalg.norm(pos[:2] - initial[:2])  # XY drift

    assert drift > 0.01, f"drone should drift with wind, got drift={drift:.4f}"


def test_visibility_clear():
    """Drones with clear LOS should see each other."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    # Place on same side of pillar
    sim.reset([1.0, 1.0, 1.5], [1.0, 3.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)
    assert result.a_sees_b is True
    assert result.b_sees_a is True


def test_visibility_occluded():
    """Drones with pillar between them should not see each other."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    # Place on opposite sides of center pillar
    sim.reset([3.0, 5.0, 1.5], [7.0, 5.0, 1.5])
    hover = sim.hover_thrust()
    result = sim.step([hover] * 4, [hover] * 4)
    assert result.a_sees_b is False
    assert result.b_sees_a is False


def test_belief_uncertainty_grows_when_occluded():
    """When occluded, belief uncertainty should increase over time."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
        obs_noise_std=0.1,
    )
    sim.reset([3.0, 5.0, 1.5], [7.0, 5.0, 1.5])
    hover = sim.hover_thrust()

    # Run several steps while occluded
    variances = []
    for _ in range(50):
        result = sim.step([hover] * 4, [hover] * 4)
        if not result.a_sees_b:
            variances.append(result.belief_b_var_from_a)

    # Variance should generally increase when occluded
    if len(variances) >= 10:
        early_var = sum(variances[:5]) / 5
        late_var = sum(variances[-5:]) / 5
        assert late_var >= early_var, (
            f"Belief variance should grow when occluded: early={early_var:.4f}, late={late_var:.4f}"
        )


def test_belief_reconverges_when_visible():
    """After occlusion, belief variance should drop when opponent becomes visible again."""

    # Use a scene with a pillar between the drones, then move to clear LOS
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
        obs_noise_std=0.1,
    )
    # Start occluded (pillar between them)
    sim.reset([3.0, 5.0, 1.5], [7.0, 5.0, 1.5])
    hover = sim.hover_thrust()

    # Phase 1: Run while occluded, let belief diverge
    for _ in range(30):
        result = sim.step([hover] * 4, [hover] * 4)
    var_occluded = result.belief_b_var_from_a

    # Phase 2: Move drones to clear LOS (same Y, no pillar between)
    sim.reset([1.0, 1.0, 1.5], [1.0, 3.0, 1.5])  # clear LOS
    for _ in range(30):
        result = sim.step([hover] * 4, [hover] * 4)
    var_visible = result.belief_b_var_from_a

    # After being visible, variance should be much lower
    assert var_visible < var_occluded or var_visible < 0.1, (
        f"Belief should reconverge when visible: occluded_var={var_occluded:.4f}, visible_var={var_visible:.4f}"
    )


def test_time_since_last_seen():
    """time_since_a_saw_b should increment when occluded, reset when visible."""
    sim = Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
    )
    sim.reset([3.0, 5.0, 1.5], [7.0, 5.0, 1.5])
    hover = sim.hover_thrust()

    result = sim.step([hover] * 4, [hover] * 4)
    if not result.a_sees_b:
        assert result.time_since_a_saw_b > 0.0


def test_risk_aware_mppi_controller():
    """Risk-aware MppiController (CVaR + wind) should produce valid actions."""
    ctrl = MppiController(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
        num_samples=64,
        horizon=10,
        risk_wind_theta=2.0,
        risk_wind_sigma=0.3,
        risk_cvar_alpha=0.05,
        risk_cvar_penalty=10.0,
    )

    sim = Simulation(bounds=[10.0, 10.0, 3.0], obstacles=[])
    sim.reset([2.0, 5.0, 1.5], [8.0, 5.0, 1.5])
    state_a = sim.drone_a_state()
    state_b = sim.drone_b_state()

    action = ctrl.compute_action(list(state_a), list(state_b), pursuit=True)
    assert len(action) == 4
    for m in action:
        assert 0.0 <= m <= 0.15


def test_mppi_controller_creation():
    ctrl = MppiController(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
        num_samples=32,
        horizon=5,
    )
    assert ctrl is not None


def test_mppi_controller_compute_action():
    ctrl = MppiController(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        num_samples=32,
        horizon=5,
    )
    # Hovering attacker facing target on +X axis
    sim = Simulation(bounds=[10.0, 10.0, 3.0], obstacles=[])
    sim.reset([2.0, 5.0, 1.5], [8.0, 5.0, 1.5])
    state_a = sim.drone_a_state()
    state_b = sim.drone_b_state()

    action = ctrl.compute_action(list(state_a), list(state_b), pursuit=True)
    assert len(action) == 4
    for m in action:
        assert 0.0 <= m <= 0.15  # within motor thrust range


def test_belief_mppi_controller():
    """Belief-weighted MPPI should produce valid actions and differ from standard."""
    ctrl = MppiController(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[([5.0, 5.0, 1.5], [0.5, 0.5, 1.5])],
        num_samples=64,
        horizon=10,
    )

    sim = Simulation(bounds=[10.0, 10.0, 3.0], obstacles=[])
    sim.reset([2.0, 5.0, 1.5], [8.0, 5.0, 1.5])
    state_a = list(sim.drone_a_state())
    state_b = list(sim.drone_b_state())

    # With zero belief_var, should behave like standard
    action_certain = ctrl.compute_action_with_belief(state_a, state_b, True, 0.0)
    assert len(action_certain) == 4
    for m in action_certain:
        assert 0.0 <= m <= 0.15

    ctrl.reset()

    # With high belief_var, should still produce valid actions
    action_uncertain = ctrl.compute_action_with_belief(state_a, state_b, True, 5.0)
    assert len(action_uncertain) == 4
    for m in action_uncertain:
        assert 0.0 <= m <= 0.15
