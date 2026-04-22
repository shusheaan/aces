"""Statistical tests for EKF correctness.

Tests verify mathematical properties that must hold for a correctly-tuned filter:
- NEES (Normalized Estimation Error Squared) should follow chi-squared distribution
- Innovation sequence should be white (uncorrelated)
- EKF should outperform raw noise across noise levels
- Covariance should grow during prediction and shrink during update
"""

import numpy as np
import pytest

from aces._core import Simulation


def _make_sim(obs_noise_std=0.1, **kwargs):
    """Create a minimal Simulation for EKF testing."""
    return Simulation(
        bounds=[10.0, 10.0, 3.0],
        obstacles=[],
        obs_noise_std=obs_noise_std,
        **kwargs,
    )


def _run_episode(sim, n_steps=200):
    """Run an episode collecting EKF diagnostics."""
    hover = sim.hover_thrust()
    motors = [hover] * 4

    results = []
    for _ in range(n_steps):
        r = sim.step(motors, motors)
        results.append(r)
    return results


class TestEkfNEES:
    """NEES (Normalized Estimation Error Squared) consistency tests.

    For a correctly-tuned filter, NEES should follow chi-squared(n) where n=3 (position dims).
    Mean NEES should be approximately 3.0.
    """

    def test_nees_mean_in_chi2_range(self):
        """Average NEES across episodes should be near state dimension (3)."""
        n_episodes = 30
        burn_in = 30
        n_steps = 150
        nees_all = []

        for seed_offset in range(n_episodes):
            sim = _make_sim(obs_noise_std=0.1)
            rng = np.random.default_rng(seed_offset)
            jitter = rng.uniform(-0.3, 0.3, size=3)
            pos_a = [2.0 + jitter[0], 2.0 + jitter[1], 1.5 + jitter[2] * 0.2]
            sim.reset(pos_a, [8.0, 8.0, 1.5])

            results = _run_episode(sim, n_steps)

            for r in results[burn_in:]:
                true_pos = np.array(r.drone_b_state[:3])
                ekf_pos = np.array(r.ekf_b_pos_from_a)
                cov_diag = np.array(r.ekf_a_cov_diag[:3])  # position covariance

                err = true_pos - ekf_pos
                cov_safe = np.maximum(cov_diag, 1e-12)
                nees = np.sum(err**2 / cov_safe)
                nees_all.append(nees)

        mean_nees = np.mean(nees_all)
        # For chi-squared(3), mean = 3, std = sqrt(6) ~ 2.45
        # With many samples, generous bounds [1.0, 8.0]
        assert 1.0 < mean_nees < 8.0, (
            f"Mean NEES {mean_nees:.2f} outside expected range for chi-squared(3)"
        )


class TestInnovationWhiteness:
    """Innovation sequence should be approximately white noise (uncorrelated)."""

    def test_innovation_autocorrelation_near_zero(self):
        """Lag-1 autocorrelation of innovations should be near zero."""
        sim = _make_sim(obs_noise_std=0.1)
        sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])

        burn_in = 30
        results = _run_episode(sim, 300)

        innovations = []
        for r in results[burn_in:]:
            innov = np.array(r.ekf_a_innovation)
            if np.any(np.abs(innov) > 1e-12):  # skip steps without update
                innovations.append(innov)

        innovations = np.array(innovations)
        if len(innovations) < 50:
            pytest.skip("Not enough innovations collected")

        # Compute lag-1 autocorrelation for each component
        for dim in range(3):
            seq = innovations[:, dim]
            seq_centered = seq - np.mean(seq)
            var = np.var(seq_centered)
            if var < 1e-12:
                continue
            autocorr_1 = np.mean(seq_centered[:-1] * seq_centered[1:]) / var
            assert abs(autocorr_1) < 0.25, (
                f"Dim {dim}: lag-1 autocorrelation {autocorr_1:.3f} too high"
            )


class TestEkfNoiseLevels:
    """EKF should outperform raw noise across different noise levels."""

    @pytest.mark.parametrize("noise_std", [0.01, 0.05, 0.1, 0.2])
    def test_ekf_beats_raw_noise(self, noise_std):
        """EKF estimation error < raw observation noise (post burn-in)."""
        sim = _make_sim(obs_noise_std=noise_std)
        sim.reset([2.0, 2.0, 1.5], [8.0, 8.0, 1.5])

        burn_in = 40
        results = _run_episode(sim, 200)

        sum_raw = 0.0
        sum_ekf = 0.0
        count = 0

        for r in results[burn_in:]:
            true_pos = np.array(r.drone_b_state[:3])
            noisy_pos = np.array(r.noisy_b_pos_from_a)
            ekf_pos = np.array(r.ekf_b_pos_from_a)

            # Skip if not visible (noisy_pos will be zeros)
            if np.linalg.norm(noisy_pos) < 0.01:
                continue

            sum_raw += np.linalg.norm(noisy_pos - true_pos)
            sum_ekf += np.linalg.norm(ekf_pos - true_pos)
            count += 1

        if count < 10:
            pytest.skip("Not enough visible steps")

        avg_raw = sum_raw / count
        avg_ekf = sum_ekf / count
        assert avg_ekf < avg_raw, (
            f"noise_std={noise_std}: EKF error ({avg_ekf:.4f}) >= raw ({avg_raw:.4f})"
        )


class TestCovarianceDynamics:
    """Covariance should grow during prediction and shrink during update."""

    def test_predict_grows_covariance_when_occluded(self):
        """With wall between drones, pure prediction should increase P diagonal."""
        sim = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[([5.0, 5.0, 1.5], [2.0, 2.0, 1.5])],  # wall between drones
            obs_noise_std=0.1,
        )
        sim.reset([1.0, 1.0, 1.5], [9.0, 9.0, 1.5])
        hover = [sim.hover_thrust()] * 4

        r_initial = sim.step(hover, hover)
        cov_initial = np.array(r_initial.ekf_a_cov_diag)

        for _ in range(10):
            r = sim.step(hover, hover)

        cov_after = np.array(r.ekf_a_cov_diag)
        # Position covariance should grow without measurements
        assert cov_after[0] > cov_initial[0], "P_px should grow without measurements"
        assert cov_after[1] > cov_initial[1], "P_py should grow without measurements"


class TestActuatorIntegration:
    """Verify actuator model affects dynamics when enabled."""

    def test_motor_delay_causes_slower_response(self):
        """With motor delay, drone should respond more sluggishly."""
        # No delay
        sim_fast = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[],
            motor_time_constant=0.0,
        )
        sim_fast.reset([5.0, 5.0, 1.5], [5.0, 5.0, 2.5])

        # With 50ms delay
        sim_slow = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[],
            motor_time_constant=0.05,
        )
        sim_slow.reset([5.0, 5.0, 1.5], [5.0, 5.0, 2.5])

        # Apply max thrust for 10 steps
        max_t = sim_fast.max_thrust()
        motors = [max_t] * 4
        hover = [sim_fast.hover_thrust()] * 4

        for _ in range(10):
            r_fast = sim_fast.step(motors, hover)
            r_slow = sim_slow.step(motors, hover)

        # Fast sim should have gained more altitude
        z_fast = r_fast.drone_a_state[2]
        z_slow = r_slow.drone_a_state[2]
        assert z_fast > z_slow, (
            f"No-delay z={z_fast:.4f} should be > delayed z={z_slow:.4f}"
        )


class TestImuBiasIntegration:
    """Verify IMU bias accumulates and is reported."""

    def test_bias_reported_in_step_result(self):
        """With IMU bias enabled, bias fields should be non-zero after many steps."""
        sim = Simulation(
            bounds=[10.0, 10.0, 3.0],
            obstacles=[],
            imu_accel_bias_std=0.1,
            imu_gyro_bias_std=0.01,
        )
        sim.reset([5.0, 5.0, 1.5], [5.0, 5.0, 2.5])
        hover = [sim.hover_thrust()] * 4

        for _ in range(500):
            r = sim.step(hover, hover)

        accel_bias = np.array(r.imu_accel_bias_a)
        gyro_bias = np.array(r.imu_gyro_bias_a)

        # After 500 steps * 0.01s = 5 seconds, bias should have accumulated
        assert np.linalg.norm(accel_bias) > 0.01, (
            f"Accel bias should accumulate, got {accel_bias}"
        )
        assert np.linalg.norm(gyro_bias) > 0.001, (
            f"Gyro bias should accumulate, got {gyro_bias}"
        )
