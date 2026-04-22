import gymnasium as gym
import numpy as np
from aces.env import DroneDogfightEnv


def test_env_creation():
    env = DroneDogfightEnv()
    assert env.observation_space.shape == (21,)
    assert env.action_space.shape == (4,)


def test_env_reset():
    env = DroneDogfightEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (21,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_env_step():
    env = DroneDogfightEnv()
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (21,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "agent_pos" in info
    assert "opponent_pos" in info


def test_env_episode_runs():
    env = DroneDogfightEnv(max_episode_steps=50)
    obs, _ = env.reset(seed=42)
    total_reward = 0.0
    steps = 0
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    assert steps > 0
    assert isinstance(total_reward, float)


def test_env_with_mppi_opponent():
    env = DroneDogfightEnv(opponent="mppi", mppi_samples=32, mppi_horizon=5)
    obs, _ = env.reset(seed=42)
    obs2, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs2.shape == (21,)


def test_env_with_noise():
    """Env with noise enabled should produce valid observations with EKF."""
    env = DroneDogfightEnv(
        max_episode_steps=50,
        wind_sigma=0.3,
        obs_noise_std=0.1,
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (21,)
    assert np.all(np.isfinite(obs))

    # Run a few steps
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (21,)
        assert np.all(np.isfinite(obs))
        assert "ekf_opponent_pos" in info
        assert "wind_force" in info
        if terminated or truncated:
            break


def test_env_noise_disabled_explicitly():
    """Explicitly disabling noise should give clean observations."""
    env = DroneDogfightEnv(
        max_episode_steps=50,
        wind_sigma=0.0,
        obs_noise_std=0.0,
    )
    obs, _ = env.reset(seed=42)
    obs2, _, _, _, _ = env.step(env.action_space.sample())
    assert obs2.shape == (21,)
    assert np.all(np.isfinite(obs2))


def test_env_visibility_info():
    """Env info dict should include visibility and belief state."""
    env = DroneDogfightEnv(max_episode_steps=50)
    obs, _ = env.reset(seed=42)
    obs, _, _, _, info = env.step(env.action_space.sample())

    assert "opponent_visible" in info
    assert "belief_pos" in info
    assert "belief_var" in info
    assert "time_since_last_seen" in info
    assert isinstance(info["opponent_visible"], bool)
    assert isinstance(info["belief_var"], float)

    # obs should be 21-dim with visibility features
    assert obs.shape == (21,)
    # Last 3 dims: opponent_visible, belief_uncertainty, time_since_last_seen
    assert np.isfinite(obs[-3])  # opponent_visible
    assert np.isfinite(obs[-2])  # belief_uncertainty
    assert np.isfinite(obs[-1])  # time_since_last_seen


def test_env_gymnasium_api():
    env = DroneDogfightEnv(max_episode_steps=10)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(3):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


# ---------------------------------------------------------------------------
# Level 4: FPV / Camera tests
# ---------------------------------------------------------------------------


def test_fpv_env_creation():
    """FPV env should have Dict observation space."""
    env = DroneDogfightEnv(fpv=True)
    assert isinstance(env.observation_space, gym.spaces.Dict)
    assert "image" in env.observation_space.spaces
    assert "vector" in env.observation_space.spaces
    assert env.observation_space["image"].shape == (1, 60, 80)
    assert env.observation_space["vector"].shape == (12,)


def test_fpv_env_reset():
    """FPV reset returns dict observation with correct shapes."""
    env = DroneDogfightEnv(fpv=True)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert obs["image"].shape == (1, 60, 80)
    assert obs["vector"].shape == (12,)
    assert obs["image"].dtype == np.float32
    assert obs["vector"].dtype == np.float32
    assert np.all(np.isfinite(obs["image"]))
    assert np.all(np.isfinite(obs["vector"]))


def test_fpv_env_step():
    """FPV step returns dict observation and depth image values in valid range."""
    env = DroneDogfightEnv(fpv=True, max_episode_steps=20)
    obs, _ = env.reset(seed=42)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(obs, dict)
        assert obs["image"].shape == (1, 60, 80)
        assert obs["vector"].shape == (12,)
        # Depth image values should be in [0, 1] (normalized)
        assert obs["image"].min() >= 0.0
        assert obs["image"].max() <= 1.0
        assert np.all(np.isfinite(obs["image"]))
        assert np.all(np.isfinite(obs["vector"]))
        if terminated or truncated:
            break


def test_fpv_detection_info():
    """FPV step info should include detection data."""
    env = DroneDogfightEnv(fpv=True, max_episode_steps=10)
    obs, _ = env.reset(seed=42)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert "detection" in info
    det = info["detection"]
    assert "detected" in det
    assert "bbox" in det
    assert "confidence" in det
    assert "depth" in det
    assert "pixel_center" in det
    assert isinstance(det["detected"], bool)


def test_fpv_camera_renders_at_30hz():
    """Camera should render approximately once every 3.3 control steps (100Hz / 30Hz)."""
    env = DroneDogfightEnv(fpv=True, max_episode_steps=100)
    env.reset(seed=42)
    render_count = 0
    total_steps = 60
    for _ in range(total_steps):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if info.get("camera_rendered", False):
            render_count += 1
        if terminated or truncated:
            break
    # At 100 Hz control and 30 Hz render, expect ~18 renders in 60 steps
    # Allow some tolerance
    assert render_count >= 10, f"Expected ~18 renders, got {render_count}"
    assert render_count <= 30, f"Expected ~18 renders, got {render_count}"


def test_fpv_camera_render_time():
    """Measure camera render time per frame.

    Spec target: <10ms per 320x240 frame (release build).
    Debug builds are ~20-50x slower, so this test only validates that
    rendering completes and prints timing info.
    Build with `maturin develop --release` for accurate performance.
    """
    import time

    env = DroneDogfightEnv(fpv=True, max_episode_steps=200)
    env.reset(seed=42)

    render_times = []
    non_render_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000
        if info.get("camera_rendered", False):
            render_times.append(elapsed)
        else:
            non_render_times.append(elapsed)
        if terminated or truncated:
            env.reset(seed=42)

    assert len(render_times) > 0, "No camera renders occurred"
    avg_render = sum(render_times) / len(render_times)
    avg_no_render = (
        sum(non_render_times) / len(non_render_times) if non_render_times else 0
    )
    # Estimate render-only cost by subtracting non-render step time
    render_cost = avg_render - avg_no_render
    print(
        f"\n  Camera perf ({len(render_times)} renders): "
        f"render_step={avg_render:.1f}ms, "
        f"non_render_step={avg_no_render:.1f}ms, "
        f"est_render_only={render_cost:.1f}ms"
    )


def test_fpv_gymnasium_api():
    """FPV env should pass gymnasium API checks."""
    env = DroneDogfightEnv(fpv=True, max_episode_steps=10)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_env_task_pursuit_evasive():
    """pursuit_evasive task creates env, steps without error."""
    env = DroneDogfightEnv(task="pursuit_evasive", max_episode_steps=10)
    obs, info = env.reset(seed=42)
    assert obs.shape == (21,)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_env_task_default_is_dogfight():
    """Default task is dogfight (backward compatible)."""
    env = DroneDogfightEnv(max_episode_steps=5)
    assert env._task == "dogfight"


def test_env_task_reward_overrides():
    """pursuit_linear overrides approach_reward."""
    env = DroneDogfightEnv(task="pursuit_linear", max_episode_steps=5)
    assert env._reward_cfg["approach_reward"] == 0.2
    assert env._reward_cfg["info_gain_reward"] == 0.0


def test_env_task_search_pursuit_reward():
    """search_pursuit boosts info_gain_reward."""
    env = DroneDogfightEnv(task="search_pursuit", max_episode_steps=5)
    assert env._reward_cfg["info_gain_reward"] == 0.1
    assert env._reward_cfg["lost_contact_penalty"] == 0.02


def test_env_task_pursuit_linear():
    """pursuit_linear task steps without error."""
    env = DroneDogfightEnv(task="pursuit_linear", max_episode_steps=10)
    obs, info = env.reset(seed=42)
    assert obs.shape == (21,)
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break


def test_env_search_pursuit_occluded_spawn():
    """search_pursuit spawns drones out of line-of-sight."""
    env = DroneDogfightEnv(task="search_pursuit", max_episode_steps=10)
    occluded_count = 0
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        # obs[18] is opponent_visible in the 21-dim vector
        if obs[18] < 0.5:
            occluded_count += 1
    # At least half should be occluded (fallback allows some visible)
    assert occluded_count >= 10, f"Only {occluded_count}/20 were occluded"
