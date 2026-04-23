"""Tests for self-play PPO training."""

from aces.training import CurriculumTrainer, SelfPlayTrainer, evaluate


def test_trainer_creation():
    trainer = SelfPlayTrainer(total_timesteps=512, n_steps=256, max_episode_steps=50)
    assert trainer.model is not None


def test_trainer_short_run():
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
    )
    trainer.train()
    assert trainer.model is not None


def test_opponent_updates():
    trainer = SelfPlayTrainer(
        total_timesteps=1024,
        n_steps=256,
        batch_size=64,
        opponent_update_interval=256,
        max_episode_steps=50,
    )
    trainer.train()
    assert trainer.opponent_update_count > 0


def test_trainer_with_noise():
    """Training with wind and observation noise should work."""
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
        wind_sigma=0.3,
        obs_noise_std=0.1,
    )
    trainer.train()
    assert trainer.model is not None
    assert trainer.stats["episodes"] > 0


def test_trainer_no_noise_override():
    """Explicitly disabling noise should work."""
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
        wind_sigma=0.0,
        obs_noise_std=0.0,
    )
    trainer.train()
    assert trainer.model is not None


def test_trainer_stats():
    """Training should produce summary statistics."""
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
    )
    trainer.train()
    stats = trainer.stats
    assert "episodes" in stats
    assert "mean_reward" in stats
    assert "kills" in stats
    assert "deaths" in stats


def test_evaluate_function(tmp_path):
    """Evaluate function should run and return results dict."""
    # Train a tiny model first
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
    )
    trainer.train()
    model_path = str(tmp_path / "test_model")
    trainer.save(model_path)

    # Evaluate
    results = evaluate(
        model_path=model_path,
        n_episodes=3,
        max_episode_steps=50,
        opponent="random",
    )
    assert results["n_episodes"] == 3
    assert "win_rate" in results
    assert "avg_survival_time" in results
    assert "mean_reward" in results
    assert isinstance(results["mean_reward"], float)


def test_evaluate_with_noise(tmp_path):
    """Evaluation with noise enabled should work."""
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
    )
    trainer.train()
    model_path = str(tmp_path / "test_model_noise")
    trainer.save(model_path)

    results = evaluate(
        model_path=model_path,
        n_episodes=3,
        max_episode_steps=50,
        opponent="random",
        wind_sigma=0.3,
        obs_noise_std=0.1,
    )
    assert results["n_episodes"] == 3
    assert isinstance(results["win_rate"], float)


def test_trainer_fpv():
    """FPV trainer should use MultiInputPolicy with CnnImuExtractor."""
    trainer = SelfPlayTrainer(
        total_timesteps=512,
        n_steps=256,
        batch_size=64,
        max_episode_steps=50,
        fpv=True,
    )
    assert trainer._fpv is True
    trainer.train()
    assert trainer.model is not None
    assert trainer.stats["episodes"] > 0


def test_curriculum_trainer_runs():
    """CurriculumTrainer runs 2 stages end-to-end."""
    trainer = CurriculumTrainer(
        stages=[
            {"task": "pursuit_linear", "timesteps": 256},
            {"task": "pursuit_evasive", "timesteps": 256},
        ],
        n_steps=128,
        batch_size=64,
    )
    model = trainer.train()
    assert model is not None
    assert len(trainer.stage_stats) == 2


def test_curriculum_trainer_saves_models(tmp_path):
    """Each stage saves a model checkpoint."""
    trainer = CurriculumTrainer(
        stages=[
            {"task": "pursuit_linear", "timesteps": 128},
        ],
        n_steps=128,
        batch_size=64,
        save_dir=str(tmp_path),
    )
    trainer.train()
    assert (tmp_path / "stage0_pursuit_linear.zip").exists()


def test_curriculum_full_pipeline():
    """Full 4-stage curriculum runs without error and produces models."""
    trainer = CurriculumTrainer(
        stages=[
            {"task": "pursuit_linear", "timesteps": 128},
            {"task": "pursuit_evasive", "timesteps": 128},
            {"task": "search_pursuit", "timesteps": 128},
            {"task": "dogfight", "timesteps": 128},
        ],
        n_steps=128,
        batch_size=64,
    )
    model = trainer.train()
    assert model is not None
    assert len(trainer.stage_stats) == 4
