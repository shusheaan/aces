"""Tests for aces.curriculum — TOML-driven curriculum manager."""

import pytest

from aces.curriculum import CurriculumManager, Phase, load_curriculum


# ---------------------------------------------------------------------------
# load_curriculum
# ---------------------------------------------------------------------------


def test_load_curriculum() -> None:
    """Load 6 phases from curriculum.toml; check first and last."""
    phases = load_curriculum()
    assert len(phases) == 6
    assert phases[0].name == "hover_stabilize"
    assert phases[0].task == "hover"
    assert phases[1].name == "pursuit_linear"
    assert phases[-1].use_fpv is True


# ---------------------------------------------------------------------------
# CurriculumManager — initial state
# ---------------------------------------------------------------------------


def test_curriculum_manager_initial_phase() -> None:
    """Manager starts at phase 0."""
    phases = load_curriculum()
    mgr = CurriculumManager(phases)
    assert mgr.phase_index == 0
    assert mgr.current_phase().name == "hover_stabilize"


# ---------------------------------------------------------------------------
# Promotion logic
# ---------------------------------------------------------------------------


def test_curriculum_promote() -> None:
    """Not enough data -> False; enough data + condition met -> True."""
    phases = [
        Phase(
            name="test_phase",
            promote_condition="win_rate>0.3",
            promote_window=100,
        ),
        Phase(name="next_phase"),
    ]
    mgr = CurriculumManager(phases)

    # Not enough episodes
    assert mgr.should_promote({"episodes": 50, "win_rate": 0.5}) is False

    # Enough episodes but condition not met
    assert mgr.should_promote({"episodes": 100, "win_rate": 0.2}) is False

    # Enough episodes and condition met
    assert mgr.should_promote({"episodes": 100, "win_rate": 0.5}) is True

    # Promote advances to next phase
    new_phase = mgr.promote()
    assert new_phase is not None
    assert new_phase.name == "next_phase"
    assert mgr.phase_index == 1


def test_curriculum_promote_steps_condition() -> None:
    """'steps' condition always returns False (model.learn runs to completion)."""
    phases = [
        Phase(name="step_phase", promote_condition="steps"),
        Phase(name="next"),
    ]
    mgr = CurriculumManager(phases)
    # Even with plenty of episodes and a high win_rate, "steps" never promotes
    assert mgr.should_promote({"episodes": 0}) is False
    assert mgr.should_promote({"episodes": 1000, "win_rate": 0.99}) is False


def test_curriculum_promote_reward_condition() -> None:
    """'reward>X' condition works like win_rate."""
    phases = [
        Phase(
            name="reward_phase",
            promote_condition="reward>50.0",
            promote_window=10,
        ),
        Phase(name="next"),
    ]
    mgr = CurriculumManager(phases)

    assert mgr.should_promote({"episodes": 10, "reward": 30.0}) is False
    assert mgr.should_promote({"episodes": 10, "reward": 60.0}) is True


# ---------------------------------------------------------------------------
# Last-phase behaviour
# ---------------------------------------------------------------------------


def test_curriculum_promote_last_phase() -> None:
    """Promote returns None at the last phase."""
    phases = [Phase(name="only_phase")]
    mgr = CurriculumManager(phases)

    assert mgr.is_last_phase() is True
    assert mgr.promote() is None
    assert mgr.phase_index == 0  # unchanged


# ---------------------------------------------------------------------------
# State dict round-trip
# ---------------------------------------------------------------------------


def test_curriculum_state_dict() -> None:
    """Round-trip state_dict / load_state_dict preserves phase_index."""
    phases = load_curriculum()
    mgr = CurriculumManager(phases)

    mgr.promote()
    mgr.promote()
    assert mgr.phase_index == 2

    state = mgr.state_dict()
    assert state == {"phase_index": 2}

    mgr2 = CurriculumManager(phases)
    mgr2.load_state_dict(state)
    assert mgr2.phase_index == 2
    assert mgr2.current_phase().name == mgr.current_phase().name


# ---------------------------------------------------------------------------
# Bug #28 — promote-condition regex whitespace tolerance
# ---------------------------------------------------------------------------


def test_promote_condition_whitespace_accepted() -> None:
    """'win_rate > 0.30' (with spaces) should parse and promote correctly."""
    phases = [
        Phase(
            name="ws_phase",
            promote_condition="win_rate > 0.30",
            promote_window=10,
        ),
        Phase(name="next"),
    ]
    mgr = CurriculumManager(phases)
    assert mgr.should_promote({"episodes": 10, "win_rate": 0.50}) is True
    assert mgr.should_promote({"episodes": 10, "win_rate": 0.10}) is False


def test_promote_condition_double_gt_rejected() -> None:
    """'win_rate >> 0.30' (double >) should raise ValueError."""
    phases = [
        Phase(
            name="bad_phase",
            promote_condition="win_rate >> 0.30",
            promote_window=10,
        ),
        Phase(name="next"),
    ]
    mgr = CurriculumManager(phases)
    with pytest.raises(ValueError, match="Unknown promote_condition"):
        mgr.should_promote({"episodes": 100, "win_rate": 0.99})
