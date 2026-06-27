"""Tests for the curriculum-learning wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from rotorenv.envs.curriculum import CurriculumWrapper
from rotorenv.envs.waypoint_env import WaypointEnv


def _run_episode(env, action=None) -> None:
    """Drive one episode to termination/truncation."""
    env.reset(seed=0)
    term = trunc = False
    act = action if action is not None else np.zeros(env.action_space.shape, dtype=np.float32)
    steps = 0
    while not (term or trunc) and steps < 700:
        _o, _r, term, trunc, _i = env.step(act)
        steps += 1


def test_invalid_mode_raises() -> None:
    """An unknown schedule mode is rejected."""
    with pytest.raises(ValueError, match="mode must be"):
        CurriculumWrapper(WaypointEnv(), mode="quantum")


def test_reset_injects_difficulty_into_options() -> None:
    """The wrapper passes its difficulty to the env and reports it in info."""
    env = CurriculumWrapper(WaypointEnv(), mode="success", start_difficulty=0.3)
    _obs, info = env.reset(seed=0)
    assert info["difficulty"] == pytest.approx(0.3)
    assert env.unwrapped.difficulty == pytest.approx(0.3)


def test_step_mode_anneals_toward_one() -> None:
    """In step mode, difficulty ramps up monotonically with env steps."""
    env = CurriculumWrapper(
        WaypointEnv(), mode="step", start_difficulty=0.0, anneal_steps=200
    )
    env.reset(seed=0)
    d_start = env.difficulty
    for _ in range(150):
        _o, _r, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
        if term or trunc:
            env.reset(seed=0)
    assert env.difficulty > d_start
    # After exceeding anneal_steps it saturates at 1.0.
    for _ in range(200):
        _o, _r, term, trunc, _i = env.step(np.zeros(4, dtype=np.float32))
        if term or trunc:
            env.reset(seed=0)
    assert env.difficulty == pytest.approx(1.0)


def test_success_mode_advances_on_wins() -> None:
    """Difficulty increases once the success window is full of successes."""
    env = CurriculumWrapper(
        WaypointEnv(),
        mode="success",
        start_difficulty=0.2,
        difficulty_step=0.1,
        success_threshold=0.7,
        window=5,
    )
    # Force "success" by reporting a tiny distance regardless of dynamics.
    env.success_distance = float("inf")  # every episode counts as a success
    start = env.difficulty
    for _ in range(5):
        _run_episode(env)
    assert env.difficulty > start


def test_success_mode_regresses_on_losses() -> None:
    """Difficulty decreases when the success rate collapses."""
    env = CurriculumWrapper(
        WaypointEnv(),
        mode="success",
        start_difficulty=0.8,
        difficulty_step=0.1,
        failure_threshold=0.2,
        window=5,
    )
    env.success_distance = 0.0  # nothing can succeed -> all failures
    start = env.difficulty
    for _ in range(5):
        _run_episode(env)
    assert env.difficulty < start


def test_difficulty_clamped_to_unit_interval() -> None:
    """Difficulty never escapes [0, 1] under repeated advancement."""
    env = CurriculumWrapper(
        WaypointEnv(), mode="success", start_difficulty=0.95,
        difficulty_step=0.2, success_threshold=0.5, window=3,
    )
    env.success_distance = float("inf")
    for _ in range(30):
        _run_episode(env)
    assert 0.0 <= env.difficulty <= 1.0
