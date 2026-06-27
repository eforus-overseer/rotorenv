"""Tests for the procedural obstacle-field NavigationEnv (Phase 6)."""

from __future__ import annotations

import numpy as np
import pytest

import rotorenv
from rotorenv.envs.navigation_env import (
    DRONE_RADIUS,
    GOAL_POSITION,
    GOAL_RADIUS,
    START_POSITION,
    NavigationEnv,
)


@pytest.mark.parametrize("env_id", ["Navigation-v0", "Navigation6DOF-v0"])
def test_registered_and_runnable(env_id: str) -> None:
    """Navigation variants are registered and step cleanly."""
    env = rotorenv.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "n_obstacles" in info and "reached_goal" in info
    env.step(env.action_space.sample())
    env.close()


def test_invalid_perception_raises() -> None:
    """An unknown perception mode is rejected."""
    with pytest.raises(ValueError, match="perception"):
        NavigationEnv(perception="telepathy")


def test_spawns_at_start() -> None:
    """The drone spawns at the fixed start position."""
    env = NavigationEnv()
    env.reset(seed=0)
    np.testing.assert_allclose(env.unwrapped.state.position, START_POSITION)


def test_obstacle_count_scales_with_difficulty() -> None:
    """More difficulty -> more obstacles (averaged over seeds to beat sampling noise)."""
    env = NavigationEnv()

    def avg_obstacles(difficulty: float) -> float:
        counts = []
        for seed in range(10):
            env.reset(seed=seed, options={"difficulty": difficulty})
            counts.append(env.obstacles.shape[0])
        return float(np.mean(counts))

    assert avg_obstacles(1.0) > avg_obstacles(0.1)


def test_obstacles_clear_start_and_goal() -> None:
    """No obstacle is generated on top of the start or goal regions."""
    env = NavigationEnv()
    for seed in range(15):
        env.reset(seed=seed, options={"difficulty": 1.0})
        for box in env.obstacles:
            center_xy = box[:2]
            assert np.linalg.norm(center_xy - START_POSITION[:2]) > 0.5
            assert np.linalg.norm(center_xy - GOAL_POSITION[:2]) > 0.5


def test_collision_detected_inside_obstacle() -> None:
    """A position inside an obstacle box is flagged as a collision."""
    env = NavigationEnv()
    env.reset(seed=0, options={"difficulty": 1.0})
    if env.obstacles.shape[0] == 0:
        pytest.skip("no obstacles sampled this seed")
    box_center = env.obstacles[0, :3]
    assert env._in_collision(box_center)


def test_no_collision_in_open_space() -> None:
    """The start position is collision-free (clearance is enforced)."""
    env = NavigationEnv()
    env.reset(seed=0, options={"difficulty": 1.0})
    assert not env._in_collision(START_POSITION)


def test_reaching_goal_terminates_without_crash() -> None:
    """Teleporting to the goal terminates the episode as a success, not a crash."""
    env = NavigationEnv()
    env.reset(seed=0)
    env.unwrapped.state.position = GOAL_POSITION.copy()
    terminated, crashed = env.unwrapped._is_terminated(env.unwrapped.state)
    assert terminated and not crashed


def test_seed_reproducible_layout() -> None:
    """Same seed reproduces the same obstacle field."""
    env = NavigationEnv()
    env.reset(seed=3, options={"difficulty": 1.0})
    layout_a = env.obstacles.copy()
    env.reset(seed=3, options={"difficulty": 1.0})
    np.testing.assert_array_equal(layout_a, env.obstacles)
