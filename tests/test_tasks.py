"""Tests for the Phase-4 task environments (waypoint, trajectory)."""

from __future__ import annotations

import numpy as np
import pytest

import rotorenv
from rotorenv.envs.trajectory_env import TrajectoryEnv
from rotorenv.envs.waypoint_env import WaypointEnv


@pytest.mark.parametrize(
    "env_id",
    ["Waypoint-v0", "Waypoint6DOF-v0", "Trajectory-v0", "Trajectory6DOF-v0"],
)
def test_new_tasks_registered_and_runnable(env_id: str) -> None:
    """Each Phase-4 task is registered and runs a step."""
    env = rotorenv.make(env_id)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.step(env.action_space.sample())
    env.close()


def test_waypoint_target_varies_with_seed() -> None:
    """Waypoint targets are sampled (differ across seeds) and reproducible."""
    env = WaypointEnv()
    env.reset(seed=1)
    t1 = env.target.copy()
    env.reset(seed=2)
    t2 = env.target.copy()
    env.reset(seed=1)
    t1_again = env.target.copy()
    assert not np.allclose(t1, t2)            # sampling actually varies
    np.testing.assert_array_equal(t1, t1_again)  # seed-reproducible


def test_waypoint_difficulty_zero_is_nominal() -> None:
    """At difficulty 0 the waypoint collapses to the nominal hover point."""
    env = WaypointEnv()
    env.reset(seed=0, options={"difficulty": 0.0})
    np.testing.assert_allclose(env.target, [0.0, 0.0, 1.0], atol=1e-9)


def test_waypoint_target_stays_off_floor() -> None:
    """Sampled target height is always clamped above ground."""
    env = WaypointEnv()
    for seed in range(25):
        env.reset(seed=seed, options={"difficulty": 1.0})
        assert env.target[2] >= 0.25


def test_trajectory_target_moves_over_time() -> None:
    """The trajectory target changes as the episode advances."""
    env = TrajectoryEnv()
    env.reset(seed=0, options={"difficulty": 1.0})
    start = env.target.copy()
    for _ in range(50):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert not np.allclose(start, env.target)


def test_trajectory_difficulty_zero_is_stationary() -> None:
    """At difficulty 0 the trajectory collapses to the fixed centre point."""
    env = TrajectoryEnv()
    env.reset(seed=0, options={"difficulty": 0.0})
    start = env.target.copy()
    for _ in range(50):
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    np.testing.assert_allclose(env.target, start, atol=1e-9)


def test_trajectory_spawns_on_path() -> None:
    """The drone spawns exactly on the trajectory start point."""
    env = TrajectoryEnv()
    obs, _info = env.reset(seed=0, options={"difficulty": 1.0})
    # distance-to-target (obs[9:12]) should be ~zero at spawn
    np.testing.assert_allclose(obs[9:12], 0.0, atol=1e-6)
