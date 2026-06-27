"""Tests for the HoverEnv Gymnasium integration."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import rotorenv
from rotorenv.envs.base_env import ACT_DIM, OBS_DIM
from rotorenv.envs.hover_env import HoverEnv


def test_registered_and_makeable() -> None:
    """The env is registered and constructible via rotorenv.make."""
    env = rotorenv.make("Hover-v0")
    assert env is not None
    env.close()


def test_observation_and_action_space_shapes() -> None:
    """Spaces match the spec: obs (13,), action (4,)."""
    env = HoverEnv()
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)


def test_reset_returns_valid_obs() -> None:
    """reset returns an observation inside the observation space."""
    env = HoverEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)
    assert "distance" in info


def test_reset_is_seed_reproducible() -> None:
    """Same seed yields identical initial observations."""
    env = HoverEnv()
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_step_returns_five_tuple() -> None:
    """step returns the Gymnasium (obs, reward, terminated, truncated, info) tuple."""
    env = HoverEnv()
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_episode_terminates_or_truncates() -> None:
    """A zero-thrust policy should crash (terminate) within the step budget."""
    env = HoverEnv()
    env.reset(seed=0)
    # Zero raw action -> thrust 0.5 (hover); push thrust to minimum to force a crash.
    crash_action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 600:
        _obs, _r, terminated, truncated, _info = env.step(crash_action)
        steps += 1
    assert terminated  # fell below z=0
    env.close()


def test_upward_thrust_takes_off() -> None:
    """Full upward thrust climbs off the ground without crashing.

    The drone spawns exactly on the crash plane (z=0), so a neutral action under
    spawn tilt noise sinks below z=0. Commanding full thrust must instead climb.
    """
    env = HoverEnv()
    env.reset(seed=0)
    obs, _r, terminated, _trunc, _info = env.step(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    assert not terminated
    assert obs[2] > 0.0   # z position rose above the ground
    assert obs[5] > 0.0   # z velocity is upward


def test_gymnasium_make_also_works() -> None:
    """The ID is in Gymnasium's global registry too."""
    env = gym.make("Hover-v0")
    env.reset(seed=1)
    env.close()
