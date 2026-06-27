"""Tests for the HoverEnv Gymnasium integration."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import rotorenv
from rotorenv.core.enums import ACTION_DIMS, OBSERVATION_DIMS, ActionType, ObservationType
from rotorenv.envs.hover_env import HoverEnv

# Default space dims: FULL observation (16,) and ATTITUDE action (4,).
OBS_DIM = OBSERVATION_DIMS[ObservationType.FULL]
ACT_DIM = ACTION_DIMS[ActionType.ATTITUDE]


def test_registered_and_makeable() -> None:
    """The env is registered and constructible via rotorenv.make."""
    env = rotorenv.make("Hover-v0")
    assert env is not None
    env.close()


def test_observation_and_action_space_shapes() -> None:
    """Default spaces: FULL obs (16,), ATTITUDE action (4,)."""
    env = HoverEnv()
    assert env.observation_space.shape == (OBS_DIM,)
    assert env.action_space.shape == (ACT_DIM,)


def test_minimal_observation_is_13d() -> None:
    """The MINIMAL observation reproduces the legacy (13,) layout."""
    env = HoverEnv(observation_type="minimal")
    assert env.observation_space.shape == (13,)
    obs, _info = env.reset(seed=0)
    assert obs.shape == (13,)


def test_thrust_only_action_is_1d() -> None:
    """THRUST_ONLY exposes a 1-D action and holds attitude commands at zero."""
    env = HoverEnv(action_type="thrust_only")
    assert env.action_space.shape == (1,)
    env.reset(seed=0)
    obs, _r, terminated, _trunc, _info = env.step(np.array([1.0], dtype=np.float32))
    assert not terminated  # full thrust climbs
    assert obs[2] > 0.0


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


def test_six_dof_variant_registered_and_uses_six_dof() -> None:
    """Hover6DOF-v0 is registered and runs on the SixDOFPhysics backend."""
    from rotorenv.physics.six_dof import SixDOFPhysics

    env = rotorenv.make("Hover6DOF-v0")
    inner = env.unwrapped
    assert isinstance(inner.physics, SixDOFPhysics)
    obs, _info = env.reset(seed=0)
    assert inner.observation_space.contains(obs)
    env.step(np.zeros(4, dtype=np.float32))
    env.close()


def test_default_env_uses_point_mass() -> None:
    """The default Hover-v0 still uses the Phase-1 point-mass backend."""
    from rotorenv.physics.point_mass import PointMassPhysics

    env = HoverEnv()
    assert isinstance(env.physics, PointMassPhysics)


def test_unknown_physics_model_raises() -> None:
    """An unknown physics_model name is rejected with a clear error."""
    import pytest

    with pytest.raises(ValueError, match="Unknown physics_model"):
        HoverEnv(physics_model="warp_drive")
