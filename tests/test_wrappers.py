"""Tests for example wrappers and vectorized-env support."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import rotorenv
from rotorenv.envs.hover_env import HoverEnv
from rotorenv.envs.wrappers import NormalizeObservation, RewardScale


def test_normalize_observation_in_unit_box() -> None:
    """Normalised observations live in [-1, 1] and the space reflects that."""
    env = NormalizeObservation(HoverEnv())
    assert env.observation_space.low.min() == -1.0
    assert env.observation_space.high.max() == 1.0
    obs, _info = env.reset(seed=0)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    for _ in range(20):
        obs, _r, term, trunc, _i = env.step(env.action_space.sample())
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
        if term or trunc:
            break
    env.close()


def test_reward_scale_multiplies() -> None:
    """RewardScale multiplies the underlying reward by the scale factor."""
    seed = 7
    action = np.zeros(4, dtype=np.float32)

    base = HoverEnv()
    base.reset(seed=seed)
    _o, base_r, _t, _tr, _i = base.step(action)

    scaled = RewardScale(HoverEnv(), scale=10.0)
    scaled.reset(seed=seed)
    _o, scaled_r, _t, _tr, _i = scaled.step(action)

    assert scaled_r == base_r * 10.0


def test_wrapped_env_still_passes_check_env() -> None:
    """A wrapped env remains Gymnasium-conformant.

    We deliberately check the *wrapped* env, so silence env_checker's expected
    "you passed a wrapped env" advisory.
    """
    import warnings

    from gymnasium.utils.env_checker import check_env

    env = NormalizeObservation(HoverEnv())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*different from the unwrapped.*")
        check_env(env, skip_render_check=True)
    env.close()


def test_vectorized_envs_run() -> None:
    """make_vec builds parallel copies that step together (guide pattern)."""
    vec = gym.make_vec("Hover-v0", num_envs=3)
    obs, _info = vec.reset(seed=0)
    assert obs.shape[0] == 3
    actions = np.zeros((3,) + vec.single_action_space.shape, dtype=np.float32)
    obs, rewards, terminated, truncated, _info = vec.step(actions)
    assert obs.shape[0] == 3
    assert rewards.shape == (3,)
    vec.close()
