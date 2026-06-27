"""Example Gymnasium wrappers for rotorenv.

Wrappers modify an environment's behaviour without touching its core
implementation — the idiomatic Gymnasium way to add observation/reward
transforms. These are thin demonstrations of the pattern; compose them with
``gymnasium.wrappers`` (e.g. ``TimeLimit``, ``FlattenObservation``) as needed.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.ObservationWrapper):
    """Rescale observations from the env's finite bounds into ``[-1, 1]``.

    Relies on the wrapped env exposing a finite ``Box`` observation space (which
    :class:`~rotorenv.envs.base_env.DroneEnv` does). Fields whose bound is zero
    are passed through unchanged to avoid division by zero.
    """

    def __init__(self, env: gym.Env) -> None:
        """Wrap ``env`` and expose a ``[-1, 1]`` observation space.

        Args:
            env: The environment to wrap; must have a finite ``Box`` obs space.
        """
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float64)
        high = np.asarray(env.observation_space.high, dtype=np.float64)
        self._center = (high + low) / 2.0
        self._half_range = (high - low) / 2.0
        self._safe = self._half_range != 0.0
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Map a raw observation into ``[-1, 1]`` per-field."""
        obs = np.asarray(observation, dtype=np.float64)
        scaled = obs.copy()
        scaled[self._safe] = (
            (obs[self._safe] - self._center[self._safe]) / self._half_range[self._safe]
        )
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)


class RewardScale(gym.RewardWrapper):
    """Multiply every step reward by a constant ``scale`` factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0) -> None:
        """Wrap ``env`` and scale its rewards.

        Args:
            env: The environment to wrap.
            scale: Multiplicative factor applied to each step's reward.
        """
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward: float) -> float:
        """Return ``reward * scale``."""
        return float(reward) * self.scale
