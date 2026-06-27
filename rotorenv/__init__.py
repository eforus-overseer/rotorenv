"""rotorenv — a Gymnasium-compatible drone RL environment.

Importing this package registers the environment IDs with Gymnasium's global
registry, so ``rotorenv.make("Hover-v0")`` (or ``gymnasium.make("Hover-v0")``)
works after ``import rotorenv``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.envs.registration import register

__version__ = "0.1.0"

# Register environment IDs exactly once, even on repeated imports.
if "Hover-v0" not in gym.registry:
    register(
        id="Hover-v0",
        entry_point="rotorenv.envs.hover_env:HoverEnv",
        max_episode_steps=500,  # 10 s at dt=0.02
    )


def make(env_id: str, **kwargs: Any) -> gym.Env:
    """Create a registered rotorenv environment.

    Thin wrapper over :func:`gymnasium.make` provided so that callers can use
    ``rotorenv.make(...)`` without importing gymnasium directly.

    Args:
        env_id: A registered environment ID, e.g. ``"Hover-v0"``.
        **kwargs: Forwarded to :func:`gymnasium.make`.

    Returns:
        The constructed environment.
    """
    return gym.make(env_id, **kwargs)


__all__ = ["make", "__version__"]
