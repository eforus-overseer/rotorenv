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

_ENTRY_POINT = "rotorenv.envs.hover_env:HoverEnv"
_MAX_EPISODE_STEPS = 500  # 10 s at dt=0.02

# Registered task variants. Each is a (kwargs) configuration over the same
# HoverEnv, mirroring how mature env libraries expose many small registered
# variants (MiniGrid) and select behaviour via registry kwargs
# (gym-pybullet-drones). Defaults: FULL (16-D) obs, point-mass physics.
_VARIANTS: dict[str, dict[str, Any]] = {
    "Hover-v0": {},
    "Hover6DOF-v0": {"physics_model": "six_dof"},
    "HoverMinimal-v0": {"observation_type": "minimal"},
    "HoverThrustOnly-v0": {"action_type": "thrust_only"},
}

for _env_id, _kwargs in _VARIANTS.items():
    if _env_id not in gym.registry:
        register(
            id=_env_id,
            entry_point=_ENTRY_POINT,
            max_episode_steps=_MAX_EPISODE_STEPS,
            kwargs=_kwargs,
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
