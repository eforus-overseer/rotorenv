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

_HOVER = "rotorenv.envs.hover_env:HoverEnv"
_WAYPOINT = "rotorenv.envs.waypoint_env:WaypointEnv"
_TRAJECTORY = "rotorenv.envs.trajectory_env:TrajectoryEnv"

# Registered task variants: (entry_point, max_episode_steps, kwargs). Many small
# registered variants over a few task classes mirrors MiniGrid; selecting
# behaviour via registry kwargs mirrors gym-pybullet-drones. Episode caps follow
# each task's time limit at dt=0.02 (500 = 10 s, 600 = 12 s). Defaults: FULL
# (16-D) obs, ATTITUDE action, point-mass physics.
_VARIANTS: dict[str, tuple[str, int, dict[str, Any]]] = {
    # Hover family
    "Hover-v0": (_HOVER, 500, {}),
    "Hover6DOF-v0": (_HOVER, 500, {"physics_model": "six_dof"}),
    "HoverMinimal-v0": (_HOVER, 500, {"observation_type": "minimal"}),
    "HoverThrustOnly-v0": (_HOVER, 500, {"action_type": "thrust_only"}),
    # Airborne-spawn hover: starts at target height, so the task is pure
    # attitude-stabilised hovering (no takeoff). Far more learnable from scratch.
    "HoverEasy-v0": (_HOVER, 500, {"physics_model": "six_dof", "spawn_height": 1.0}),
    # Waypoint family (Phase 4)
    "Waypoint-v0": (_WAYPOINT, 500, {}),
    "Waypoint6DOF-v0": (_WAYPOINT, 500, {"physics_model": "six_dof"}),
    # Trajectory family (Phase 4)
    "Trajectory-v0": (_TRAJECTORY, 600, {}),
    "Trajectory6DOF-v0": (_TRAJECTORY, 600, {"physics_model": "six_dof"}),
}

for _env_id, (_entry, _max_steps, _kwargs) in _VARIANTS.items():
    if _env_id not in gym.registry:
        register(
            id=_env_id,
            entry_point=_entry,
            max_episode_steps=_max_steps,
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
