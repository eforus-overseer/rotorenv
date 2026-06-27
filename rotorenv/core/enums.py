"""Configuration enums for observation and action spaces.

Following the pattern used by mature Gymnasium drone environments
(e.g. ``gym-pybullet-drones``'s ``ObservationType`` / ``ActionType``), the shape
and meaning of the observation and action spaces are selected by enum rather
than hardcoded. This lets one ``DroneEnv`` expose several space configurations
without changing the step loop.

The enums subclass ``str`` so a value is both a rich enum member and a plain
string. That keeps Gymnasium ``register(kwargs=...)`` entries JSON-simple
(``{"observation_type": "minimal"}``) while still accepting the enum directly.
"""

from __future__ import annotations

from enum import Enum


class ObservationType(str, Enum):
    """Selects which fields are flattened into the observation vector.

    Members:
        MINIMAL: ``position(3) + velocity(3) + orientation(3) +
            distance_to_target(3) + time(1)`` -> shape ``(13,)``. The Phase-1/2
            layout (excludes angular velocity).
        FULL: ``MINIMAL`` plus ``angular_velocity(3)`` -> shape ``(16,)``. The
            complete kinematic state; this is the default.
    """

    MINIMAL = "minimal"
    FULL = "full"


class ActionType(str, Enum):
    """Selects the action-space layout and how raw actions map to commands.

    Members:
        ATTITUDE: ``[thrust, roll, pitch, yaw]`` in ``[-1, 1]`` -> shape ``(4,)``.
            The default high-level command used since Phase 1.
        THRUST_ONLY: ``[thrust]`` in ``[-1, 1]`` -> shape ``(1,)``. Roll/pitch/yaw
            are forced to zero; reduces hover to a vertical-only control problem
            (analogous to ``gym-pybullet-drones``'s ``ONE_D`` action types).
    """

    ATTITUDE = "attitude"
    THRUST_ONLY = "thrust_only"


#: Observation vector length for each :class:`ObservationType`.
OBSERVATION_DIMS: dict[ObservationType, int] = {
    ObservationType.MINIMAL: 13,
    ObservationType.FULL: 16,
}

#: Action vector length for each :class:`ActionType`.
ACTION_DIMS: dict[ActionType, int] = {
    ActionType.ATTITUDE: 4,
    ActionType.THRUST_ONLY: 1,
}
