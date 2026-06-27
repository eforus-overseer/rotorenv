"""Environments: the abstract base, concrete tasks, and wrappers."""

from rotorenv.envs.base_env import DroneEnv
from rotorenv.envs.curriculum import CurriculumWrapper
from rotorenv.envs.hover_env import HoverEnv
from rotorenv.envs.trajectory_env import TrajectoryEnv
from rotorenv.envs.waypoint_env import WaypointEnv
from rotorenv.envs.wrappers import NormalizeObservation, RewardScale

__all__ = [
    "DroneEnv",
    "HoverEnv",
    "WaypointEnv",
    "TrajectoryEnv",
    "CurriculumWrapper",
    "NormalizeObservation",
    "RewardScale",
]
