"""Core domain models: state, action, enums, and composable rewards."""

from rotorenv.core.action import DroneAction
from rotorenv.core.enums import (
    ACTION_DIMS,
    OBSERVATION_DIMS,
    ActionType,
    ObservationType,
)
from rotorenv.core.reward import (
    CompositeReward,
    CrashPenalty,
    DistancePenalty,
    EnergyPenalty,
    HoverZoneBonus,
    ProgressReward,
    RewardTerm,
)
from rotorenv.core.state import DroneState

__all__ = [
    "DroneState",
    "DroneAction",
    "ObservationType",
    "ActionType",
    "OBSERVATION_DIMS",
    "ACTION_DIMS",
    "RewardTerm",
    "CompositeReward",
    "DistancePenalty",
    "HoverZoneBonus",
    "EnergyPenalty",
    "ProgressReward",
    "CrashPenalty",
]
