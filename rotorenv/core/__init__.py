"""Core domain models: state, action, and composable rewards."""

from rotorenv.core.action import DroneAction
from rotorenv.core.reward import (
    CompositeReward,
    CrashPenalty,
    DistancePenalty,
    EnergyPenalty,
    HoverZoneBonus,
    RewardTerm,
)
from rotorenv.core.state import DroneState

__all__ = [
    "DroneState",
    "DroneAction",
    "RewardTerm",
    "CompositeReward",
    "DistancePenalty",
    "HoverZoneBonus",
    "EnergyPenalty",
    "CrashPenalty",
]
