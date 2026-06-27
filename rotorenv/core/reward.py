"""Composable reward functions for drone tasks.

Rewards are modelled as small callable "terms". Each term maps
``(state, action, target, crashed) -> float``. A :class:`CompositeReward` sums a
list of terms, so a task's reward shaping is expressed as *data* (a list of
configured objects) rather than as a hard-coded function. This keeps reward,
physics, and observation as independent, swappable concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from rotorenv.core.action import DroneAction
from rotorenv.core.state import DroneState


class RewardTerm(Protocol):
    """Interface for a single additive component of a reward function."""

    def __call__(
        self,
        state: DroneState,
        action: DroneAction,
        target: np.ndarray,
        *,
        crashed: bool,
    ) -> float:
        """Return this term's contribution to the reward for one step."""
        ...


@dataclass
class DistancePenalty:
    """Linear penalty proportional to distance from the target position."""

    weight: float = 0.5

    def __call__(
        self, state: DroneState, action: DroneAction, target: np.ndarray, *, crashed: bool
    ) -> float:
        """Return ``-weight * ||target - position||``."""
        distance = float(np.linalg.norm(target - state.position))
        return -self.weight * distance


@dataclass
class HoverZoneBonus:
    """Constant bonus awarded while within ``radius`` metres of the target."""

    radius: float = 0.1
    bonus: float = 1.0

    def __call__(
        self, state: DroneState, action: DroneAction, target: np.ndarray, *, crashed: bool
    ) -> float:
        """Return ``bonus`` inside the hover zone, else ``0``."""
        distance = float(np.linalg.norm(target - state.position))
        return self.bonus if distance < self.radius else 0.0


@dataclass
class EnergyPenalty:
    """Quadratic penalty on control effort, ``weight * ||action||^2``."""

    weight: float = 0.01

    def __call__(
        self, state: DroneState, action: DroneAction, target: np.ndarray, *, crashed: bool
    ) -> float:
        """Return ``-weight * ||action||^2`` over ``[thrust, roll, pitch, yaw]``."""
        a = action.to_array()
        return -self.weight * float(np.dot(a, a))


@dataclass
class CrashPenalty:
    """One-off penalty applied on the terminal crash step."""

    penalty: float = 5.0

    def __call__(
        self, state: DroneState, action: DroneAction, target: np.ndarray, *, crashed: bool
    ) -> float:
        """Return ``-penalty`` if ``crashed`` else ``0``."""
        return -self.penalty if crashed else 0.0


@dataclass
class CompositeReward:
    """A reward function built by summing a sequence of :class:`RewardTerm`.

    Being itself callable with the same signature, a ``CompositeReward`` is a
    ``RewardTerm`` and can be nested inside another composite.
    """

    terms: Sequence[RewardTerm]

    def __call__(
        self, state: DroneState, action: DroneAction, target: np.ndarray, *, crashed: bool
    ) -> float:
        """Return the sum of every term's contribution for this step."""
        return float(
            sum(term(state, action, target, crashed=crashed) for term in self.terms)
        )
