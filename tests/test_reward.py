"""Tests for the composable reward terms."""

from __future__ import annotations

import numpy as np
import pytest

from rotorenv.core.action import DroneAction
from rotorenv.core.reward import (
    CompositeReward,
    CrashPenalty,
    DistancePenalty,
    EnergyPenalty,
    HoverZoneBonus,
)
from rotorenv.core.state import DroneState

TARGET = np.array([0.0, 0.0, 1.0])


def _state_at(position: np.ndarray) -> DroneState:
    return DroneState(
        position=position,
        velocity=np.zeros(3),
        orientation=np.zeros(3),
        angular_velocity=np.zeros(3),
        time=0.0,
    )


def _noop_action() -> DroneAction:
    return DroneAction(thrust=0.0, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)


def test_distance_penalty_scales_with_distance() -> None:
    """Distance penalty is -weight * Euclidean distance."""
    term = DistancePenalty(weight=0.5)
    state = _state_at(np.array([0.0, 0.0, 0.0]))  # 1 m below target
    assert term(state, _noop_action(), TARGET, crashed=False) == pytest.approx(-0.5)


def test_hover_zone_bonus_inside_and_outside() -> None:
    """Bonus applies strictly inside the radius and not outside."""
    term = HoverZoneBonus(radius=0.1, bonus=1.0)
    inside = _state_at(np.array([0.0, 0.0, 1.05]))
    outside = _state_at(np.array([0.0, 0.0, 1.5]))
    assert term(inside, _noop_action(), TARGET, crashed=False) == 1.0
    assert term(outside, _noop_action(), TARGET, crashed=False) == 0.0


def test_energy_penalty_quadratic() -> None:
    """Energy penalty is -weight * ||action||^2 over [thrust, roll, pitch, yaw]."""
    term = EnergyPenalty(weight=0.01)
    action = DroneAction(thrust=1.0, roll_cmd=1.0, pitch_cmd=0.0, yaw_cmd=0.0)
    # ||[1,1,0,0]||^2 = 2
    assert term(_state_at(TARGET), action, TARGET, crashed=False) == pytest.approx(-0.02)


def test_crash_penalty_only_on_crash() -> None:
    """Crash penalty fires only when crashed is True."""
    term = CrashPenalty(penalty=5.0)
    assert term(_state_at(TARGET), _noop_action(), TARGET, crashed=True) == -5.0
    assert term(_state_at(TARGET), _noop_action(), TARGET, crashed=False) == 0.0


def test_composite_sums_terms() -> None:
    """CompositeReward equals the sum of its terms."""
    reward = CompositeReward(
        terms=[
            HoverZoneBonus(radius=0.1, bonus=1.0),
            DistancePenalty(weight=0.5),
            EnergyPenalty(weight=0.01),
            CrashPenalty(penalty=5.0),
        ]
    )
    state = _state_at(np.array([0.0, 0.0, 1.0]))  # exactly on target
    action = _noop_action()
    # bonus +1, distance 0, energy 0, no crash -> +1.0
    assert reward(state, action, TARGET, crashed=False) == pytest.approx(1.0)
