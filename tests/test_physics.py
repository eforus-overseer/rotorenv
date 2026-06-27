"""Tests for the point-mass physics backend."""

from __future__ import annotations

import numpy as np
import pytest

from rotorenv.core.action import DroneAction
from rotorenv.core.state import DroneState
from rotorenv.physics.base_physics import DronePhysics
from rotorenv.physics.point_mass import PointMassPhysics


def _zero_state() -> DroneState:
    return DroneState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=np.zeros(3),
        angular_velocity=np.zeros(3),
        time=0.0,
    )


def test_satisfies_protocol() -> None:
    """PointMassPhysics structurally satisfies the DronePhysics protocol."""
    assert isinstance(PointMassPhysics(), DronePhysics)


def test_hover_thrust_holds_altitude() -> None:
    """At 50% throttle and level attitude, vertical acceleration is ~zero."""
    physics = PointMassPhysics()
    action = DroneAction(thrust=0.5, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(_zero_state(), action)
    assert next_state.velocity[2] == pytest.approx(0.0, abs=1e-9)
    assert next_state.position[2] == pytest.approx(0.0, abs=1e-9)


def test_zero_thrust_falls_under_gravity() -> None:
    """With no thrust, the drone accelerates downward at -g."""
    physics = PointMassPhysics()
    action = DroneAction(thrust=0.0, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(_zero_state(), action)
    assert next_state.velocity[2] == pytest.approx(-physics.gravity * physics.dt)


def test_full_thrust_accelerates_up() -> None:
    """At full throttle the net upward acceleration is +g (max_thrust = 2mg)."""
    physics = PointMassPhysics()
    action = DroneAction(thrust=1.0, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(_zero_state(), action)
    assert next_state.velocity[2] == pytest.approx(physics.gravity * physics.dt)


def test_step_does_not_mutate_input() -> None:
    """step() must return a new state without mutating the input."""
    physics = PointMassPhysics()
    state = _zero_state()
    physics.step(state, DroneAction(thrust=1.0, roll_cmd=0.5, pitch_cmd=0.0, yaw_cmd=0.0))
    assert np.all(state.position == 0.0)
    assert np.all(state.velocity == 0.0)
    assert state.time == 0.0


def test_yaw_command_sets_angular_velocity() -> None:
    """Yaw command maps to angular velocity via max_tilt_rate."""
    physics = PointMassPhysics(max_tilt_rate=2.0)
    action = DroneAction(thrust=0.5, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=1.0)
    next_state = physics.step(_zero_state(), action)
    assert next_state.angular_velocity[2] == pytest.approx(2.0)


def test_max_thrust_value() -> None:
    """max_thrust equals 2*m*g so hover sits at 50% throttle."""
    physics = PointMassPhysics(mass=0.5, gravity=9.81)
    assert physics.max_thrust == pytest.approx(2.0 * 0.5 * 9.81)
