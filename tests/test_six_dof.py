"""Tests for the 6-DOF rigid-body physics backend."""

from __future__ import annotations

import numpy as np
import pytest

from rotorenv.core.action import DroneAction
from rotorenv.core.state import DroneState
from rotorenv.physics.base_physics import DronePhysics
from rotorenv.physics.six_dof import SixDOFPhysics


def _level_state() -> DroneState:
    return DroneState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=np.zeros(3),
        angular_velocity=np.zeros(3),
        time=0.0,
    )


def test_satisfies_protocol() -> None:
    """SixDOFPhysics structurally satisfies the DronePhysics protocol."""
    assert isinstance(SixDOFPhysics(), DronePhysics)


def test_hover_thrust_holds_altitude_when_level() -> None:
    """At 50% throttle, level and no drag motion, vertical velocity stays ~zero."""
    physics = SixDOFPhysics(linear_drag=0.0)
    action = DroneAction(thrust=0.5, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(_level_state(), action)
    assert next_state.velocity[2] == pytest.approx(0.0, abs=1e-9)


def test_zero_thrust_falls() -> None:
    """No thrust and no drag yields downward acceleration of -g."""
    physics = SixDOFPhysics(linear_drag=0.0)
    action = DroneAction(thrust=0.0, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(_level_state(), action)
    assert next_state.velocity[2] == pytest.approx(-physics.gravity * physics.dt)


def test_roll_torque_induces_roll_rate() -> None:
    """A positive roll command spins up a positive roll angular velocity."""
    physics = SixDOFPhysics()
    action = DroneAction(thrust=0.5, roll_cmd=1.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(_level_state(), action)
    assert next_state.angular_velocity[0] > 0.0
    assert next_state.orientation[0] > 0.0   # integrated into a roll angle


def test_yaw_command_only_affects_yaw_axis() -> None:
    """A pure yaw command leaves roll/pitch rates at zero."""
    physics = SixDOFPhysics()
    action = DroneAction(thrust=0.5, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=1.0)
    next_state = physics.step(_level_state(), action)
    assert next_state.angular_velocity[0] == pytest.approx(0.0, abs=1e-12)
    assert next_state.angular_velocity[1] == pytest.approx(0.0, abs=1e-12)
    assert next_state.angular_velocity[2] > 0.0


def test_step_does_not_mutate_input() -> None:
    """step() returns a new state without mutating the input."""
    physics = SixDOFPhysics()
    state = _level_state()
    physics.step(state, DroneAction(thrust=1.0, roll_cmd=1.0, pitch_cmd=0.5, yaw_cmd=0.0))
    assert np.all(state.position == 0.0)
    assert np.all(state.angular_velocity == 0.0)
    assert state.time == 0.0


def test_tilted_thrust_has_horizontal_component() -> None:
    """When rolled, thrust gains a horizontal component (drone translates)."""
    physics = SixDOFPhysics(linear_drag=0.0)
    rolled = DroneState(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=np.array([0.5, 0.0, 0.0]),  # 0.5 rad roll
        angular_velocity=np.zeros(3),
        time=0.0,
    )
    action = DroneAction(thrust=1.0, roll_cmd=0.0, pitch_cmd=0.0, yaw_cmd=0.0)
    next_state = physics.step(rolled, action)
    # A roll tilts thrust toward -y in this convention; the key point is nonzero
    # horizontal motion appears.
    assert np.linalg.norm(next_state.velocity[:2]) > 1e-3


def test_orientation_stays_finite_over_long_rollout() -> None:
    """Quaternion integration keeps orientation bounded and finite."""
    physics = SixDOFPhysics()
    state = _level_state()
    action = DroneAction(thrust=0.5, roll_cmd=1.0, pitch_cmd=-1.0, yaw_cmd=1.0)
    for _ in range(500):
        state = physics.step(state, action)
    assert np.all(np.isfinite(state.orientation))
    assert np.all(np.abs(state.orientation) <= np.pi + 1e-6)
