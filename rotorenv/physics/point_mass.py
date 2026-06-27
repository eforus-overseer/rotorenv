"""Phase-1 point-mass physics.

A deliberately naive model: the drone is a point mass with a thrust vector that
can be tilted by its orientation. There is no aerodynamic drag, no rotor lag,
and no inertia matrix. Orientation is driven directly by the roll/pitch/yaw
commands (command -> angular velocity -> Euler integration), rather than by a
full rigid-body torque model. This is enough to make hovering a non-trivial
control problem while keeping the maths transparent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rotorenv.core.action import DroneAction
from rotorenv.core.rotations import euler_to_rotmat
from rotorenv.core.state import DroneState

_Z_HAT = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class PointMassPhysics:
    """Simple point-mass quadrotor dynamics integrated at a fixed timestep.

    Attributes:
        mass: Vehicle mass in kg.
        gravity: Gravitational acceleration in m/s^2.
        dt: Integration timestep in seconds.
        max_tilt_rate: Max angular rate (rad/s) commanded at full stick deflection.
    """

    mass: float = 0.5
    gravity: float = 9.81
    dt: float = 0.02
    max_tilt_rate: float = 2.0

    @property
    def max_thrust(self) -> float:
        """Maximum collective thrust force in newtons (hover at 50% throttle)."""
        return 2.0 * self.mass * self.gravity

    def step(self, state: DroneState, action: DroneAction) -> DroneState:
        """Advance the point-mass dynamics by one timestep.

        The thrust magnitude scales linearly with ``action.thrust`` up to
        :attr:`max_thrust`, and is applied along the body up-axis rotated into
        the world frame. Gravity is a constant world-frame downward force.
        Orientation is updated by treating the stick commands as angular-rate
        setpoints and integrating once (semi-implicit Euler for translation).

        Args:
            state: Current drone state.
            action: Control command to apply.

        Returns:
            A new :class:`DroneState` advanced by ``self.dt``.
        """
        dt = self.dt

        # --- Angular update: commands set angular velocity, then integrate. ---
        angular_velocity = self.max_tilt_rate * np.array(
            [action.roll_cmd, action.pitch_cmd, action.yaw_cmd], dtype=np.float64
        )
        orientation = state.orientation + angular_velocity * dt
        # Wrap yaw into [-pi, pi]; clamp roll/pitch to avoid the gimbal/flip regime.
        orientation[2] = (orientation[2] + np.pi) % (2.0 * np.pi) - np.pi
        orientation[0] = float(np.clip(orientation[0], -np.pi / 2.0, np.pi / 2.0))
        orientation[1] = float(np.clip(orientation[1], -np.pi / 2.0, np.pi / 2.0))

        # --- Translational update: F = thrust(world) - m*g*z_hat ---
        thrust_force = action.thrust * self.max_thrust
        up_world = euler_to_rotmat(orientation) @ _Z_HAT
        f_net = thrust_force * up_world - self.mass * self.gravity * _Z_HAT
        acceleration = f_net / self.mass

        velocity = state.velocity + acceleration * dt          # semi-implicit Euler
        position = state.position + velocity * dt

        return DroneState(
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            time=state.time + dt,
        )
