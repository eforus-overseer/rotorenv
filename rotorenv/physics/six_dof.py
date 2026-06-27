"""Phase-2 full 6-DOF rigid-body quadrotor physics.

This backend models the drone as a rigid body with mass and a diagonal inertia
tensor. The high-level action ``[thrust, roll, pitch, yaw]`` is interpreted as a
collective-thrust setpoint plus desired body torques (so Phase-1 policies remain
valid), which are realised through a virtual cross-frame motor mix.

Dynamics per step (semi-implicit Euler):

    Translation (world frame):
        F = R(q) @ (T * z_body) - m*g*z_world - k_drag * v
        v <- v + (F / m) * dt
        p <- p + v * dt

    Rotation (body frame, Euler's equations):
        tau = desired torque - k_ang * omega   (linear angular drag)
        omega_dot = I^-1 @ (tau - omega x (I @ omega))
        omega <- omega + omega_dot * dt
        q     <- normalize(q + 0.5 * (q ⊗ [0, omega]) * dt)

Attitude is integrated as a quaternion internally (no gimbal lock) and converted
back to Euler angles at the :class:`DroneState` boundary to honour the fixed
domain-model contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rotorenv.core.action import DroneAction
from rotorenv.core.rotations import (
    euler_to_quat,
    normalize_quat,
    quat_mul,
    quat_to_euler,
    quat_to_rotmat,
)
from rotorenv.core.state import DroneState

_Z_HAT = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class SixDOFPhysics:
    """Full rigid-body quadrotor dynamics with a diagonal inertia tensor.

    Attributes:
        mass: Vehicle mass in kg.
        gravity: Gravitational acceleration in m/s^2.
        dt: Integration timestep in seconds.
        inertia: Diagonal of the body inertia tensor ``(Ixx, Iyy, Izz)`` in kg·m^2.
        max_torque: Peak body torque ``(tau_x, tau_y, tau_z)`` at full stick, N·m.
        linear_drag: Linear aerodynamic drag coefficient (world frame), N·s/m.
        angular_drag: Angular drag coefficient (body frame), N·m·s/rad.
    """

    mass: float = 0.5
    gravity: float = 9.81
    dt: float = 0.02
    inertia: np.ndarray = field(
        default_factory=lambda: np.array([3.2e-3, 3.2e-3, 5.5e-3], dtype=np.float64)
    )
    max_torque: np.ndarray = field(
        default_factory=lambda: np.array([0.10, 0.10, 0.05], dtype=np.float64)
    )
    linear_drag: float = 0.05
    angular_drag: float = 2.0e-3

    def __post_init__(self) -> None:
        """Coerce inertia/torque arrays to float64 of shape ``(3,)``."""
        self.inertia = np.asarray(self.inertia, dtype=np.float64).reshape(3)
        self.max_torque = np.asarray(self.max_torque, dtype=np.float64).reshape(3)

    @property
    def max_thrust(self) -> float:
        """Maximum collective thrust force in newtons (hover at 50% throttle)."""
        return 2.0 * self.mass * self.gravity

    def step(self, state: DroneState, action: DroneAction) -> DroneState:
        """Advance the rigid-body dynamics by one timestep.

        Args:
            state: Current drone state.
            action: High-level command; ``thrust`` in ``[0, 1]`` sets collective
                thrust, and ``roll/pitch/yaw_cmd`` in ``[-1, 1]`` set desired body
                torques scaled by :attr:`max_torque`.

        Returns:
            A new :class:`DroneState` advanced by ``self.dt``.
        """
        dt = self.dt
        q = euler_to_quat(state.orientation)
        omega = state.angular_velocity.astype(np.float64)

        # --- Rotational dynamics (body frame) ---------------------------- #
        desired_torque = self.max_torque * np.array(
            [action.roll_cmd, action.pitch_cmd, action.yaw_cmd], dtype=np.float64
        )
        torque = desired_torque - self.angular_drag * omega
        inertia = self.inertia
        gyroscopic = np.cross(omega, inertia * omega)
        omega_dot = (torque - gyroscopic) / inertia
        omega = omega + omega_dot * dt

        # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, omega]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
        q = normalize_quat(q + 0.5 * quat_mul(q, omega_quat) * dt)

        # --- Translational dynamics (world frame) ------------------------ #
        thrust_force = action.thrust * self.max_thrust
        up_world = quat_to_rotmat(q) @ _Z_HAT
        f_net = (
            thrust_force * up_world
            - self.mass * self.gravity * _Z_HAT
            - self.linear_drag * state.velocity
        )
        acceleration = f_net / self.mass
        velocity = state.velocity + acceleration * dt          # semi-implicit Euler
        position = state.position + velocity * dt

        return DroneState(
            position=position,
            velocity=velocity,
            orientation=quat_to_euler(q),
            angular_velocity=omega,
            time=state.time + dt,
        )
