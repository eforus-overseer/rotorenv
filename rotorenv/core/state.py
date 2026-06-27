"""Core state representation for a drone.

``DroneState`` is the single source of truth for the physical configuration of
the vehicle at one instant. It is deliberately a plain dataclass of numpy arrays
so that physics backends, reward terms, and observation builders can all read
the same well-defined structure without coupling to each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DroneState:
    """Full kinematic state of the drone at a single timestep.

    Attributes:
        position: World-frame position ``(x, y, z)`` in metres, shape ``(3,)``.
        velocity: World-frame linear velocity in m/s, shape ``(3,)``.
        orientation: Euler angles ``(roll, pitch, yaw)`` in radians, shape ``(3,)``.
        angular_velocity: Body angular rates in rad/s, shape ``(3,)``.
        time: Elapsed simulation time in seconds.
    """

    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    time: float

    def __post_init__(self) -> None:
        """Coerce array fields to ``float64`` numpy arrays of shape ``(3,)``."""
        self.position = np.asarray(self.position, dtype=np.float64).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=np.float64).reshape(3)
        self.orientation = np.asarray(self.orientation, dtype=np.float64).reshape(3)
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64).reshape(3)
        self.time = float(self.time)

    def copy(self) -> "DroneState":
        """Return a copy with independent array buffers."""
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            time=self.time,
        )
