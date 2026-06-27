"""Core action representation for a drone.

``DroneAction`` is the semantic, named form of a control command. Policies emit
a raw 4-vector in ``[-1, 1]`` (the Gym action space); ``from_array`` converts it
into this structured form, rescaling the thrust channel so that a neutral
command of ``0`` corresponds to a 50% throttle hover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DroneAction:
    """A single control command for the drone.

    Attributes:
        thrust: Normalised collective thrust in ``[0, 1]`` (0 = off, 1 = max).
        roll_cmd: Roll command in ``[-1, 1]``.
        pitch_cmd: Pitch command in ``[-1, 1]``.
        yaw_cmd: Yaw command in ``[-1, 1]``.
    """

    thrust: float
    roll_cmd: float
    pitch_cmd: float
    yaw_cmd: float

    @classmethod
    def from_array(cls, raw: np.ndarray) -> "DroneAction":
        """Build a ``DroneAction`` from a raw policy vector in ``[-1, 1]^4``.

        The vector is ``[thrust, roll, pitch, yaw]``. The thrust channel is
        rescaled from ``[-1, 1]`` to ``[0, 1]`` (so neutral 0 -> 50% throttle);
        all channels are clipped to their valid ranges.

        Args:
            raw: Array-like of length 4 in ``[-1, 1]``.

        Returns:
            The corresponding structured ``DroneAction``.
        """
        raw = np.asarray(raw, dtype=np.float64).reshape(4)
        thrust = float(np.clip((raw[0] + 1.0) / 2.0, 0.0, 1.0))
        roll = float(np.clip(raw[1], -1.0, 1.0))
        pitch = float(np.clip(raw[2], -1.0, 1.0))
        yaw = float(np.clip(raw[3], -1.0, 1.0))
        return cls(thrust=thrust, roll_cmd=roll, pitch_cmd=pitch, yaw_cmd=yaw)

    def to_array(self) -> np.ndarray:
        """Return ``[thrust, roll, pitch, yaw]`` as a ``float64`` array.

        Note the thrust channel is in its semantic ``[0, 1]`` range here, which
        is what the energy reward term operates on.
        """
        return np.array(
            [self.thrust, self.roll_cmd, self.pitch_cmd, self.yaw_cmd],
            dtype=np.float64,
        )
