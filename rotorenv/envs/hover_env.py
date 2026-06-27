"""HoverEnv — hold position at a fixed target point.

The agent must reach and hold ``(0, 0, 1.0)``. The episode terminates on a crash
(``z < 0``) or on flying too far (``distance > 5 m``), and truncates at the 10 s
time limit. Reward shaping is assembled from the composable terms in
:mod:`rotorenv.core.reward`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rotorenv.core.reward import (
    CompositeReward,
    CrashPenalty,
    DistancePenalty,
    EnergyPenalty,
    HoverZoneBonus,
    RewardTerm,
)
from rotorenv.core.state import DroneState
from rotorenv.envs.base_env import DroneEnv

TARGET_POSITION = np.array([0.0, 0.0, 1.0], dtype=np.float64)
MAX_DISTANCE = 5.0
MAX_TIME = 10.0
SPAWN_ORIENTATION_NOISE = 0.1  # radians, uniform +/- per axis


class HoverEnv(DroneEnv):
    """Single-target hover task built on :class:`DroneEnv`.

    The drone spawns at the origin with zero velocity and a small random
    orientation perturbation, and is rewarded for holding the target point.
    """

    def _make_target(self) -> np.ndarray:
        """Return the fixed hover target ``(0, 0, 1.0)``."""
        return TARGET_POSITION.copy()

    def _initial_state(self) -> DroneState:
        """Spawn at the origin with zero velocity and noisy orientation.

        Orientation noise is drawn from ``self.np_random`` (seeded via
        ``reset(seed=...)``), so episodes are reproducible.
        """
        orientation = self.np_random.uniform(
            -SPAWN_ORIENTATION_NOISE, SPAWN_ORIENTATION_NOISE, size=3
        )
        return DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=orientation,
            angular_velocity=np.zeros(3),
            time=0.0,
        )

    def _build_reward(self) -> RewardTerm:
        """Compose the hover reward from distance, hover-zone, energy, crash terms."""
        return CompositeReward(
            terms=[
                HoverZoneBonus(radius=0.1, bonus=1.0),
                DistancePenalty(weight=0.5),
                EnergyPenalty(weight=0.01),
                CrashPenalty(penalty=5.0),
            ]
        )

    def _is_terminated(self, state: DroneState) -> tuple[bool, bool]:
        """Terminate on crash (``z < 0``) or leaving the ``5 m`` sphere.

        Returns:
            ``(terminated, crashed)`` — ``crashed`` is true only for the
            ground-collision case, which is what triggers the crash penalty.
        """
        crashed = bool(state.position[2] < 0.0)
        distance = float(np.linalg.norm(self.target - state.position))
        too_far = distance > MAX_DISTANCE
        return (crashed or too_far), crashed

    def _is_truncated(self, state: DroneState) -> bool:
        """Truncate once elapsed time exceeds the ``10 s`` limit (500 steps)."""
        return state.time > MAX_TIME
