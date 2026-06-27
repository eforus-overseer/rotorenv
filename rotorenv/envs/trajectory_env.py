"""TrajectoryEnv — track a moving target along a parametric path.

The hardest task in the ladder: the target is not stationary but traces a
Lissajous curve (a generalisation of a circle), and the agent must follow it.
This is the pattern used by reference trajectory-tracking drone envs.

``self.difficulty`` scales how aggressive the trajectory is: at difficulty 0 the
path collapses to the nominal hover point (so it reduces to hovering), and at
difficulty 1 it uses the full radius and speed. The curriculum can therefore
grow a learner from "hold still" to "chase a fast figure-eight".
"""

from __future__ import annotations

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

CENTER = np.array([0.0, 0.0, 1.5], dtype=np.float64)
MAX_RADIUS = np.array([2.0, 2.0, 0.5], dtype=np.float64)   # x, y, z amplitudes
# Lissajous angular frequencies (rad/s); the 2:1 x:y ratio gives a figure-eight.
FREQ = np.array([0.6, 1.2, 0.6], dtype=np.float64)
PHASE = np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64)
MAX_TRACKING_ERROR = 4.0
MAX_TIME = 12.0
SPAWN_ORIENTATION_NOISE = 0.1


class TrajectoryEnv(DroneEnv):
    """Track a moving Lissajous-curve target, with difficulty-scaled aggressiveness."""

    def _target_at(self, t: float) -> np.ndarray:
        """Return the reference position on the Lissajous curve at time ``t``.

        Args:
            t: Elapsed simulation time in seconds.

        Returns:
            The world-frame target position, shape ``(3,)``.
        """
        amplitude = self.difficulty * MAX_RADIUS
        return CENTER + amplitude * np.sin(FREQ * t + PHASE)

    def _make_target(self) -> np.ndarray:
        """Return the trajectory start point (``t = 0``)."""
        return self._target_at(0.0)

    def _update_target(self, state: DroneState) -> None:
        """Advance the moving target to the current simulation time."""
        self.target = self._target_at(state.time)

    def _initial_state(self) -> DroneState:
        """Spawn at the trajectory start point so tracking begins on-path."""
        orientation = self.np_random.uniform(
            -SPAWN_ORIENTATION_NOISE, SPAWN_ORIENTATION_NOISE, size=3
        )
        return DroneState(
            position=self._target_at(0.0),
            velocity=np.zeros(3),
            orientation=orientation,
            angular_velocity=np.zeros(3),
            time=0.0,
        )

    def _build_reward(self) -> RewardTerm:
        """Reward tight tracking; a small zone bonus rewards staying on-path."""
        return CompositeReward(
            terms=[
                HoverZoneBonus(radius=0.3, bonus=1.0),
                DistancePenalty(weight=1.0),
                EnergyPenalty(weight=0.01),
                CrashPenalty(penalty=5.0),
            ]
        )

    def _is_terminated(self, state: DroneState) -> tuple[bool, bool]:
        """Terminate on crash or if tracking error exceeds ``MAX_TRACKING_ERROR``."""
        crashed = bool(state.position[2] < 0.0)
        error = float(np.linalg.norm(self.target - state.position))
        lost = error > MAX_TRACKING_ERROR
        return (crashed or lost), crashed

    def _is_truncated(self, state: DroneState) -> bool:
        """Truncate at the 12 s time limit."""
        return state.time > MAX_TIME
