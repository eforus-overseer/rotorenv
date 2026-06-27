"""WaypointEnv — fly to a randomly placed target position.

A step up from hovering: the target is *sampled* each episode rather than fixed,
so the agent must learn to translate to an arbitrary point and hold it. Target
spread scales with ``self.difficulty`` (set via ``reset(options=...)``), which is
what the curriculum wrapper anneals — at difficulty 0 the target sits right at
the nominal hover point; at difficulty 1 it can be anywhere in the sampling box.
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

NOMINAL_TARGET = np.array([0.0, 0.0, 1.0], dtype=np.float64)
MAX_HORIZONTAL = 3.0   # m, half-width of the x/y sampling box at difficulty 1
MAX_VERTICAL = 1.5     # m, +/- z spread around the nominal height at difficulty 1
MAX_DISTANCE = 6.0
MAX_TIME = 10.0
SPAWN_ORIENTATION_NOISE = 0.1


class WaypointEnv(DroneEnv):
    """Reach-and-hold a per-episode sampled target, with difficulty scaling."""

    def _make_target(self) -> np.ndarray:
        """Sample a target around the nominal point, scaled by difficulty.

        Drawn from ``self.np_random`` so it is seed-reproducible. The vertical
        component is clamped to stay above ground.
        """
        spread = self.difficulty
        offset = self.np_random.uniform(-1.0, 1.0, size=3) * np.array(
            [MAX_HORIZONTAL, MAX_HORIZONTAL, MAX_VERTICAL]
        )
        target = NOMINAL_TARGET + spread * offset
        target[2] = max(0.25, float(target[2]))   # keep target off the floor
        return target

    def _initial_state(self) -> DroneState:
        """Spawn at the origin with zero velocity and noisy orientation."""
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
        """Distance-driven reach reward with hover-zone bonus and penalties."""
        return CompositeReward(
            terms=[
                HoverZoneBonus(radius=0.2, bonus=1.0),
                DistancePenalty(weight=0.5),
                EnergyPenalty(weight=0.01),
                CrashPenalty(penalty=5.0),
            ]
        )

    def _is_terminated(self, state: DroneState) -> tuple[bool, bool]:
        """Terminate on crash (``z < 0``) or leaving the ``MAX_DISTANCE`` sphere."""
        crashed = bool(state.position[2] < 0.0)
        distance = float(np.linalg.norm(self.target - state.position))
        too_far = distance > MAX_DISTANCE
        return (crashed or too_far), crashed

    def _is_truncated(self, state: DroneState) -> bool:
        """Truncate at the 10 s time limit."""
        return state.time > MAX_TIME
