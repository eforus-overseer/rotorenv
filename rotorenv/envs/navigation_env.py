"""NavigationEnv — fly through a procedurally generated obstacle field to a goal.

This is the "MiniGrid in 3D" task: each episode lays out random box obstacles
(pillars) in an open volume, and the drone must fly from a fixed start to a goal
region without colliding. Obstacle count/density scales with ``self.difficulty``
so the :class:`~rotorenv.envs.curriculum.CurriculumWrapper` can grow the layout
from "empty room" to "dense forest of pillars".

Perception is pluggable (``perception``):

- ``"state"`` (default here): the standard kinematic observation, whose
  ``distance_to_target`` component already points at the goal. Obstacles are
  *not* encoded — this mode is for fast mechanics testing and as a baseline.
- ``"depth"``: an onboard depth-camera image (PEDRA-style). Requires the
  optional ``[render]`` extra and a depth sensor; added in a later step.

The obstacle layout, collision, goal, and reward are pure Python and fast; the
camera is a separate sensor layer, so the expensive perception is isolated.
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

START_POSITION = np.array([-4.5, 0.0, 1.5], dtype=np.float64)
GOAL_POSITION = np.array([4.5, 0.0, 1.5], dtype=np.float64)
GOAL_RADIUS = 0.6
DRONE_RADIUS = 0.2          # collision inflation around the drone point
# Arena bounds (axis-aligned): leaving these terminates the episode.
ARENA_MIN = np.array([-5.5, -3.5, 0.0], dtype=np.float64)
ARENA_MAX = np.array([5.5, 3.5, 4.0], dtype=np.float64)
MAX_TIME = 15.0
SPAWN_ORIENTATION_NOISE = 0.05

MIN_OBSTACLES = 1
MAX_OBSTACLES = 12
# Obstacles are kept out of spheres around start and goal so episodes stay feasible.
CLEARANCE = 1.0


class NavigationEnv(DroneEnv):
    """Procedural obstacle-field navigation from a start to a goal region.

    Args:
        perception: ``"state"`` (kinematic vector, default) or ``"depth"``
            (onboard depth camera; requires the render extra).
        **kwargs: Forwarded to :class:`DroneEnv`.

    Attributes:
        obstacles: ``(N, 6)`` array of ``[cx, cy, cz, hx, hy, hz]`` axis-aligned
            boxes for the current episode.
    """

    #: Depth-image dimensions for ``perception="depth"`` (H, W).
    DEPTH_SHAPE = (64, 64)

    def __init__(self, perception: str = "state", **kwargs) -> None:
        """Store perception mode and initialise the base env."""
        if perception not in ("state", "depth"):
            raise ValueError(
                f"perception must be 'state' or 'depth', got {perception!r}."
            )
        self.perception = perception
        self.obstacles: np.ndarray = np.zeros((0, 6), dtype=np.float64)
        self._depth_camera = None
        self._initialized = False
        super().__init__(**kwargs)
        self._initialized = True

        # In depth mode the observation is an image, not the kinematic vector,
        # so override the Box the base class built (gym-pybullet-drones'
        # KIN-vs-RGB pattern: the space itself depends on perception).
        if self.perception == "depth":
            from gymnasium import spaces

            h, w = self.DEPTH_SHAPE
            # Channel-first (1, H, W) to match stable-baselines3's CNN convention.
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(1, h, w), dtype=np.float32
            )

    def _build_depth_camera(self):
        """Construct a fresh depth camera for the current obstacle layout."""
        from rotorenv.rendering.depth_camera import DepthCamera

        if self._depth_camera is not None:
            self._depth_camera.close()
        h, w = self.DEPTH_SHAPE
        self._depth_camera = DepthCamera(
            self.obstacles, height=h, width=w, max_depth=12.0
        )

    def _get_obs(self, state: DroneState) -> np.ndarray:
        """Return the observation for the active perception mode.

        ``"state"`` defers to the base kinematic vector; ``"depth"`` renders the
        onboard depth image (building the camera lazily on first use).
        """
        if self.perception == "state":
            return super()._get_obs(state)
        if self._depth_camera is None:
            self._build_depth_camera()
        return self._depth_camera.capture(state)

    # ------------------------------------------------------------------ #
    # Task setup.                                                         #
    # ------------------------------------------------------------------ #
    def _make_target(self) -> np.ndarray:
        """Set the goal and (re)generate the obstacle field for this episode.

        In depth mode the camera scene is tied to the obstacle layout, so it is
        rebuilt here whenever a new layout is generated.
        """
        self.obstacles = self._generate_obstacles()
        # Rebuild the camera only once the env is fully constructed (skip the
        # base-class __init__ call, which has no real episode yet).
        if self.perception == "depth" and getattr(self, "_initialized", False):
            self._build_depth_camera()
        return GOAL_POSITION.copy()

    def close(self) -> None:
        """Release the depth camera (if any) and base rendering resources."""
        if self._depth_camera is not None:
            self._depth_camera.close()
            self._depth_camera = None
        super().close()

    def _generate_obstacles(self) -> np.ndarray:
        """Sample difficulty-scaled box obstacles avoiding start/goal clearance.

        Returns:
            ``(N, 6)`` array of ``[cx, cy, cz, hx, hy, hz]``. ``N`` grows with
            ``self.difficulty``. Drawn from ``self.np_random`` (seed-reproducible).
        """
        n = int(round(MIN_OBSTACLES + self.difficulty * (MAX_OBSTACLES - MIN_OBSTACLES)))
        boxes = []
        attempts = 0
        while len(boxes) < n and attempts < n * 20:
            attempts += 1
            cx = self.np_random.uniform(ARENA_MIN[0] + 1.0, ARENA_MAX[0] - 1.0)
            cy = self.np_random.uniform(ARENA_MIN[1] + 0.5, ARENA_MAX[1] - 0.5)
            hx = self.np_random.uniform(0.2, 0.5)
            hy = self.np_random.uniform(0.2, 0.5)
            height = self.np_random.uniform(1.5, ARENA_MAX[2])
            cz, hz = height / 2.0, height / 2.0  # pillar rising from the floor
            center_xy = np.array([cx, cy])
            # Reject if too close to the start or goal (keep a feasible path).
            if np.linalg.norm(center_xy - START_POSITION[:2]) < CLEARANCE + max(hx, hy):
                continue
            if np.linalg.norm(center_xy - GOAL_POSITION[:2]) < CLEARANCE + max(hx, hy):
                continue
            boxes.append([cx, cy, cz, hx, hy, hz])
        return np.array(boxes, dtype=np.float64) if boxes else np.zeros((0, 6))

    def _initial_state(self) -> DroneState:
        """Spawn at the start point with small attitude noise."""
        orientation = self.np_random.uniform(
            -SPAWN_ORIENTATION_NOISE, SPAWN_ORIENTATION_NOISE, size=3
        )
        return DroneState(
            position=START_POSITION.copy(),
            velocity=np.zeros(3),
            orientation=orientation,
            angular_velocity=np.zeros(3),
            time=0.0,
        )

    def _build_reward(self) -> RewardTerm:
        """Reward: big bonus at the goal, distance shaping, energy + collision."""
        return CompositeReward(
            terms=[
                HoverZoneBonus(radius=GOAL_RADIUS, bonus=10.0),
                DistancePenalty(weight=0.2),
                EnergyPenalty(weight=0.01),
                CrashPenalty(penalty=10.0),
            ]
        )

    # ------------------------------------------------------------------ #
    # Dynamics-coupled logic.                                             #
    # ------------------------------------------------------------------ #
    def _in_collision(self, position: np.ndarray) -> bool:
        """Return True if the drone (a sphere of ``DRONE_RADIUS``) hits a box.

        Uses an axis-aligned box test inflated by the drone radius (a cheap,
        slightly conservative sphere-vs-box approximation).
        """
        if self.obstacles.shape[0] == 0:
            return False
        centers = self.obstacles[:, :3]
        halfs = self.obstacles[:, 3:] + DRONE_RADIUS
        inside = np.all(np.abs(position - centers) <= halfs, axis=1)
        return bool(np.any(inside))

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        """Return True if the drone has left the arena volume."""
        return bool(np.any(position < ARENA_MIN) or np.any(position > ARENA_MAX))

    def _is_terminated(self, state: DroneState) -> tuple[bool, bool]:
        """Terminate on goal reached, collision, ground, or out of bounds.

        Returns:
            ``(terminated, crashed)`` — ``crashed`` (collision/ground/oob) fires
            the crash penalty; reaching the goal terminates without it.
        """
        reached = bool(np.linalg.norm(GOAL_POSITION - state.position) < GOAL_RADIUS)
        crashed = bool(
            state.position[2] < 0.0
            or self._in_collision(state.position)
            or self._out_of_bounds(state.position)
        )
        return (reached or crashed), crashed

    def _is_truncated(self, state: DroneState) -> bool:
        """Truncate at the time limit."""
        return state.time > MAX_TIME

    def _get_info(self, state: DroneState):
        """Augment base info with goal/collision diagnostics."""
        info = super()._get_info(state)
        info["reached_goal"] = bool(
            np.linalg.norm(GOAL_POSITION - state.position) < GOAL_RADIUS
        )
        info["n_obstacles"] = int(self.obstacles.shape[0])
        return info
