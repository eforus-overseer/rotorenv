"""Abstract base environment for drone tasks.

``DroneEnv`` implements all the Gymnasium plumbing shared by every task:
the observation/action spaces, seeded RNG, and the per-step pipeline of
``action -> physics.step -> reward -> termination -> observation``. Concrete
tasks subclass it and implement the four task-specific hooks below; they never
re-implement the step loop or the spaces.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rotorenv.core.action import DroneAction
from rotorenv.core.reward import RewardTerm
from rotorenv.core.state import DroneState
from rotorenv.physics.base_physics import DronePhysics
from rotorenv.physics.point_mass import PointMassPhysics
from rotorenv.physics.six_dof import SixDOFPhysics

# Observation layout (13,): position(3) velocity(3) orientation(3) distance(3) time(1).
OBS_DIM = 13
ACT_DIM = 4


class DroneEnv(gym.Env):
    """Abstract Gymnasium environment for a single drone.

    Subclasses must implement :meth:`_initial_state`, :meth:`_make_target`,
    :meth:`_build_reward`, and :meth:`_is_terminated`.

    Attributes:
        physics: The pluggable dynamics backend (any :class:`DronePhysics`).
        state: The current :class:`DroneState` (set on :meth:`reset`).
        target: The current world-frame target position, shape ``(3,)``.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        physics: Optional[DronePhysics] = None,
        physics_model: str = "point_mass",
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialise spaces, physics backend, and reward function.

        Args:
            physics: Explicit dynamics backend instance. If given, it takes
                precedence over ``physics_model``.
            physics_model: Name of a built-in backend to construct when
                ``physics`` is not supplied. One of ``"point_mass"`` (Phase 1)
                or ``"six_dof"`` (Phase 2). Strings are used so the backend is
                selectable through the Gymnasium registry.
            render_mode: Optional Gymnasium render mode (``"human"``).
        """
        super().__init__()
        self.physics: DronePhysics = (
            physics if physics is not None else self._make_physics(physics_model)
        )
        self.render_mode = render_mode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        self.reward_fn: RewardTerm = self._build_reward()
        self.state: Optional[DroneState] = None
        self.target: np.ndarray = self._make_target()
        self._renderer: Any = None

    @staticmethod
    def _make_physics(physics_model: str) -> DronePhysics:
        """Construct a built-in physics backend by name.

        Args:
            physics_model: ``"point_mass"`` or ``"six_dof"``.

        Returns:
            A fresh physics backend instance.

        Raises:
            ValueError: If ``physics_model`` is not a known backend name.
        """
        backends = {"point_mass": PointMassPhysics, "six_dof": SixDOFPhysics}
        if physics_model not in backends:
            raise ValueError(
                f"Unknown physics_model {physics_model!r}; "
                f"expected one of {sorted(backends)}."
            )
        return backends[physics_model]()

    # ------------------------------------------------------------------ #
    # Task hooks — subclasses implement these.                            #
    # ------------------------------------------------------------------ #
    def _initial_state(self) -> DroneState:
        """Return the starting :class:`DroneState` for a new episode."""
        raise NotImplementedError

    def _make_target(self) -> np.ndarray:
        """Return the world-frame target position, shape ``(3,)``."""
        raise NotImplementedError

    def _build_reward(self) -> RewardTerm:
        """Return the (composable) reward function for this task."""
        raise NotImplementedError

    def _is_terminated(self, state: DroneState) -> tuple[bool, bool]:
        """Return ``(terminated, crashed)`` for the given state.

        ``terminated`` is the MDP terminal flag; ``crashed`` selects whether the
        crash penalty applies this step.
        """
        raise NotImplementedError

    def _is_truncated(self, state: DroneState) -> bool:
        """Return whether the episode hit a non-terminal time/space limit."""
        return False

    # ------------------------------------------------------------------ #
    # Gymnasium API.                                                      #
    # ------------------------------------------------------------------ #
    def _get_obs(self, state: DroneState) -> np.ndarray:
        """Flatten a state into the 13-D observation vector.

        Layout: ``[position(3), velocity(3), orientation(3),
        distance_to_target(3), time(1)]``. Note angular velocity is *not*
        included, by design, to hit the specified ``(13,)`` shape.
        """
        distance_to_target = self.target - state.position
        obs = np.concatenate(
            [
                state.position,
                state.velocity,
                state.orientation,
                distance_to_target,
                [state.time],
            ]
        )
        return obs.astype(np.float32)

    def _get_info(self, state: DroneState) -> dict[str, Any]:
        """Return diagnostic info (Euclidean distance to target)."""
        return {"distance": float(np.linalg.norm(self.target - state.position))}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode and return ``(observation, info)``.

        Seeding goes through ``gym.Env.reset`` which populates
        ``self.np_random`` (a ``numpy`` ``Generator``); subclasses must draw all
        randomness from it rather than the global numpy RNG.
        """
        super().reset(seed=seed)
        self.target = self._make_target()
        self.state = self._initial_state()
        if self._renderer is not None:
            self._renderer.reset()
        return self._get_obs(self.state), self._get_info(self.state)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one action and return the Gymnasium 5-tuple.

        Args:
            action: Raw policy action in ``[-1, 1]^4`` (``[thrust, roll, pitch, yaw]``).

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        if self.state is None:
            raise RuntimeError("step() called before reset().")

        drone_action = DroneAction.from_array(np.asarray(action, dtype=np.float64))
        self.state = self.physics.step(self.state, drone_action)

        terminated, crashed = self._is_terminated(self.state)
        truncated = self._is_truncated(self.state)
        reward = float(
            self.reward_fn(self.state, drone_action, self.target, crashed=crashed)
        )

        return (
            self._get_obs(self.state),
            reward,
            terminated,
            truncated,
            self._get_info(self.state),
        )

    def render(self) -> Any:
        """Render the current state via the matplotlib renderer (lazy import)."""
        if self.render_mode != "human" or self.state is None:
            return None
        if self._renderer is None:
            from rotorenv.rendering.matplotlib_renderer import MatplotlibRenderer

            self._renderer = MatplotlibRenderer()
        return self._renderer.render(self.state, self.target)

    def close(self) -> None:
        """Release any rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
