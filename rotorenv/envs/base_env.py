"""Abstract base environment for drone tasks.

``DroneEnv`` implements all the Gymnasium plumbing shared by every task:
the observation/action spaces, seeded RNG, and the per-step pipeline of
``action -> physics.step -> reward -> termination -> observation``. Concrete
tasks subclass it and implement the four task-specific hooks below; they never
re-implement the step loop or the spaces.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rotorenv.core.action import DroneAction
from rotorenv.core.enums import (
    ACTION_DIMS,
    OBSERVATION_DIMS,
    ActionType,
    ObservationType,
)
from rotorenv.core.reward import RewardTerm
from rotorenv.core.state import DroneState
from rotorenv.physics.base_physics import DronePhysics
from rotorenv.physics.point_mass import PointMassPhysics
from rotorenv.physics.six_dof import SixDOFPhysics


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
        observation_type: Union[ObservationType, str] = ObservationType.FULL,
        action_type: Union[ActionType, str] = ActionType.ATTITUDE,
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
            observation_type: Which fields to expose in the observation. See
                :class:`~rotorenv.core.enums.ObservationType`. Defaults to
                ``FULL`` (16-D, includes angular velocity).
            action_type: Action-space layout. See
                :class:`~rotorenv.core.enums.ActionType`. Defaults to
                ``ATTITUDE`` (4-D ``[thrust, roll, pitch, yaw]``).
            render_mode: Optional Gymnasium render mode (``"human"``).
        """
        super().__init__()
        self.physics: DronePhysics = (
            physics if physics is not None else self._make_physics(physics_model)
        )
        self.observation_type = ObservationType(observation_type)
        self.action_type = ActionType(action_type)
        self.render_mode = render_mode

        act_dim = ACTION_DIMS[self.action_type]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(act_dim,), dtype=np.float32)
        low, high = self._observation_bounds()
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

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
    # Generous physical limits used to give the observation space *finite*
    # bounds (the Gymnasium env-checker flags -inf/+inf as uninformative).
    _POS_LIMIT = 20.0       # m
    _VEL_LIMIT = 50.0       # m/s
    _ANG_LIMIT = np.pi      # rad (wrapped)
    _ANGVEL_LIMIT = 50.0    # rad/s
    _TIME_LIMIT = 1.0e4     # s

    def _observation_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` arrays bounding the observation space.

        Bounds are per-field and finite so that normalisation wrappers and
        algorithms reading ``space.high``/``space.low`` behave. Layout matches
        :meth:`_get_obs` for the active ``observation_type``.
        """
        p, v, a, t = self._POS_LIMIT, self._VEL_LIMIT, self._ANG_LIMIT, self._TIME_LIMIT
        # position(3), velocity(3), orientation(3), distance_to_target(3), time(1)
        high = [p, p, p, v, v, v, a, a, a, 2 * p, 2 * p, 2 * p, t]
        if self.observation_type is ObservationType.FULL:
            high += [self._ANGVEL_LIMIT] * 3   # angular_velocity(3)
        high_arr = np.array(high, dtype=np.float32)
        # Symmetric except time and the position-derived fields stay non-negative
        # only where physical; keeping it symmetric is simplest and valid.
        low_arr = -high_arr.copy()
        low_arr[12] = 0.0   # time is non-negative
        return low_arr, high_arr

    def _get_obs(self, state: DroneState) -> np.ndarray:
        """Flatten a state into the observation vector for ``observation_type``.

        ``MINIMAL`` (13,): ``[position(3), velocity(3), orientation(3),
        distance_to_target(3), time(1)]``.
        ``FULL`` (16,): the above with ``angular_velocity(3)`` appended.
        """
        distance_to_target = self.target - state.position
        fields = [
            state.position,
            state.velocity,
            state.orientation,
            distance_to_target,
            [state.time],
        ]
        if self.observation_type is ObservationType.FULL:
            fields.append(state.angular_velocity)
        return np.concatenate(fields).astype(np.float32)

    def _preprocess_action(self, action: np.ndarray) -> DroneAction:
        """Map a raw policy action to a :class:`DroneAction` per ``action_type``.

        For ``ATTITUDE`` the 4-vector is ``[thrust, roll, pitch, yaw]``. For
        ``THRUST_ONLY`` the 1-vector is ``[thrust]`` and roll/pitch/yaw are held
        at zero. In both cases the thrust channel is rescaled ``[-1,1] -> [0,1]``
        by :meth:`DroneAction.from_array`.

        Args:
            action: Raw action whose length matches the action space.

        Returns:
            The structured command to hand to the physics backend.
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.action_type is ActionType.THRUST_ONLY:
            action = np.array([action[0], 0.0, 0.0, 0.0], dtype=np.float64)
        return DroneAction.from_array(action)

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
            action: Raw policy action in ``[-1, 1]``, length matching the action
                space (4 for ``ATTITUDE``, 1 for ``THRUST_ONLY``).

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        if self.state is None:
            raise RuntimeError("step() called before reset().")

        drone_action = self._preprocess_action(action)
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
