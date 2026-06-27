"""Curriculum-learning wrapper.

Difficulty progression is a property of *training*, not of the MDP, so it lives
in a wrapper that drives the env through ``reset(options={"difficulty": d})``
rather than inside the env. Two schedules are supported (selectable via
``mode``), reflecting the two mechanisms seen across reference drone-RL projects:

- ``"success"``: performance-staged — difficulty rises when the recent success
  rate clears a threshold and falls when it collapses (QuadCtrl-style staging).
- ``"step"``: step-annealed — difficulty ramps linearly from ``start_difficulty``
  to 1.0 over ``anneal_steps`` environment steps, regardless of performance
  (quad-swarm-rl's ``anneal_*_steps`` pattern).

The wrapped env must accept ``reset(options={"difficulty": float})`` and expose a
``"distance"`` entry in its ``info`` dict (rotorenv envs do both).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Optional

import gymnasium as gym


class CurriculumWrapper(gym.Wrapper):
    """Anneal task difficulty across episodes by success rate or step count.

    Attributes:
        difficulty: Current difficulty in ``[0, 1]`` passed to the env on reset.
    """

    def __init__(
        self,
        env: gym.Env,
        mode: str = "success",
        start_difficulty: float = 0.1,
        difficulty_step: float = 0.1,
        success_distance: float = 0.3,
        success_threshold: float = 0.7,
        failure_threshold: float = 0.2,
        window: int = 20,
        anneal_steps: int = 200_000,
    ) -> None:
        """Configure the curriculum schedule.

        Args:
            env: Environment to wrap (must accept ``difficulty`` reset option).
            mode: ``"success"`` (performance-staged) or ``"step"`` (step-annealed).
            start_difficulty: Initial difficulty in ``[0, 1]``.
            difficulty_step: Increment/decrement applied per stage (success mode).
            success_distance: Final distance below which an episode counts as a
                success.
            success_threshold: Success rate (over ``window``) above which
                difficulty increases.
            failure_threshold: Success rate below which difficulty decreases.
            window: Number of recent episodes used to estimate success rate.
            anneal_steps: Steps over which difficulty ramps to 1.0 (step mode).

        Raises:
            ValueError: If ``mode`` is not ``"success"`` or ``"step"``.
        """
        super().__init__(env)
        if mode not in ("success", "step"):
            raise ValueError(f"mode must be 'success' or 'step', got {mode!r}.")
        self.mode = mode
        self.difficulty = float(start_difficulty)
        self._start_difficulty = float(start_difficulty)
        self.difficulty_step = float(difficulty_step)
        self.success_distance = float(success_distance)
        self.success_threshold = float(success_threshold)
        self.failure_threshold = float(failure_threshold)
        self.anneal_steps = int(anneal_steps)

        self._results: Deque[bool] = deque(maxlen=int(window))
        self._total_steps = 0
        self._last_distance = float("inf")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the env at the current curriculum difficulty.

        Any caller-provided ``options`` are preserved; we only inject/override
        the ``difficulty`` key.
        """
        merged: dict[str, Any] = dict(options) if options else {}
        merged["difficulty"] = self.difficulty
        obs, info = self.env.reset(seed=seed, options=merged)
        info["difficulty"] = self.difficulty
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the env, track progress, and advance the curriculum on episode end."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1
        if "distance" in info:
            self._last_distance = float(info["distance"])

        if self.mode == "step":
            frac = min(1.0, self._total_steps / max(1, self.anneal_steps))
            self.difficulty = self._start_difficulty + frac * (1.0 - self._start_difficulty)
        elif terminated or truncated:
            self._record_episode_and_maybe_advance()

        info["difficulty"] = self.difficulty
        return obs, reward, terminated, truncated, info

    def _record_episode_and_maybe_advance(self) -> None:
        """Record success/failure and adjust difficulty (success mode only)."""
        succeeded = self._last_distance < self.success_distance
        self._results.append(succeeded)
        if len(self._results) < self._results.maxlen:
            return
        rate = sum(self._results) / len(self._results)
        if rate >= self.success_threshold and self.difficulty < 1.0:
            self.difficulty = min(1.0, self.difficulty + self.difficulty_step)
            self._results.clear()
        elif rate <= self.failure_threshold and self.difficulty > 0.0:
            self.difficulty = max(0.0, self.difficulty - self.difficulty_step)
            self._results.clear()
