"""Lightweight 3D rendering with matplotlib.

This renderer is intentionally minimal for Phase 1: it plots the drone position
and the target in a 3D axes and redraws in place. matplotlib is imported lazily
by callers so that headless usage never pays for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rotorenv.core.state import DroneState

_AXIS_LIMIT = 5.0


class MatplotlibRenderer:
    """Minimal interactive 3D renderer for a single drone and its target."""

    def __init__(self) -> None:
        """Create the interactive 3D figure."""
        import matplotlib.pyplot as plt  # local import: keep matplotlib optional

        self._plt = plt
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")

    def render(self, state: DroneState, target: np.ndarray) -> Any:
        """Draw the drone (blue) and target (red x) for the current state.

        Args:
            state: Current drone state to draw.
            target: World-frame target position, shape ``(3,)``.
        """
        ax = self.ax
        ax.clear()
        ax.set_xlim(-_AXIS_LIMIT, _AXIS_LIMIT)
        ax.set_ylim(-_AXIS_LIMIT, _AXIS_LIMIT)
        ax.set_zlim(0.0, 2 * _AXIS_LIMIT)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        p = state.position
        ax.scatter(p[0], p[1], p[2], c="tab:blue", s=60, label="drone")
        ax.scatter(target[0], target[1], target[2], c="tab:red", marker="x", s=80, label="target")
        ax.set_title(f"t = {state.time:.2f} s")
        ax.legend(loc="upper right")

        self.fig.canvas.draw_idle()
        self._plt.pause(0.001)
        return None

    def close(self) -> None:
        """Close the figure and release resources."""
        self._plt.close(self.fig)
