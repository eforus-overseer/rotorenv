"""Lightweight 3D rendering with matplotlib.

Phase-2 renderer: draws the quadrotor as a 4-arm cross that tilts with the
drone's true attitude (so roll/pitch/yaw are visible), plus a fading trajectory
trail and the target marker. matplotlib is imported lazily by callers so that
headless usage never pays for it.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque

import numpy as np

from rotorenv.core.rotations import euler_to_rotmat
from rotorenv.core.state import DroneState

_AXIS_LIMIT = 5.0
_ARM_LENGTH = 0.25          # metres, half-span of each rotor arm (visual scale)
_TRAIL_LENGTH = 200         # number of past positions retained for the trail

# Body-frame endpoints of the four arms (X-config): front-right, back-left, etc.
_ARM_TIPS_BODY = np.array(
    [
        [_ARM_LENGTH, _ARM_LENGTH, 0.0],
        [-_ARM_LENGTH, -_ARM_LENGTH, 0.0],
        [_ARM_LENGTH, -_ARM_LENGTH, 0.0],
        [-_ARM_LENGTH, _ARM_LENGTH, 0.0],
    ],
    dtype=np.float64,
)


class MatplotlibRenderer:
    """Interactive 3D renderer: a tilting quad cross, target, and trajectory trail."""

    def __init__(self) -> None:
        """Create the interactive 3D figure and trajectory buffer."""
        import matplotlib.pyplot as plt  # local import: keep matplotlib optional

        self._plt = plt
        plt.ion()
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._trail: Deque[np.ndarray] = deque(maxlen=_TRAIL_LENGTH)

    def reset(self) -> None:
        """Clear the trajectory trail (call at the start of a new episode)."""
        self._trail.clear()

    def render(self, state: DroneState, target: np.ndarray) -> Any:
        """Draw the quad (attitude-aware), target, and fading trail.

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
        self._trail.append(p.copy())

        # --- Target -------------------------------------------------------- #
        ax.scatter(target[0], target[1], target[2], c="tab:red", marker="x", s=80, label="target")

        # --- Trajectory trail --------------------------------------------- #
        if len(self._trail) >= 2:
            trail = np.array(self._trail)
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], c="tab:gray", lw=1.0, alpha=0.6)

        # --- Quad cross, rotated into the world frame by attitude ---------- #
        rot = euler_to_rotmat(state.orientation)
        tips_world = (rot @ _ARM_TIPS_BODY.T).T + p   # (4, 3)
        # Two arms as lines through the hub (FR-BL and FR(-y)-BL(+y) pairs).
        for i, j in ((0, 1), (2, 3)):
            ax.plot(
                [tips_world[i, 0], tips_world[j, 0]],
                [tips_world[i, 1], tips_world[j, 1]],
                [tips_world[i, 2], tips_world[j, 2]],
                c="tab:blue",
                lw=2.0,
            )
        ax.scatter(tips_world[:, 0], tips_world[:, 1], tips_world[:, 2], c="tab:blue", s=25)
        # Body-up axis (shows which way "up" the drone thinks it is).
        up = rot @ np.array([0.0, 0.0, _ARM_LENGTH])
        ax.plot([p[0], p[0] + up[0]], [p[1], p[1] + up[1]], [p[2], p[2] + up[2]],
                c="tab:green", lw=1.5, label="body-up")
        ax.scatter(p[0], p[1], p[2], c="tab:blue", s=40, label="drone")

        ax.set_title(f"t = {state.time:.2f} s")
        ax.legend(loc="upper right")

        self.fig.canvas.draw_idle()
        self._plt.pause(0.001)
        return None

    def close(self) -> None:
        """Close the figure and release resources."""
        self._plt.close(self.fig)
