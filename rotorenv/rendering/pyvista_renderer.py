"""Cinematic 3D rendering with PyVista (VTK), for saved animations.

Unlike the lightweight :mod:`~rotorenv.rendering.matplotlib_renderer` (a fixed
schematic camera, zero extra deps), this renderer uses a real 3D scene engine to
produce game-like flythrough video. It supports moving cameras — third-person
**chase**, **pov** (cockpit), and a static **orbit** — and is designed for
off-screen rendering to an MP4/GIF.

PyVista/VTK is an optional dependency (the ``[render]`` extra); this module is
imported lazily so the core env never requires it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rotorenv.core.rotations import euler_to_rotmat
from rotorenv.core.state import DroneState

# Body-frame arm endpoints (X-config), metres — the visual drone scale.
_ARM = 0.18
_ARM_TIPS_BODY = np.array(
    [
        [_ARM, _ARM, 0.0],
        [-_ARM, -_ARM, 0.0],
        [_ARM, -_ARM, 0.0],
        [-_ARM, _ARM, 0.0],
    ],
    dtype=np.float64,
)


class PyVistaRenderer:
    """Off-screen PyVista renderer producing camera-tracked flight animations.

    Attributes:
        camera_mode: One of ``"chase"``, ``"pov"``, ``"orbit"``.
    """

    def __init__(
        self,
        camera_mode: str = "chase",
        window_size: tuple[int, int] = (960, 720),
    ) -> None:
        """Create an off-screen plotter with a ground plane and lighting.

        Args:
            camera_mode: Camera behaviour — ``"chase"`` (third-person follow),
                ``"pov"`` (onboard, looking forward), or ``"orbit"`` (static
                wide shot the drone flies within).
            window_size: Render resolution ``(width, height)`` in pixels.

        Raises:
            ValueError: If ``camera_mode`` is not recognised.
        """
        if camera_mode not in ("chase", "pov", "orbit"):
            raise ValueError(
                f"camera_mode must be 'chase', 'pov', or 'orbit', got {camera_mode!r}."
            )
        import pyvista as pv  # local import: keep PyVista optional

        self._pv = pv
        self.camera_mode = camera_mode
        self.plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        self.plotter.set_background("lightskyblue", top="white")

        # Ground plane + a faint grid for depth perception.
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=20, j_size=20)
        self.plotter.add_mesh(ground, color="darkseagreen", opacity=0.6)
        self.plotter.add_mesh(
            pv.Plane(center=(0, 0, 0.001), direction=(0, 0, 1), i_size=20, j_size=20,
                     i_resolution=20, j_resolution=20),
            style="wireframe", color="gray", opacity=0.25, line_width=1,
        )
        self._drone_actors: list = []
        self._target_actor = None
        self._trail_points: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    def _add_drone(self, state: DroneState) -> list:
        """Add a richer quad model (hub + arms + rotor disks) to the scene.

        Returns the list of actors added, so the caller can remove them before
        the next frame. The front-right rotor is coloured distinctly so heading
        is readable.
        """
        pv = self._pv
        rot = euler_to_rotmat(state.orientation)
        p = state.position
        tips = (rot @ _ARM_TIPS_BODY.T).T + p
        disk_normal = rot @ np.array([0.0, 0.0, 1.0])   # rotor plane normal
        actors = []

        # Arms (two crossing tubes).
        for i, j in ((0, 1), (2, 3)):
            actors.append(
                self.plotter.add_mesh(
                    pv.Line(tips[i], tips[j]), color="black",
                    line_width=4, render_lines_as_tubes=True,
                )
            )
        # Center hub.
        actors.append(
            self.plotter.add_mesh(pv.Sphere(radius=0.05, center=tuple(p)), color="dimgray")
        )
        # Rotor disks at each arm tip; first one (front-right) highlighted.
        rotor_colors = ["orangered", "navy", "navy", "navy"]
        for tip, col in zip(tips, rotor_colors):
            disk = pv.Disc(center=tuple(tip), normal=tuple(disk_normal),
                           inner=0.0, outer=0.08, r_res=1, c_res=24)
            actors.append(self.plotter.add_mesh(disk, color=col, opacity=0.85))
        return actors

    def add_obstacles(self, obstacles: np.ndarray) -> None:
        """Add static box obstacles to the scene (call once per episode).

        Args:
            obstacles: ``(N, 6)`` array of ``[cx, cy, cz, hx, hy, hz]``
                axis-aligned boxes (centre + half-extents).
        """
        pv = self._pv
        for box in np.asarray(obstacles, dtype=np.float64):
            c, h = box[:3], box[3:]
            cube = pv.Cube(center=tuple(c), x_length=2 * h[0],
                           y_length=2 * h[1], z_length=2 * h[2])
            self.plotter.add_mesh(cube, color="slategray", opacity=0.9)

    def _place_camera(self, state: DroneState) -> None:
        """Position the camera according to ``camera_mode`` for this frame."""
        rot = euler_to_rotmat(state.orientation)
        forward = rot @ np.array([1.0, 0.0, 0.0])   # body +x in world
        up = rot @ np.array([0.0, 0.0, 1.0])
        p = state.position

        if self.camera_mode == "chase":
            cam_pos = p - 1.6 * forward + 0.6 * np.array([0, 0, 1])
            self.plotter.camera.position = tuple(cam_pos)
            self.plotter.camera.focal_point = tuple(p + 0.3 * forward)
            self.plotter.camera.up = (0, 0, 1)
        elif self.camera_mode == "pov":
            cam_pos = p + 0.05 * forward + 0.05 * up
            self.plotter.camera.position = tuple(cam_pos)
            self.plotter.camera.focal_point = tuple(p + 2.0 * forward)
            self.plotter.camera.up = tuple(up)
        else:  # orbit — static wide shot
            self.plotter.camera.position = (6.0, 6.0, 5.0)
            self.plotter.camera.focal_point = (0.0, 0.0, 1.0)
            self.plotter.camera.up = (0, 0, 1)

    def render_frame(self, state: DroneState, target: np.ndarray) -> None:
        """Update drone, trail, target, and camera for the current state."""
        pv = self._pv
        # Refresh drone (remove last frame's actors, add new ones).
        for actor in self._drone_actors:
            self.plotter.remove_actor(actor)
        self._drone_actors = self._add_drone(state)

        # Target: translucent wireframe sphere so the drone shows through when
        # it sits right on the target (a tight hover would otherwise be hidden).
        if self._target_actor is None:
            self._target_actor = self.plotter.add_mesh(
                pv.Sphere(radius=0.18, center=tuple(target)),
                color="red", style="wireframe", line_width=2, opacity=0.5,
            )

        # Trail.
        self._trail_points.append(state.position.copy())
        if len(self._trail_points) >= 2:
            pts = np.array(self._trail_points)
            self.plotter.add_mesh(pv.lines_from_points(pts), color="gold", line_width=2,
                                  name="trail")

        self._place_camera(state)

    def screenshot(self) -> np.ndarray:
        """Return the current frame as an ``(H, W, 3)`` uint8 array."""
        return self.plotter.screenshot(return_img=True)

    def close(self) -> None:
        """Release the plotter."""
        self.plotter.close()

    def reset(self) -> None:
        """Clear the trajectory trail (new episode)."""
        self._trail_points.clear()
