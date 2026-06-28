"""Onboard depth camera — PEDRA-style pixel perception.

Renders the world from the drone's point of view and returns a normalised depth
image, suitable as a CNN observation. This is the expensive perception path:
each ``capture`` runs an off-screen render, so using it inside a training loop is
far slower than the state-vector observation. It is therefore isolated here and
imported lazily; PyVista (the ``[render]`` extra) is required.

Depth convention: PyVista returns depth along the camera's -Z axis (negative
values). We negate to a positive distance, fill empty sky with ``max_depth``,
clip to ``[0, max_depth]`` and scale to ``[0, 1]`` so the CNN sees a clean image.
"""

from __future__ import annotations

import numpy as np

from rotorenv.core.rotations import euler_to_rotmat
from rotorenv.core.state import DroneState


class DepthCamera:
    """Off-screen depth camera rigidly attached to the drone body.

    Attributes:
        height: Image height in pixels.
        width: Image width in pixels.
        max_depth: Far clip in metres; sky/empty pixels map to this distance.
    """

    def __init__(
        self,
        obstacles: np.ndarray,
        height: int = 64,
        width: int = 64,
        max_depth: float = 10.0,
        fov_deg: float = 90.0,
    ) -> None:
        """Build the off-screen scene (ground + obstacle boxes) once.

        Args:
            obstacles: ``(N, 6)`` ``[cx, cy, cz, hx, hy, hz]`` boxes to render.
            height: Output image height in pixels.
            width: Output image width in pixels.
            max_depth: Far clipping distance in metres.
            fov_deg: Vertical camera field of view in degrees.
        """
        import pyvista as pv  # local import: keep PyVista optional

        self._pv = pv
        self.height = int(height)
        self.width = int(width)
        self.max_depth = float(max_depth)

        self.plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
        self.plotter.add_mesh(
            pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=40, j_size=40),
            color="gray",
        )
        for box in np.asarray(obstacles, dtype=np.float64):
            c, h = box[:3], box[3:]
            self.plotter.add_mesh(
                pv.Cube(center=tuple(c), x_length=2 * h[0],
                        y_length=2 * h[1], z_length=2 * h[2]),
                color="white",
            )
        self.plotter.camera.view_angle = float(fov_deg)
        # Prime the render pipeline with depth-buffer storage enabled so every
        # subsequent get_image_depth() works (otherwise the buffer is discarded
        # after each render and get_image_depth raises on reuse).
        self.plotter.show(auto_close=False, store_image_depth=True)

    def capture(self, state: DroneState) -> np.ndarray:
        """Return the normalised depth image from the drone's POV.

        Args:
            state: Current drone state (sets camera pose).

        Returns:
            ``(1, height, width)`` float32 array in ``[0, 1]`` — 0 = at the
            camera, 1 = at/under ``max_depth`` or empty sky. Channel-first to
            match stable-baselines3's CNN convention.
        """
        rot = euler_to_rotmat(state.orientation)
        forward = rot @ np.array([1.0, 0.0, 0.0])
        up = rot @ np.array([0.0, 0.0, 1.0])
        p = state.position
        self.plotter.camera.position = tuple(p)
        self.plotter.camera.focal_point = tuple(p + forward)
        self.plotter.camera.up = tuple(up)
        self.plotter.render()

        depth = self.plotter.get_image_depth(fill_value=np.nan)  # (H, W), -Z, NaN sky
        depth = -np.asarray(depth, dtype=np.float32)             # -> positive distance
        depth[~np.isfinite(depth)] = self.max_depth              # sky -> far
        depth = np.clip(depth, 0.0, self.max_depth) / self.max_depth
        return depth[None, :, :]                                  # (1, H, W) channel-first

    def close(self) -> None:
        """Release the plotter."""
        self.plotter.close()
