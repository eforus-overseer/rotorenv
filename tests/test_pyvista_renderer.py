"""Smoke tests for the optional PyVista renderer.

Skipped automatically if PyVista isn't installed (it's an optional extra), so
the suite stays green without the ``[render]`` deps.
"""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from rotorenv.core.state import DroneState
from rotorenv.rendering.pyvista_renderer import PyVistaRenderer

# VTK 9.6 emits NumPy-2.5 deprecation warnings from its own numpy_support
# module (not our code); silence them so the suite stays quiet.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _state(z: float = 1.0, roll: float = 0.0) -> DroneState:
    return DroneState(
        position=np.array([0.0, 0.0, z]),
        velocity=np.zeros(3),
        orientation=np.array([roll, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        time=0.0,
    )


def test_invalid_camera_mode_raises() -> None:
    """An unknown camera mode is rejected."""
    with pytest.raises(ValueError, match="camera_mode"):
        PyVistaRenderer(camera_mode="bird")


@pytest.mark.parametrize("camera", ["chase", "pov", "orbit"])
def test_renders_frames_offscreen(camera: str) -> None:
    """Each camera mode renders RGB frames off-screen without a display."""
    r = PyVistaRenderer(camera_mode=camera, window_size=(160, 120))
    r.reset()
    target = np.array([0.0, 0.0, 1.0])
    last = None
    for i in range(3):
        r.render_frame(_state(z=1.0 - 0.1 * i, roll=0.05 * i), target)
        last = r.screenshot()
    r.close()
    assert last.shape == (120, 160, 3)
    assert last.dtype == np.uint8
