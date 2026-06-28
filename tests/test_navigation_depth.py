"""Tests for depth-camera (PEDRA-style) perception on NavigationEnv.

Skipped if PyVista (the optional [render] extra) isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista")

# VTK 9.6 emits NumPy-2.5 deprecation warnings from its own internals.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

import rotorenv
from rotorenv.envs.navigation_env import NavigationEnv


def test_depth_observation_space_is_image() -> None:
    """Depth perception exposes a (1, H, W) channel-first image in [0, 1]."""
    env = NavigationEnv(perception="depth")
    h, w = NavigationEnv.DEPTH_SHAPE
    assert env.observation_space.shape == (1, h, w)
    assert env.observation_space.low.min() == 0.0
    assert env.observation_space.high.max() == 1.0
    env.close()


def test_depth_obs_is_valid_image() -> None:
    """reset/step return a normalised depth image inside the space."""
    env = NavigationEnv(perception="depth")
    h, w = NavigationEnv.DEPTH_SHAPE
    obs, _info = env.reset(seed=0, options={"difficulty": 0.6})
    assert obs.shape == (1, h, w)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    obs2, _r, _t, _tr, _i = env.step(np.array([0.6, 0, 0, 0], dtype=np.float32))
    assert env.observation_space.contains(obs2)
    env.close()


def test_depth_registered_variant() -> None:
    """NavigationDepth-v0 is registered and produces image observations."""
    env = rotorenv.make("NavigationDepth-v0")
    obs, _info = env.reset(seed=0)
    assert obs.ndim == 3 and obs.shape[0] == 1
    env.close()


def test_depth_responds_to_geometry() -> None:
    """A near obstacle yields smaller (nearer) depth values than open sky."""
    env = NavigationEnv(perception="depth")
    obs, _info = env.reset(seed=4, options={"difficulty": 1.0})
    # Some pixels must be clearly "near" (< 0.5) given obstacles/floor in view.
    assert float(obs.min()) < 0.5
    # And some "far"/sky pixels at the cap.
    assert float(obs.max()) >= 0.99
    env.close()
