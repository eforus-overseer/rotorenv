"""Rendering backends.

``MatplotlibRenderer`` is the zero-dependency default (fixed schematic camera).
``PyVistaRenderer`` is the optional cinematic engine (chase/POV/orbit cameras,
saved video); it requires the ``[render]`` extra and is imported lazily, so it is
intentionally *not* imported here.
"""

from rotorenv.rendering.matplotlib_renderer import MatplotlibRenderer

__all__ = ["MatplotlibRenderer"]
