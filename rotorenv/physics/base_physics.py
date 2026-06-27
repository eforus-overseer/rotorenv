"""Physics backend interface.

The physics layer is intentionally decoupled from the environment via a
``Protocol``. Any object that exposes a matching ``step`` (and the documented
constants) is a valid physics backend — no inheritance required. This is what
lets a Phase-2 6-DOF rigid-body model replace the Phase-1 point mass without the
environment importing or knowing about either concrete class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rotorenv.core.action import DroneAction
from rotorenv.core.state import DroneState


@runtime_checkable
class DronePhysics(Protocol):
    """Structural interface every physics backend must satisfy.

    Attributes:
        dt: Integration timestep in seconds.
    """

    dt: float

    def step(self, state: DroneState, action: DroneAction) -> DroneState:
        """Advance the simulation by one ``dt`` and return the new state.

        Implementations must treat ``state`` as immutable (return a new
        :class:`DroneState` rather than mutating the input).

        Args:
            state: Current drone state.
            action: Control command to apply over this step.

        Returns:
            The drone state after one integration step.
        """
        ...
