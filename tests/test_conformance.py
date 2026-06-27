"""Gymnasium API conformance tests via the official env_checker.

``check_env`` exercises the env against the Gymnasium contract: space/dtype
agreement, the 5-tuple ``step`` signature, ``reset`` seeding, and render-mode
handling. Mature environment libraries treat this as table stakes; running it
across every registered variant and space configuration guards the contract
cheaply.
"""

from __future__ import annotations

import pytest
from gymnasium.utils.env_checker import check_env

from rotorenv.core.enums import ActionType, ObservationType
from rotorenv.envs.hover_env import HoverEnv


@pytest.mark.parametrize(
    "env_id",
    [
        "Hover-v0",
        "Hover6DOF-v0",
        "HoverMinimal-v0",
        "HoverThrustOnly-v0",
        "Waypoint-v0",
        "Waypoint6DOF-v0",
        "Trajectory-v0",
        "Trajectory6DOF-v0",
        "Navigation-v0",
        "Navigation6DOF-v0",
    ],
)
def test_registered_envs_pass_check_env(env_id: str) -> None:
    """Each registered environment satisfies the Gymnasium API contract."""
    import rotorenv

    env = rotorenv.make(env_id)
    # skip_render_check avoids opening a matplotlib window during CI.
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("obs_type", list(ObservationType))
@pytest.mark.parametrize("act_type", list(ActionType))
def test_all_space_configs_pass_check_env(
    obs_type: ObservationType, act_type: ActionType
) -> None:
    """Every (observation_type, action_type) combination is API-conformant."""
    env = HoverEnv(observation_type=obs_type, action_type=act_type)
    check_env(env, skip_render_check=True)
    env.close()
