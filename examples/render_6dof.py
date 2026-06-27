"""Watch a 6-DOF episode in the live 3D window.

Opens an interactive matplotlib window showing the tilting quadrotor cross, its
body-up axis, a fading trajectory trail, and the target. A simple hand-tuned
"hover-ish" policy is used (full takeoff thrust, then settle) so there is some
visible flight rather than an instant random crash.

Run:
    python examples/render_6dof.py
"""

from __future__ import annotations

import numpy as np

import rotorenv

N_STEPS = 400
SEED = 0


def simple_policy(obs: np.ndarray, step: int) -> np.ndarray:
    """A crude altitude controller: climb hard, then hold ~hover thrust.

    Uses the vertical distance-to-target component (obs index 11) to bias
    thrust. This is *not* a trained policy — just enough to produce watchable
    flight. Action layout is ``[thrust, roll, pitch, yaw]`` in ``[-1, 1]``.

    Args:
        obs: Current observation vector, shape ``(13,)``.
        step: Current step index (unused; kept for signature clarity).

    Returns:
        A raw action in ``[-1, 1]^4``.
    """
    dz = float(obs[11])  # target_z - z  (distance-to-target, z component)
    # Map a desired vertical correction to raw thrust in [-1, 1] (0 -> 50%).
    thrust_raw = float(np.clip(0.1 + 2.0 * dz, -1.0, 1.0))
    return np.array([thrust_raw, 0.0, 0.0, 0.0], dtype=np.float32)


def run() -> None:
    """Run one rendered 6-DOF episode with the simple policy."""
    env = rotorenv.make("Hover6DOF-v0", render_mode="human")
    obs, _info = env.reset(seed=SEED)
    env.render()

    for step in range(N_STEPS):
        action = simple_policy(obs, step)
        obs, _reward, terminated, truncated, _info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"episode ended at step {step} "
                  f"({'terminated' if terminated else 'truncated'})")
            break

    input("Press Enter to close the window...")
    env.close()


if __name__ == "__main__":
    run()
