"""Demonstrate the curriculum wrapper on the waypoint task.

Runs the success-based schedule and prints how difficulty adapts as a crude
hand-coded policy gains (or loses) competence. This is illustrative — a real
training loop would replace the policy with a learning agent.

Run:
    python examples/curriculum_demo.py
"""

from __future__ import annotations

import numpy as np

import rotorenv
from rotorenv.envs.curriculum import CurriculumWrapper

N_EPISODES = 40
SEED = 0


def altitude_seeking_action(obs: np.ndarray) -> np.ndarray:
    """Crude policy: bias thrust by the vertical distance-to-target (obs[11])."""
    thrust = float(np.clip(0.2 + 1.5 * obs[11], -1.0, 1.0))
    return np.array([thrust, 0.0, 0.0, 0.0], dtype=np.float32)


def run() -> None:
    """Run the success-based curriculum and report difficulty over episodes."""
    env = CurriculumWrapper(
        rotorenv.make("Waypoint-v0"),
        mode="success",
        start_difficulty=0.1,
        difficulty_step=0.15,
        success_distance=1.0,
        success_threshold=0.6,
        window=5,
    )

    print("Curriculum (success mode) on Waypoint-v0")
    print("ep  difficulty  final_distance  result")
    for episode in range(N_EPISODES):
        obs, info = env.reset(seed=SEED + episode)
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _r, terminated, truncated, info = env.step(altitude_seeking_action(obs))
        success = info["distance"] < env.success_distance
        print(f"{episode:2d}    {info['difficulty']:.2f}        "
              f"{info['distance']:.2f} m         {'hit' if success else 'miss'}")

    env.close()


if __name__ == "__main__":
    run()
