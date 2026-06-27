"""Sanity check: run a random policy and print total reward per episode.

This is the Phase-1 runnable artifact. It exercises the full pipeline
(reset -> step loop -> termination) with actions sampled uniformly from the
action space, and reports per-episode return, length, and final distance.
"""

from __future__ import annotations

import numpy as np

import rotorenv

N_EPISODES = 3
SEED = 0


def run() -> None:
    """Run :data:`N_EPISODES` episodes with a random policy and print results."""
    env = rotorenv.make("Hover-v0")
    # Seed the action space so the random policy is reproducible too.
    env.action_space.seed(SEED)

    for episode in range(N_EPISODES):
        _obs, info = env.reset(seed=SEED + episode)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        print(
            f"episode {episode}: "
            f"return={total_reward:8.3f}  "
            f"steps={steps:4d}  "
            f"final_distance={info['distance']:.3f} m  "
            f"{'terminated' if terminated else 'truncated'}"
        )

    env.close()


if __name__ == "__main__":
    run()
