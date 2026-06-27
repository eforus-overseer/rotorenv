"""Demonstrate wrappers and vectorized envs (the mature-env patterns).

Shows three idioms straight from the Gymnasium guide:
1. Composing rotorenv wrappers (``NormalizeObservation`` + ``RewardScale``).
2. Confirming the normalised observation stays in ``[-1, 1]``.
3. Running parallel copies with ``gymnasium.make_vec``.

Run:
    python examples/wrapped_agent.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import rotorenv
from rotorenv.envs.wrappers import NormalizeObservation, RewardScale

SEED = 0


def demo_wrappers() -> None:
    """Run one episode through normalised observations and scaled rewards."""
    env = RewardScale(NormalizeObservation(rotorenv.make("Hover6DOF-v0")), scale=0.1)
    obs, _info = env.reset(seed=SEED)
    env.action_space.seed(SEED)

    total, steps, term, trunc = 0.0, 0, False, False
    obs_min, obs_max = obs.copy(), obs.copy()
    while not (term or trunc):
        obs, r, term, trunc, _info = env.step(env.action_space.sample())
        obs_min, obs_max = np.minimum(obs_min, obs), np.maximum(obs_max, obs)
        total += r
        steps += 1
    env.close()

    print("[wrappers] NormalizeObservation + RewardScale(0.1)")
    print(f"  scaled return over {steps} steps: {total:.3f}")
    print(f"  obs range observed: [{obs_min.min():.3f}, {obs_max.max():.3f}] "
          f"(should stay within [-1, 1])")


def demo_vectorized() -> None:
    """Step 4 parallel copies of the env together."""
    vec = gym.make_vec("Hover-v0", num_envs=4)
    vec.reset(seed=SEED)
    actions = np.zeros((4,) + vec.single_action_space.shape, dtype=np.float32)
    _obs, rewards, _term, _trunc, _info = vec.step(actions)
    vec.close()
    print("\n[vectorized] gym.make_vec(num_envs=4)")
    print(f"  per-env rewards on a neutral step: {np.round(rewards, 3)}")


if __name__ == "__main__":
    demo_wrappers()
    demo_vectorized()
