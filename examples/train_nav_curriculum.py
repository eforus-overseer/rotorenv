"""Train PPO on NavigationEnv with the success-based curriculum wrapper.

Direct training at maximum obstacle difficulty produces a "do nothing" policy
(the goal is too sparse to find by random exploration). The curriculum starts in
an empty arena, lets PPO master "fly to goal", then promotes difficulty by
success rate — the same pattern MiniGrid uses for long-horizon procedural tasks.

Usage:
    python examples/train_nav_curriculum.py --steps 200000

Saves model + plot under ``runs/Navigation6DOF-v0_curriculum/``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import rotorenv  # noqa: F401


def main() -> None:
    """Train, eval at multiple difficulties, save the artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Navigation6DOF-v0")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rotorenv.envs.curriculum import CurriculumWrapper

    out_dir = os.path.join("runs", f"{args.env}_curriculum")
    os.makedirs(out_dir, exist_ok=True)

    def make_curr() -> Monitor:
        """Curriculum-wrapped env: starts in an empty arena, advances on success."""
        env = rotorenv.make(args.env)
        env = CurriculumWrapper(
            env,
            mode="success",
            start_difficulty=0.0,
            difficulty_step=0.1,
            success_distance=1.0,
            success_threshold=0.6,
            failure_threshold=0.15,
            window=20,
        )
        return Monitor(env)

    train_env = DummyVecEnv([make_curr])
    # Image observations (depth perception) need a CNN with image-normalisation
    # disabled (our depth is already [0, 1]); state vectors use an MLP.
    is_image = len(train_env.observation_space.shape) == 3
    policy = "CnnPolicy" if is_image else "MlpPolicy"
    policy_kwargs = {"normalize_images": False} if is_image else None
    model = PPO(policy, train_env, seed=args.seed, policy_kwargs=policy_kwargs, verbose=0)
    print(f"Training PPO ({policy}) on {args.env} with success-based curriculum "
          f"for {args.steps:,} steps...")
    model.learn(total_timesteps=args.steps)

    final_difficulty = train_env.envs[0].env.difficulty
    print(f"\nFinal curriculum difficulty reached: {final_difficulty:.2f} (0=empty, 1=max)")
    model.save(os.path.join(out_dir, "model.zip"))

    eval_difficulties = sorted({0.0, 0.25, 0.5, round(max(0.0, final_difficulty), 2)})
    print("\n=== Eval @ multiple difficulties (20 episodes each) ===")
    for diff in eval_difficulties:
        env = rotorenv.make(args.env)
        dists, successes = [], 0
        for ep in range(20):
            obs, _ = env.reset(seed=2000 + ep, options={"difficulty": diff})
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
            dists.append(info["distance"])
            if info.get("reached_goal"):
                successes += 1
        env.close()
        print(f"  difficulty={diff:.2f}: final_dist={np.mean(dists):5.2f} m  "
              f"reached={successes}/20")


if __name__ == "__main__":
    main()
