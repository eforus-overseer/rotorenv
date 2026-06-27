"""Train a PPO policy on a rotorenv task and visualise the result.

This is an *example*, not part of the core package — it depends on the optional
``[rl]`` extra (``pip install -e ".[rl]"``). It demonstrates that the
environment produces a learnable signal: train, evaluate, plot the learning
curve, and optionally watch the trained policy fly in the live 3D window.

Usage:
    python examples/train_ppo.py --env Hover6DOF-v0 --steps 50000 --render

Outputs:
    runs/<env>/learning_curve.png   learning curve (eval mean reward vs steps)
    runs/<env>/best_model.zip       best policy found during training
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import rotorenv  # noqa: F401  (registers the env IDs)


def make_env(env_id: str, render_mode: str | None = None):
    """Factory returning a Monitor-wrapped rotorenv environment.

    Args:
        env_id: Registered environment ID, e.g. ``"Hover6DOF-v0"``.
        render_mode: Optional Gymnasium render mode.

    Returns:
        A callable (thunk) that constructs the wrapped env, as SB3 expects.
    """
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common.monitor import Monitor

    def _thunk():
        env = rotorenv.make(env_id, render_mode=render_mode)
        return Monitor(env)

    return _thunk


def train(env_id: str, steps: int, seed: int, out_dir: str) -> str:
    """Train PPO and save the best model + a learning-curve plot.

    Args:
        env_id: Task to train on.
        steps: Total environment steps to train for.
        seed: RNG seed for reproducibility.
        out_dir: Directory to write artifacts into.

    Returns:
        Path to the saved best-model zip.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    os.makedirs(out_dir, exist_ok=True)
    train_env = DummyVecEnv([make_env(env_id)])
    eval_env = DummyVecEnv([make_env(env_id)])

    # Periodically evaluate the greedy policy on a separate env -> honest curve.
    eval_freq = max(2000, steps // 20)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model = PPO("MlpPolicy", train_env, seed=seed, verbose=1)
    print(f"\nTraining PPO on {env_id} for {steps:,} steps "
          f"(eval every {eval_freq:,})...\n")
    model.learn(total_timesteps=steps, callback=eval_cb, progress_bar=False)

    best_path = os.path.join(out_dir, "best_model.zip")
    if not os.path.exists(best_path):  # fall back if no eval improvement saved
        best_path = os.path.join(out_dir, "final_model.zip")
        model.save(best_path)

    _plot_curve(out_dir)
    return best_path


def _plot_curve(out_dir: str) -> None:
    """Render the eval learning curve from EvalCallback's evaluations.npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = os.path.join(out_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        print("(no evaluations.npz found; skipping plot)")
        return
    data = np.load(npz_path)
    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)

    plt.figure(figsize=(7, 4))
    plt.plot(timesteps, mean_rewards, marker="o")
    plt.xlabel("environment steps")
    plt.ylabel("eval mean episode reward")
    plt.title("PPO learning curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(out_png, dpi=110)
    plt.close()
    print(f"\nSaved learning curve -> {out_png}")
    print(f"  first eval mean reward: {mean_rewards[0]:8.2f}")
    print(f"  last  eval mean reward: {mean_rewards[-1]:8.2f}")


def evaluate(env_id: str, model_path: str, n_episodes: int, seed: int) -> None:
    """Print a quantitative summary of the trained policy.

    Args:
        env_id: Task the policy was trained on.
        model_path: Path to the saved model zip.
        n_episodes: Number of eval episodes.
        seed: Base seed.
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = rotorenv.make(env_id)

    returns, lengths, final_dists, successes = [], [], [], 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        total, steps = 0.0, 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            steps += 1
        returns.append(total)
        lengths.append(steps)
        final_dists.append(info["distance"])
        if info["distance"] < 0.3:
            successes += 1
    env.close()

    print("\n=== Evaluation ({} episodes) ===".format(n_episodes))
    print(f"  mean return       : {np.mean(returns):8.2f} +/- {np.std(returns):.2f}")
    print(f"  mean episode len  : {np.mean(lengths):8.1f} steps")
    print(f"  mean final dist   : {np.mean(final_dists):8.3f} m")
    print(f"  success rate (<0.3m): {successes}/{n_episodes}")


def replay(env_id: str, model_path: str, seed: int) -> None:
    """Run the trained policy in the live 3D window."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = rotorenv.make(env_id, render_mode="human")
    obs, _info = env.reset(seed=seed)
    env.render()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, _info = env.step(action)
        env.render()
    input("Press Enter to close the window...")
    env.close()


def main() -> None:
    """Parse args and run train -> eval -> (optional) replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="Hover6DOF-v0", help="registered env id")
    parser.add_argument("--steps", type=int, default=50_000, help="training steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="live 3D replay after training")
    parser.add_argument("--replay-only", action="store_true",
                        help="skip training; load the saved best_model and replay in 3D")
    args = parser.parse_args()

    out_dir = os.path.join("runs", args.env)
    if args.replay_only:
        replay(args.env, os.path.join(out_dir, "best_model.zip"), seed=1000)
        return

    best = train(args.env, args.steps, args.seed, out_dir)
    evaluate(args.env, best, args.eval_episodes, seed=1000)
    if args.render:
        replay(args.env, best, seed=1000)


if __name__ == "__main__":
    main()
