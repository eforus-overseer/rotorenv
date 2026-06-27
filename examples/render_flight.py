"""Render a trained policy's flight to a cinematic MP4/GIF (PyVista).

Loads a saved PPO model, rolls out one episode, and renders it through the
PyVista 3D engine with a moving camera (third-person chase, onboard POV, or a
static orbit shot). Off-screen, so it produces a video file without a window.

Requires the optional extras:
    pip install -e ".[rl,render]"

Usage:
    python examples/render_flight.py --env HoverEasy-v0 --camera chase --out flight.mp4
    python examples/render_flight.py --env HoverEasy-v0 --camera pov   --out pov.gif
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import rotorenv  # noqa: F401  (registers env IDs)


def rollout_states(env_id: str, model_path: str, seed: int):
    """Roll out one greedy episode, returning the list of states + target.

    Args:
        env_id: Registered environment ID.
        model_path: Path to a saved stable-baselines3 model zip.
        seed: Reset seed.

    Returns:
        ``(states, target)`` — a list of :class:`DroneState` and the target.
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)
    env = rotorenv.make(env_id)
    inner = env.unwrapped

    obs, _info = env.reset(seed=seed)
    states = [inner.state.copy()]
    target = inner.target.copy()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, _info = env.step(action)
        states.append(inner.state.copy())
    env.close()
    return states, target


def render_video(states, target, camera: str, out_path: str, fps: int) -> None:
    """Render the rolled-out states to an MP4 or GIF via PyVista.

    Args:
        states: Sequence of drone states (one per frame).
        target: Target position to mark in the scene.
        camera: Camera mode (``"chase"``, ``"pov"``, ``"orbit"``).
        out_path: Output file; ``.mp4`` or ``.gif`` chosen by extension.
        fps: Frames per second.
    """
    import imageio.v2 as imageio

    from rotorenv.rendering.pyvista_renderer import PyVistaRenderer

    renderer = PyVistaRenderer(camera_mode=camera)
    renderer.reset()
    frames = []
    for state in states:
        renderer.render_frame(state, target)
        frames.append(renderer.screenshot())
    renderer.close()

    if out_path.lower().endswith(".gif"):
        imageio.mimsave(out_path, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    print(f"wrote {len(frames)} frames -> {out_path}  (camera={camera}, fps={fps})")


def main() -> None:
    """Parse args, roll out the policy, and render the flight video."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HoverEasy-v0")
    parser.add_argument("--model", default=None, help="model zip (default: runs/<env>/best_model.zip)")
    parser.add_argument("--camera", default="chase", choices=["chase", "pov", "orbit"])
    parser.add_argument("--out", default=None, help="output .mp4 or .gif")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    model_path = args.model or os.path.join("runs", args.env, "best_model.zip")
    out_path = args.out or os.path.join("runs", args.env, f"flight_{args.camera}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    states, target = rollout_states(args.env, model_path, args.seed)
    print(f"rolled out {len(states)} steps on {args.env}")
    render_video(states, target, args.camera, out_path, args.fps)


if __name__ == "__main__":
    main()
