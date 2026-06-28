# %% [markdown]
# # rotorenv — train a drone agent on Colab (GPU)
#
# Train PPO navigation policies for [`rotorenv`](https://github.com/eforus-overseer/rotorenv)
# on a free Colab GPU. Vision-based navigation (depth-camera + CNN) is
# compute-bound — the GPU here is what makes it tractable, unlike CPU.
#
# **Before running:** `Runtime → Change runtime type → Hardware accelerator → T4 GPU`.
#
# This notebook:
# 1. Installs rotorenv + RL/render extras, and a headless display (Xvfb) for off-screen PyVista.
# 2. Confirms the GPU is visible to PyTorch.
# 3. Runs a fast **state-perception** navigation train (sanity check).
# 4. Runs the **vision** (depth-camera + CNN) navigation train — the PEDRA-style task.
# 5. Renders a GIF of the trained policy flying the obstacle field.
#
# This file is the percent-format source; build the .ipynb with
# `python scripts/build_notebook.py notebooks/rotorenv_colab_train.py`.

# %% [markdown]
# ## 1. Setup — clone, install, headless display

# %%
# System libs for off-screen GL + a virtual X display (depth rendering needs both).
!apt-get -qq update && apt-get -qq install -y xvfb libgl1-mesa-glx > /dev/null 2>&1
# Clone the repo and install with the RL + render extras.
![ -d rotorenv ] || git clone https://github.com/eforus-overseer/rotorenv.git
%cd rotorenv
!git pull --quiet
# `set -e` so a failed install aborts the cell loudly instead of silently
# leaving training to crash later (which looks like "no model saved").
!set -e && pip -q install -e ".[rl,render]" && echo "INSTALL OK"

# %%
# Confirm the GPU is visible to PyTorch (the whole point of using Colab).
import torch
print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('WARNING: no GPU — set Runtime > Change runtime type > T4 GPU')

# Sanity-check the rotorenv install (catches a failed pip install before training).
import rotorenv, gymnasium
print('rotorenv import OK | gymnasium', gymnasium.__version__)

# %% [markdown]
# ## 2. Sanity check — state-perception navigation (fast)
#
# Trains on the kinematic state vector (no rendering in the loop), so it's fast
# even on CPU. With `ProgressReward` + curriculum it should reach high success —
# this confirms the env + training recipe before we spend GPU time on vision.

# %%
!python examples/train_nav_curriculum.py --env Navigation6DOF-v0 --steps 150000 --seed 0

# %% [markdown]
# ## 3. Vision navigation — depth camera + CNN (the PEDRA-style task)
#
# The agent now perceives obstacles through an onboard 64×64 depth image and
# learns with a CNN. This is the compute-heavy run the GPU is for. Start with
# ~300k steps; increase if the curriculum difficulty is still climbing at the end.
#
# The script checkpoints the model every 10k steps, so a disconnect won't lose progress.
# Run the cell to completion (or until curriculum difficulty climbs) before rendering below.
#
# **Headless rendering setup** (run the next cell ONCE before vision training):
# depth perception renders off-screen via PyVista/VTK, which needs a virtual X
# display. We launch Xvfb on :99 and export DISPLAY so the `!python` training
# subprocess below inherits it. The smoke-test then fails fast here if rendering
# is broken, instead of crashing 10 minutes into training.

# %%
import os, subprocess, time
# Launch a virtual framebuffer (idempotent: ignore if already running).
subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)
os.environ['DISPLAY'] = ':99'
os.environ['PYVISTA_OFF_SCREEN'] = 'true'
import pyvista as pv
pv.OFF_SCREEN = True

# Smoke-test off-screen depth rendering BEFORE the long training run.
import numpy as np
from rotorenv.rendering.depth_camera import DepthCamera
from rotorenv.core.state import DroneState
_cam = DepthCamera(np.zeros((0, 6)), height=32, width=32)
_img = _cam.capture(DroneState(np.array([0., 0, 1.5]), np.zeros(3), np.zeros(3), np.zeros(3), 0.))
_cam.close()
print('off-screen depth render OK, frame', _img.shape, '| DISPLAY =', os.environ.get('DISPLAY'))

# %%
!python examples/train_nav_curriculum.py --env NavigationDepth-v0 --steps 300000 --seed 0

# %% [markdown]
# ## 4. Render the trained policy flying the obstacle field

# %%
import os, glob, numpy as np, rotorenv
from stable_baselines3 import PPO
from rotorenv.rendering.pyvista_renderer import PyVistaRenderer
import imageio.v2 as imageio
from IPython.display import Image as IPyImage, display

# Auto-detect which policy was actually trained (vision preferred, else state).
# SB3 saves/loads extension-less, so a model.zip on disk loads as 'model'.
_candidates = ['NavigationDepth-v0', 'Navigation6DOF-v0']
_found = [e for e in _candidates if os.path.exists(f'runs/{e}_curriculum/model.zip')]
if not _found:
    raise FileNotFoundError(
        'No trained model found. Run the training cell(s) above first. '
        f'Looked in: {[f"runs/{e}_curriculum/" for e in _candidates]}. '
        f'Present: {glob.glob("runs/*/model.zip")}')
ENV_ID = _found[0]
print(f'Loading trained policy: {ENV_ID}')
model = PPO.load(f'runs/{ENV_ID}_curriculum/model')   # no .zip — SB3 appends it

# Find a good episode (reaches the goal) at mid difficulty.
best = None
for seed in range(2000, 2040):
    env = rotorenv.make(ENV_ID); inner = env.unwrapped
    obs, _ = env.reset(seed=seed, options={'difficulty': 0.5})
    states = [inner.state.copy()]; term = trunc = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a); states.append(inner.state.copy())
    env.close()
    if info.get('reached_goal') and (best is None or len(states) < best[0]):
        best = (len(states), states, inner.target.copy(), inner.obstacles.copy())

if best is None:
    print('No successful episode at d=0.5 — try more training steps or a lower difficulty.')
else:
    _, states, target, obstacles = best
    r = PyVistaRenderer(camera_mode='orbit', window_size=(560, 420))
    r.add_obstacles(obstacles); r.reset()
    frames = [r.render_frame(s, target) or r.screenshot() for s in states]
    r.close()
    imageio.mimsave('trained_flight.gif', frames, fps=15, loop=0)
    print(f'{len(frames)} frames')
    display(IPyImage(filename='trained_flight.gif'))

# %% [markdown]
# ## Notes
# - **Save your model:** download `runs/<env>_curriculum/model.zip`, or mount
#   Google Drive and copy it there, before the runtime disconnects.
# - **If vision training stalls** (curriculum stuck at 0.0): it needs more steps.
#   Increase `--steps`, or shrink the task (fewer obstacles) first.
# - **State vs vision:** `Navigation6DOF-v0` (state) trains in minutes and hits
#   ~80% success; `NavigationDepth-v0` (vision) is the hard, realistic version
#   that benefits most from the GPU.
