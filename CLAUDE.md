# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

`rotorenv` is a lightweight, [Gymnasium](https://gymnasium.farama.org/)-compatible
reinforcement learning environment for training autonomous quadrotor agents. Pure
Python, physics-first, built to be extended incrementally in phases.

**Current phase: Phase 6 (complete)** — "MiniGrid in 3D" procedural navigation,
vision perception, and curriculum-driven training. `NavigationEnv`: random
pillar field, fly start→goal without colliding, obstacle count scales with
`self.difficulty`. Perception modes: `"state"` (~22k steps/s) and `"depth"`
(64×64 PyVista depth, ~490 steps/s — channel-first `(1,H,W)`, needs
`policy_kwargs={"normalize_images": False}` + `CnnPolicy`).

**Key training lesson — replicate this pattern for new nav tasks:** direct
training at max difficulty stalls (0/10 success). The unlock is
(a) `ProgressReward(scale=1.0)` for dense step-wise shaping (replaces the
absolute `DistancePenalty` which was too sparse), and (b) the Phase-4
`CurriculumWrapper(mode="success", start_difficulty=0.0)` which begins in an
empty arena and advances obstacle density on a recent-success rate. With both,
200k MLP-PPO steps on `Navigation6DOF-v0` reach difficulty 0.6 with **80%
success in empty arenas, 55% at d=0.5**. See `examples/train_nav_curriculum.py`.
Registered: `Navigation-v0`, `Navigation6DOF-v0`, `NavigationDepth-v0`. See
[[rotorenv-vision-3d-navigation]].

Phase 5 — PPO training pipeline. `examples/train_ppo.py` trains/eval/plots/
replays SB3 PPO; SB3 is the optional `[rl]` extra. Env verified learnable
(~2.7→11.9 over 50k steps on `HoverEasy-v0`). Phases 1–4: point mass, 6-DOF,
enum spaces/check_env/wrappers, tasks + curriculum. Optional `[render]` extra =
PyVista cinematic renderer (chase/POV/orbit), GIFs in docs/media.

**Key training lesson:** ground-spawn tasks (z=0, crash at z<0) are nearly
unlearnable from scratch — a cold policy crashes on step 1 (reward flat at
-5.50). Train on airborne-spawn variants (`HoverEasy-v0`, spawn_height=1.0) for
pure attitude-stabilised hover. `spawn_height` is a `HoverEnv` arg (default 0.0
preserves the takeoff task and all prior tests).

We deliberately mirror how established Gymnasium envs are built
(`gym-pybullet-drones`, MiniGrid, quad-swarm-rl, QuadCtrl) rather than inventing
conventions — see the project memory note. Prefer enum-configurable spaces,
registered task variants, and wrappers over bespoke env code.

## Environment setup (read before installing)

Local Python is Homebrew's and is **externally managed (PEP 668)** — a bare
`pip install` will be refused. Always work inside the project virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run things with the venv interpreter (`.venv/bin/python`) if the env isn't activated.

## Common commands

```bash
python examples/random_agent.py    # sanity check: 3 random episodes, prints return
python examples/render_6dof.py     # live 3D window: tilting quad + trajectory trail
python examples/wrapped_agent.py   # wrappers + vectorized-env demo
python examples/curriculum_demo.py # success-based curriculum on Waypoint-v0
python examples/train_ppo.py --env HoverEasy-v0 --steps 50000   # train PPO (needs .[rl])
python examples/train_nav_curriculum.py --env Navigation6DOF-v0 --steps 200000  # curriculum nav
pytest                             # full test suite (currently 99 tests)
pytest tests/test_conformance.py   # Gymnasium check_env across all variants
```

Optional extras: `pip install -e ".[rl]"` (stable-baselines3 for training),
`pip install -e ".[render]"` (PyVista for cinematic/depth rendering).

## Notebooks (author as .py, build the .ipynb)

Do **not** hand-write `.ipynb` JSON — it's heavy and unreviewable. Author the
notebook as a percent-format `.py` (`# %%` code cells, `# %% [markdown]` prose)
under `notebooks/`, then build the `.ipynb` with the converter:

```bash
python scripts/build_notebook.py notebooks/rotorenv_colab_train.py
```

Commit the `.py` source; the generated `notebooks/*.ipynb` is git-ignored
(a build artifact). The Colab training notebook lives at
`notebooks/rotorenv_colab_train.py` — vision-nav (depth+CNN) is GPU-bound, so
run it on Colab, not CPU.

## Architecture — keep these concerns separate

Physics, reward, and observation are independent and swappable. Do not collapse
them into the env loop.

```
rotorenv/
├── core/        DroneState, DroneAction, reward terms, enums, rotations.py
├── physics/     DronePhysics Protocol + PointMassPhysics + SixDOFPhysics
├── envs/        DroneEnv base + tasks (Hover/Waypoint/Trajectory/Navigation)
│                + CurriculumWrapper + obs/reward wrappers
└── rendering/   MatplotlibRenderer (default, lazy) + PyVistaRenderer +
                 DepthCamera (both optional, [render] extra)
```

Tasks (each a `DroneEnv` subclass): `HoverEnv`, `WaypointEnv` (sampled target),
`TrajectoryEnv` (moving Lissajous target), `NavigationEnv` (procedural obstacle
field → goal, `perception` = `"state"` | `"depth"`).

- **Physics is a `typing.Protocol`** (`physics/base_physics.py`). A new backend
  only needs a matching `step(state, action) -> DroneState` and a `dt` attribute.
  The env depends on the *shape*, never on a concrete class — do not make
  backends inherit from a base class. `SixDOFPhysics` (Phase 2) was added this
  way with zero changes to `DroneEnv`/`HoverEnv`. Select via `physics_model`
  (`"point_mass"` | `"six_dof"`) or pass an explicit `physics=` instance.
- **One frame convention, in `core/rotations.py`.** ZYX Euler, scalar-first
  quaternions `[w,x,y,z]`. All backends and the renderer use it — do not inline
  ad-hoc rotation maths. 6-DOF integrates attitude as a quaternion internally and
  converts to Euler only at the `DroneState` boundary (the fixed contract).
- **Reward is data, not code.** A task's shaping is a list of `RewardTerm`
  objects summed by `CompositeReward` (`core/reward.py`). Add behaviour by adding
  a term, not by editing a monolithic reward function. Most terms are pure
  functions; `ProgressReward` is the exception (stateful — holds last distance,
  needs `reset(spawn, target)` per episode). It gives dense step-wise shaping
  and is what makes long goal-reaching tasks learnable.
- **Tasks subclass `DroneEnv`** and implement only four hooks: `_initial_state`,
  `_make_target`, `_build_reward`, `_is_terminated` (+ optional `_is_truncated`).
  They never re-implement `step()` or the spaces.

## Conventions (enforce in review)

- Python 3.10+. Type hints on all public functions; docstrings on all public
  functions and classes.
- `numpy` for all math. Array state fields are `float64`, shape `(3,)`.
- **Randomness goes through `self.np_random`** (populated by `gym.Env.reset(seed=...)`).
  Never call the global `np.random.seed()` or top-level `np.random.*`.
- Physics `step()` must be pure: return a new `DroneState`, never mutate the input.
- Use `gymnasium`, never legacy `gym`.
- No external physics-engine dependency (no PyBullet / MuJoCo); 6-DOF is
  hand-rolled numpy integration.
- matplotlib is imported lazily inside the renderer — keep `import rotorenv` and
  headless training free of a hard matplotlib-at-import cost.

## Domain model (fixed contract — changing shapes is a breaking change)

`DroneState`: `position(3)`, `velocity(3)`, `orientation(3)` [roll, pitch, yaw],
`angular_velocity(3)`, `time` (scalar).

`DroneAction`: `thrust [0,1]`, `roll_cmd/pitch_cmd/yaw_cmd [-1,1]`. The raw policy
vector is `[-1,1]^4`; `DroneAction.from_array` rescales thrust so a neutral `0`
command = 50% throttle (hover).

**Observation & action shapes are enum-configured** (`core/enums.py`), not fixed
constants — this is the `gym-pybullet-drones` pattern. Shapes come from
`OBSERVATION_DIMS[obs_type]` / `ACTION_DIMS[act_type]`; `_get_obs` and
`_preprocess_action` branch on the enum. To add a mode, add an enum member + its
dim + a branch — do **not** reintroduce a hardcoded `OBS_DIM`.

- `ObservationType.FULL` (16,, default): minimal + `angular_velocity(3)`.
- `ObservationType.MINIMAL` (13,): `position + velocity + orientation +
  distance_to_target + time` (legacy Phase 1/2 layout).
- `ActionType.ATTITUDE` (4,, default): `[thrust, roll, pitch, yaw]`.
- `ActionType.THRUST_ONLY` (1,): `[thrust]`, attitude held at zero.

The observation `Box` has **finite** per-field bounds (`_observation_bounds`) so
it passes `check_env` and works with normalization wrappers — never revert to
`Box(-inf, inf)`.

## Design decisions worth knowing

- **Thrust tilts with orientation.** Thrust is applied along the body up-axis
  rotated into the world frame (ZYX), so roll/pitch actually translate the drone.
  This is the core dynamics convention a Phase-2 model inherits.
- **Spawn sits on the crash plane (`z=0`, crash when `z<0`).** The agent must
  learn *takeoff*, not just hold — a neutral action under spawn-tilt noise sinks
  and crashes. To start airborne, change `HoverEnv._initial_state` (one line).
  (TrajectoryEnv is the exception: it spawns *on* the path.)
- **Tasks differ only in target + reward + termination.** A new task subclasses
  `DroneEnv` and overrides `_make_target` (initial target), `_initial_state`,
  `_build_reward`, `_is_terminated`, and — for a *moving* target — `_update_target`
  (called each step; default no-op). Don't touch the step loop.
- **Difficulty is env state, curriculum is a wrapper.** `self.difficulty ∈ [0,1]`
  is set via `reset(options={"difficulty": d})`; tasks scale spawn/target spread
  by it. The `CurriculumWrapper` *drives* that value (success-based or
  step-annealed) — the env stays a pure MDP, the schedule stays composable. Keep
  it that way: no training-history state inside the env.
- **Depth perception is the heavy path; isolate it.** `NavigationEnv(perception=
  "depth")` renders an off-screen depth image per step via `DepthCamera` (~490
  steps/s vs ~22k for state). The image obs is channel-first `(1,H,W)`,
  pre-normalised to `[0,1]` → CNN-PPO needs `policy_kwargs={"normalize_images":
  False}`. Gotcha that cost a run: `get_image_depth()` needs `show(...,
  store_image_depth=True)`; the camera rebuilds per episode, so `capture()` also
  retries defensively if the buffer is dropped. Long training checkpoints
  `model.zip` every 10k steps because these runs can crash/disconnect.

## Working agreement

- Build iteratively: scaffold → run → observe → adjust. Keep every change
  runnable and tested. Don't jump phases until the current one runs cleanly.
- When something is ambiguous, decide, then state what you chose and why.
- Flag any decision that's hard to reverse (frame conventions, obs/action shapes,
  the physics interface).
