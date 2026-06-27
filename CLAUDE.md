# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

`rotorenv` is a lightweight, [Gymnasium](https://gymnasium.farama.org/)-compatible
reinforcement learning environment for training autonomous quadrotor agents. Pure
Python, physics-first, built to be extended incrementally in phases.

**Current phase: Phase 2 (complete)** — full 6-DOF rigid-body physics alongside
the Phase-1 point mass. Two registered tasks (`Hover-v0` → point mass,
`Hover6DOF-v0` → 6-DOF), no ML framework dependency. The environment is the
product; policies come later.

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
pytest                             # full test suite (currently 38 tests)
pytest tests/test_six_dof.py -q    # one module
```

## Architecture — keep these concerns separate

Physics, reward, and observation are independent and swappable. Do not collapse
them into the env loop.

```
rotorenv/
├── core/        DroneState, DroneAction, reward terms, rotations.py (frame utils)
├── physics/     DronePhysics Protocol + PointMassPhysics + SixDOFPhysics
├── envs/        DroneEnv base (all Gym plumbing) + HoverEnv task
└── rendering/   Matplotlib 3D renderer (lazy-imported, attitude-aware)
```

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
  a term, not by editing a monolithic reward function.
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

**Observation** `(13,)`: `position(3) + velocity(3) + orientation(3) +
distance_to_target(3) + time(1)`. `angular_velocity` is intentionally excluded to
hit `(13,)` — if you add it, bump the shape in `envs/base_env.py:OBS_DIM` and
update `_get_obs` together.

**Action space** `(4,)`: `Box(-1, 1)` = `[thrust, roll, pitch, yaw]`.

## Design decisions worth knowing

- **Thrust tilts with orientation.** Thrust is applied along the body up-axis
  rotated into the world frame (ZYX), so roll/pitch actually translate the drone.
  This is the core dynamics convention a Phase-2 model inherits.
- **Spawn sits on the crash plane (`z=0`, crash when `z<0`).** The agent must
  learn *takeoff*, not just hold — a neutral action under spawn-tilt noise sinks
  and crashes. To start airborne, change `HoverEnv._initial_state` (one line).

## Working agreement

- Build iteratively: scaffold → run → observe → adjust. Keep every change
  runnable and tested. Don't jump phases until the current one runs cleanly.
- When something is ambiguous, decide, then state what you chose and why.
- Flag any decision that's hard to reverse (frame conventions, obs/action shapes,
  the physics interface).
