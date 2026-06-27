# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

`rotorenv` is a lightweight, [Gymnasium](https://gymnasium.farama.org/)-compatible
reinforcement learning environment for training autonomous quadrotor agents. Pure
Python, physics-first, built to be extended incrementally in phases.

**Current phase: Phase 3 (complete)** — mature-env architecture: enum-configured
spaces, `check_env` conformance, wrappers, vectorized-env support, four
registered variants. Built on Phase 1 (point mass) + Phase 2 (6-DOF). Still no
ML framework dependency. The environment is the product; policies come later.

We deliberately mirror how established Gymnasium envs are built
(`gym-pybullet-drones`, MiniGrid) rather than inventing conventions — see the
project memory note. Prefer enum-configurable spaces, registered task variants,
and wrappers over bespoke env code.

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
pytest                             # full test suite (currently 50 tests)
pytest tests/test_conformance.py   # Gymnasium check_env across all variants
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

## Working agreement

- Build iteratively: scaffold → run → observe → adjust. Keep every change
  runnable and tested. Don't jump phases until the current one runs cleanly.
- When something is ambiguous, decide, then state what you chose and why.
- Flag any decision that's hard to reverse (frame conventions, obs/action shapes,
  the physics interface).
