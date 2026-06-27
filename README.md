# rotorenv

A lightweight, [Gymnasium](https://gymnasium.farama.org/)-compatible reinforcement
learning environment for training autonomous drone (quadrotor) agents. Pure
Python, physics-first, and designed to be extended incrementally.

## Status

**Phase 1 (complete)**
- Point-mass physics (no drag, no rotor lag, no inertia matrix)
- `Hover-v0`: hold position at `(0, 0, 1.0)`
- No ML framework dependency — environment only

**Phase 2 (complete)**
- Full 6-DOF rigid-body physics: diagonal inertia tensor, body torques,
  quaternion attitude integration, linear + angular drag
- `Hover6DOF-v0`: same hover task on the 6-DOF backend
- Upgraded 3D renderer: a quadrotor cross that **tilts with true attitude**,
  body-up axis, and a fading trajectory trail
- The action contract is unchanged (`[thrust, roll, pitch, yaw]`), so Phase-1
  policies run on Phase-2 physics without modification

## Install

Python 3.10+ is required. Homebrew Python is *externally managed* (PEP 668), so
use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
import rotorenv

env = rotorenv.make("Hover-v0")          # also: gymnasium.make("Hover-v0")
obs, info = env.reset(seed=0)
for _ in range(100):
    action = env.action_space.sample()    # [thrust, roll, pitch, yaw] in [-1, 1]
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

Run the random-agent sanity check:

```bash
python examples/random_agent.py
```

Watch a 6-DOF episode in the live 3D window (tilting quad + trajectory trail):

```bash
python examples/render_6dof.py
```

Run the tests:

```bash
pytest
```

## Architecture

`rotorenv` keeps physics, reward, and observation as independent, swappable
concerns:

```
rotorenv/
├── core/        DroneState, DroneAction, reward terms, rotation utils
├── physics/     DronePhysics protocol + PointMassPhysics + SixDOFPhysics
├── envs/        DroneEnv base (Gym plumbing) + HoverEnv task
└── rendering/   Matplotlib 3D renderer (lazy-imported)
```

- **Physics is a `Protocol`.** `SixDOFPhysics` (Phase 2) was added purely by
  matching `step(state, action) -> DroneState`; the environment imports no
  concrete physics class. Select a backend via `physics_model="point_mass"`
  (default) or `"six_dof"`, or pass your own instance.
- **Reward is data.** A task's shaping is a list of `RewardTerm` objects summed
  by `CompositeReward`, not a hard-coded function.
- **Tasks subclass `DroneEnv`.** A task supplies the initial state, target,
  reward, and termination rule — never the step loop.

### Registered environments

| ID | Physics | Notes |
|----|---------|-------|
| `Hover-v0` | `PointMassPhysics` | Phase 1; thrust tilts with orientation |
| `Hover6DOF-v0` | `SixDOFPhysics` | Phase 2; full rigid-body, quaternion attitude |

### Domain model

| Field | Shape | Meaning |
|-------|-------|---------|
| `position` | (3,) | x, y, z [m] |
| `velocity` | (3,) | m/s |
| `orientation` | (3,) | roll, pitch, yaw [rad] |
| `angular_velocity` | (3,) | rad/s |
| `time` | scalar | elapsed [s] |

**Observation** (13,): `position(3) + velocity(3) + orientation(3) + distance_to_target(3) + time(1)`.

**Action** (4,): `[thrust, roll, pitch, yaw]` in `[-1, 1]`; thrust is rescaled to
`[0, 1]` so a neutral `0` command is a 50% throttle hover.

## License

MIT
