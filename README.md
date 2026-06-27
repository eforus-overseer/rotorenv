# rotorenv

A lightweight, [Gymnasium](https://gymnasium.farama.org/)-compatible reinforcement
learning environment for training autonomous drone (quadrotor) agents. Pure
Python, physics-first, and designed to be extended incrementally.

## Status — Phase 1

- Point-mass physics (no drag, no rotor lag, no inertia matrix)
- `Hover-v0`: hold position at `(0, 0, 1.0)`
- Matplotlib 3D rendering
- No ML framework dependency — environment only

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

Run the tests:

```bash
pytest
```

## Architecture

`rotorenv` keeps physics, reward, and observation as independent, swappable
concerns:

```
rotorenv/
├── core/        DroneState, DroneAction, composable reward terms
├── physics/     DronePhysics protocol + PointMassPhysics (swap-in point)
├── envs/        DroneEnv base (Gym plumbing) + HoverEnv task
└── rendering/   Matplotlib 3D renderer (lazy-imported)
```

- **Physics is a `Protocol`.** A Phase-2 6-DOF model only needs a matching
  `step()`; the environment never imports a concrete physics class directly.
- **Reward is data.** A task's shaping is a list of `RewardTerm` objects summed
  by `CompositeReward`, not a hard-coded function.
- **Tasks subclass `DroneEnv`.** A task supplies the initial state, target,
  reward, and termination rule — never the step loop.

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
