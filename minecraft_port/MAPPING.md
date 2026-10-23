# Observation/action mapping ledger: EVAAA → Minecraft

Status: initial audit only. No Minecraft mapping has been validated against a running environment yet.

## Upstream contract verified from `evaaa_train/README.md`

The documented EVAAA vector observation `obs[1]` contains:

| Vector indices | Upstream meaning | Minecraft equivalent | Status |
|---|---|---|---|
| 0–3 | Food, water, temperature, health | TBD | Do not infer equivalent units or dynamics without auditing Unity implementation |
| 4–13 | 10 olfactory resource-property values | TBD | Minecraft has no direct equivalent; define and document approximation |
| 14–21 | 8 directional thermal sensors | TBD | Requires an explicit sensing model; do not silently substitute biome temperature |
| 22–31 | 10 collision sectors (100 rays grouped into sectors) | TBD | Requires raycast/block/entity sensing and calibration |
| `obs[0]` | Visual camera observation | Minecraft screenshot/frame | Image shape, frame rate, and preprocessing must be specified and tested |

The documented discrete action indices are:

| Index | EVAAA action | Minecraft mapping | Status |
|---:|---|---|---|
| 0 | None | No-op | Straightforward, needs adapter test |
| 1 | Forward | Move forward for a bounded tick interval | Approximation; movement physics differ |
| 2 | Left (turn and move) | Turn/move left for bounded ticks | Approximation; calibrate yaw and movement |
| 3 | Right (turn and move) | Turn/move right for bounded ticks | Approximation; calibrate yaw and movement |
| 4 | Eat (consume nearby resource) | Eat inventory food if available | Not equivalent until resource range, inventory, hunger and action outcomes are defined |

## Required audit before implementing the adapter

- Inspect the Unity C# source for exact variable update equations, units, bounds, sensor calculations, reward calculation, terminal conditions, and action timing.
- Inspect `evaaa_train/simple_example.py`, environment wrappers, and DQN/PPO/DreamerV3 configurations to confirm actual shapes and algorithm assumptions.
- Decide which upstream behaviors can be represented directly in Minecraft and which require explicit approximations.
- Add contract tests for observation shape, value ranges, action validity, reset behavior, terminal states, and deterministic replay where possible.
- Record every difference in this ledger. A Minecraft port is a cross-environment adaptation, not an exact reproduction of Unity results.

## Provenance

Upstream: https://github.com/cocoanlab/evaaa
Fork: https://github.com/tony-Stank3000/evaaa
The repository is licensed CC BY-NC-SA 4.0; retain attribution and comply with its terms.
