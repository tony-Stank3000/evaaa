# Observation/action mapping ledger: EVAAA → Minecraft

Status: upstream action/reward source audited; Minecraft-side mappings are not validated against a running world yet.

## Verified upstream observations

The default training config requests RGB visual observations plus these vector channels:

| Channel | Baseline shape | Upstream meaning | Minecraft mapping | Status |
|---|---:|---|---|---|
| rgb | 64×64×3 in this adapter contract | Visual camera frame | Minecraft rendered frame | Bridge capture and preprocessing still required |
| ev | 4 | Food, water, temperature, health | TBD | Preserve order; units and dynamics must be mapped |
| olfactory | 10 | Resource-property sensing vector | TBD | No direct Minecraft equivalent; any proxy must be documented |
| thermo | 8 | Directional thermal sensing | TBD | Do not silently substitute biome temperature |
| collision | 10 | Directional collision features | TBD | Original implementation uses raycast-based sectors; Minecraft sensing needs calibration |
| touch | 1 | Touch/contact signal | TBD | Define contact semantics explicitly |

The raw Unity agent's CollectObservations also includes position, rotation, and event counters. The selected training wrapper's enabled sensors and actual observation layout must be checked against the target experiment before training. This adapter currently exposes the five configured named vector channels plus RGB and does not claim raw-vector equivalence.

## Verified action IDs

| Index | Name | Upstream Unity behavior | Minecraft mapping | Status |
|---:|---|---|---|---|
| 0 | none | Stop movement | No-op/stop | Not yet implemented in Minecraft |
| 1 | forward | Move forward | Bounded forward input | Requires tick and speed calibration |
| 2 | left | Rotate left | Turn left | Requires yaw calibration |
| 3 | right | Rotate right | Turn right | Requires yaw calibration |
| 4 | eat | Set eat-action flag and attempt nearby resource consumption | Resource interaction | Minecraft inventory/food behavior is not equivalent without a defined rule |

Source audited: evaaa_unity/Assets/Scripts/Agent/InteroceptiveAgent.cs and ResourceEating.cs.

## Reward and episode semantics audited

The Unity reward method sets all four set points to zero, normalizes each resource-level deviation by the configured maximum for that resource, sums squared deviations, takes the square root, and returns the negative value. The Unity agent updates resource levels with configured coefficients and Time.fixedDeltaTime. Episodes can end when food, water, or enabled thermal levels leave configured ranges, when health exceeds its configured maximum (despite the log label saying "too low"), on single-trial resource consumption, or when max steps are reached.

Do not copy this reward equation blindly into Minecraft. First map resource units, ranges, coefficient dynamics, tick timing, and terminal conditions. If Minecraft-native survival rewards are preferred, record them as a separate experiment rather than calling the result a faithful port.

## Required before claiming fidelity

- Implement the Minecraft Java bridge for the actual target Minecraft version and mod loader.
- Specify frame capture, vector scaling, sensor update timing, action duration, reset semantics, and resource interactions.
- Test observation shapes/ranges and action validity; test terminal/truncation behavior and reset determinism where possible.
- Run the original Unity baseline and compare training curves and episode metrics before making reproduction claims.

## Provenance

Upstream: https://github.com/cocoanlab/evaaa  
Fork: https://github.com/tony-Stank3000/evaaa  
The upstream repository is licensed CC BY-NC-SA 4.0; retain attribution and comply with its terms.
