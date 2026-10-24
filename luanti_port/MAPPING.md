# EVAAA-to-Luanti mapping ledger

Status: initial implementation scaffold only. All mappings below require live validation and calibration.

## Actions

| ID | Name | Luanti implementation | Status |
|---:|---|---|---|
| 0 | none | Stop horizontal motion | Implemented in scaffold; verify live |
| 1 | forward | Apply bounded velocity in player yaw direction | Implemented in scaffold; verify collision/movement |
| 2 | left | Rotate player yaw by a fixed increment | Implemented in scaffold; calibrate |
| 3 | right | Rotate player yaw by a fixed increment | Implemented in scaffold; calibrate |
| 4 | eat | Attempt to consume an inventory item carrying the food group; if no matching item exists, no consumption occurs | Game-specific item-group support must be verified |

## Observations

- Lua HTTP wire rgb [64,64,3]: not a screenshot. It is an explicitly synthetic egocentric voxel raster derived from nearby node names/content IDs. The Python adapter converts it to channel-first [3,64,64] for the existing EVAAA CNN pipeline. Do not describe it as camera imagery.
- ev [4]: food, water, temperature deviation, health. Food/water are experimental internal meters; health comes from the game player. They are not calibrated to the paper's physiological units.
- olfactory [10]: experimental resource-proximity features derived from sampled node categories. No direct equivalent to the original sensor.
- thermo [8]: experimental directional environment features; not a direct physiological temperature sensor.
- collision [10]: directional obstacle sampling in front/around the player; requires live calibration.
- touch [1]: contact/grounded proxy where available; verify against the installed game API.

## Reward, reset, and fidelity

The initial Luanti experiment uses a separately versioned environment reward based on its experimental need variables and state transitions. It must not be represented as the original EVAAA reward until the paper's equations, units, update frequency, and terminal semantics are implemented and verified. Reset moves the player to spawn and resets episode counters; the world itself is not reset.

The Unity source's original reward and physiological dynamics remain the reference implementation. This port is a research adaptation, not an exact simulator reproduction.
