# EVAAA Minecraft port (experimental)

This directory contains the initial Python-side bridge client and a protocol contract. It is not yet a complete, live Minecraft port: a Minecraft Java-side implementation is still required.

## Implementation status

- [x] Audit upstream action IDs in InteroceptiveAgent.cs: 0 no-op, 1 forward, 2 left, 3 right, 4 eat.
- [x] Audit upstream reward equation: negative Euclidean distance from a zero set point, normalized by each configured maximum for the four resource levels.
- [x] Implement a Gymnasium TCP client with protocol handshake, finite timeout, loopback-only default, strict observation/action validation, and baseline observation keys/shapes.
- [x] Add trainer environment config and PPO/DreamerV3 experiment configs.
- [x] Add contract tests for action mapping, observation space, malformed observations, non-finite values, pixel bounds, and loopback policy.
- [ ] Implement Minecraft Java-side bridge for the chosen Minecraft version/mod loader.
- [ ] Map real game state to EVAAA variables and sensors; calibrate units, tick duration, and movement.
- [ ] Run contract tests and a live Minecraft smoke test; compare metrics against the original baseline.

## Reproduction rules

1. Preserve upstream evaaa_train/ and evaaa_unity/ unless a required change is documented.
2. Keep Minecraft-specific behavior isolated from the Unity environment.
3. Do not infer physiological units from Minecraft values. Record every mapping, unit, update frequency, and approximation in MAPPING.md.
4. Keep baseline algorithm/config separate from Minecraft-specific experiments.
5. Do not claim paper reproduction until baseline results are compared with the publication.

## Architecture

Minecraft Java (private local world)
→ Minecraft Java bridge (not yet implemented)
→ JSON-lines TCP on loopback
→ Python Gymnasium adapter (implemented)
→ EVAAA trainer configs (PPO / DreamerV3)
→ transitions, checkpoints, metrics

## Current files

- PROTOCOL.md: version-1 wire contract.
- python/evaaa_minecraft_env.py: standalone adapter module.
- python/test_evaaa_minecraft_env.py: contract validation tests.
- evaaa_train/envs/minecraft_bridge.py: adapter wired into EVAAA training import path.
- evaaa_train/configs/env/minecraft_bridge.yaml: environment configuration.
- evaaa_train/configs/exp/ppo_minecraft.yaml and dreamer_v3_minecraft.yaml: separate training configs.
- MAPPING.md: verified upstream behavior and unresolved Minecraft mappings.

## Training entry points

From the evaaa_train directory, after the Minecraft bridge is implemented and running:

- PPO: python train.py exp=ppo_minecraft
- DreamerV3: python train.py exp=dreamer_v3_minecraft

These are configured entry points, not evidence of a successful run. The custom DQN script currently constructs Unity ML-Agents directly and has not been adapted to the Minecraft bridge.

## Runtime boundary

Use a private test world. Keep the bridge bound to loopback, validate actions, bound game ticks per action, and stop safely on disconnect. Do not expose an unauthenticated command-execution endpoint to the network.
