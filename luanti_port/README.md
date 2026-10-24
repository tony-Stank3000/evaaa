# EVAAA Luanti port (experimental)

This is the free-game target for the project. It runs on Luanti with a game such as Mineclonia or Minetest Game; it does not require Minecraft Java Edition, Fabric, or a Minecraft launcher.

## Selected game

**Use Mineclonia** for the first playable experiment because its survival-oriented gameplay is closer to Minecraft. The bridge uses the Luanti mod API, not private Mineclonia internals. Minetest Game remains a fallback for API/debugging, but its gameplay and resource systems differ.

## Status — do not overstate

- [x] Define a separate Luanti implementation path; the earlier minecraft_port/ Java/Fabric adapter is not used by this path.
- [x] Add a Lua mod scaffold and Python Gymnasium HTTP bridge contract.\n- [x] Correlate command responses with request IDs so late responses cannot satisfy later steps.
- [x] Keep the bridge local-only and require explicit Luanti HTTP permission.
- [ ] Run against the user's installed Luanti + Mineclonia build.
- [ ] Verify Mineclonia item groups and movement behavior in a live world.
- [ ] Calibrate sensors, resource dynamics, episode resets, and rewards against the EVAAA paper/source.
- [ ] Run end-to-end contract tests and training smoke tests.

This is **not yet a validated reproduction** of the original EVAAA experiment. Luanti is a different simulator from Unity/Minecraft, and the initial needs/sensor mappings are explicitly experimental.

## Architecture

Luanti server/mod (mods/evaaa_luanti) ⇄ request-correlated local HTTP polling bridge ⇄ Python Gymnasium environment ⇄ existing EVAAA training code.

The Python bridge listens only on 127.0.0.1:8766. The Lua mod uses Luanti's asynchronous HTTP API; enable it only for this mod with the server setting documented in SETUP.md. No remote listener, shell execution, or arbitrary code endpoint is provided.

## Setup

Follow SETUP.md. Start the Python bridge before entering the world. The first smoke test should be performed in a private local world. The current reset operation resets the controlled agent's episode counters and position; it does not reconstruct or reset the entire world.

## Files

- mods/evaaa_luanti/mod.conf: mod metadata.
- mods/evaaa_luanti/init.lua: Luanti-side polling, bounded actions, and world-derived observations.
- python/evaaa_luanti_env.py: Gymnasium environment and loopback HTTP server.
- SETUP.md: installation and configuration.
- MAPPING.md: current sensor/action assumptions and unresolved fidelity questions.

## Attribution

The original EVAAA repository is https://github.com/cocoanlab/evaaa. Retain upstream attribution and follow its CC BY-NC-SA 4.0 license for upstream-derived materials.
