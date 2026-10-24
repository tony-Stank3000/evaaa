# Setup: Luanti + Mineclonia + EVAAA

## 1. Select the game

Use **Mineclonia** for the first survival-oriented run. Create a local world with Mineclonia selected. Keep the world private for testing.

## 2. Install the mod

Copy the directory mods/evaaa_luanti/ into the world's worldmods/ directory, or install it as a normal Luanti mod and enable it for the target world. The directory must contain mod.conf and init.lua.

For a world-specific installation, the structure is:

    <Luanti user data>/worlds/<world-name>/worldmods/evaaa_luanti/
      mod.conf
      init.lua

Restart the world/server after installing the mod.

## 3. Allow this mod to use HTTP

The mod needs Luanti's asynchronous HTTP API to communicate with the Python process. In the Luanti server configuration (minetest.conf), add:

    secure.http_mods = evaaa_luanti

This setting grants HTTP API access to the named mod; do not add unrelated mods. The bridge URL is hard-coded to loopback (127.0.0.1:8766) by default. If your Luanti build blocks loopback HTTP requests, do not disable broader security controls; we will adjust the transport after checking the exact Luanti version/log.

## 4. Python dependencies

Use the same Python environment as EVAAA training, where gymnasium and numpy are installed. Start the environment/bridge from the repository root:

    python -m luanti_port.python.evaaa_luanti_env

This starts a loopback HTTP service on port 8766. Keep it running while the world is open. The module includes a small server entry point for transport smoke checks; training integration is a separate step.

## 5. Confirm the mod loaded

Check the Luanti log for "EVAAA Luanti bridge loaded". If the HTTP API is unavailable, the log will say so. Enter the world with a player; the mod only controls the configured/first eligible player and ignores other players.

## Current limitations

- No live Luanti or Mineclonia runtime is available to this repository authoring session, so the mod has not been built or smoke-tested against your installation.
- The first reset resets episode state and moves the controlled player to spawn; it does not reset terrain, inventory, mobs, or the full world.
- Resource values and synthetic sensor rasters are an experimental mapping, not a claim of exact equivalence with the Unity EVAAA sensors.
- Training configs must not be treated as proof of successful training until a live smoke test passes.
