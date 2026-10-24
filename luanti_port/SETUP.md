# Setup: Luanti + Mineclonia + EVAAA

## 1. Select the game

Use **Mineclonia** for the first survival-oriented run. Create a local world with Mineclonia selected. Keep the world private for testing.

## 2. Install the mod (Windows helper)

From the cloned repository root, run this in PowerShell:

    .\luanti_port\install_mod_windows.ps1

The helper searches common Luanti data folders, lets you select the Mineclonia world, copies the mod into that world's `worldmods` directory, and adds only `evaaa_luanti` to the `secure.http_mods` allowlist in `minetest.conf` when it can find that file. It backs up an existing mod folder and config before changing them. If your world is in a portable/custom location, pass it explicitly:

    .\luanti_port\install_mod_windows.ps1 -WorldPath "C:\full\path\to\world"

The selected world folder must contain `world.mt`. Restart Luanti after installation.

For a world-specific installation, the structure is:

    <Luanti user data>/worlds/<world-name>/worldmods/evaaa_luanti/
      mod.conf
      init.lua

Restart the world/server after installing the mod.

## 3. Allow this mod to use HTTP

The mod needs Luanti's asynchronous HTTP API to communicate with the Python process. In the Luanti server configuration (minetest.conf), add:

    secure.http_mods = evaaa_luanti

This setting grants HTTP API access to the named mod; do not add unrelated mods. The bridge URL is hard-coded to loopback (127.0.0.1:8766) by default. The Python adapter and Lua mod use protocol version 2 with per-command request IDs; install both from the same repository revision. If your Luanti build blocks loopback HTTP requests, do not disable broader security controls; we will adjust the transport after checking the exact Luanti version/log.

## 4. Python dependencies and live smoke test

Use the same Python environment as EVAAA training, where `gymnasium` and `numpy` are installed. Open Mineclonia, enter the selected world, and keep it running. From the repository root, run this one-command test:

    python -m luanti_port.python.live_smoke_test

The script starts the local HTTP bridge, requests one reset and one no-op step, and prints observation shapes and reward. It waits up to 30 seconds for the mod. This is the first live integration check; a passing offline contract test alone does not prove the Lua mod works. The broker's health endpoint is `http://127.0.0.1:8766/health` while the script is running. Do not start a second bridge process on the same port.

After the live smoke test passes, use the same Python environment for training. From the `evaaa_train` directory, with the mod enabled and the world open, run one of the configured experiments:

    python train.py exp=ppo_luanti

DreamerV3 is also configured:

    python train.py exp=dreamer_v3_luanti

These commands are integration targets, not proof that training already runs end-to-end. First use a short smoke test and inspect the Luanti log. The Gymnasium environment waits for the mod to respond, then raises a timeout with guidance if no live world is connected.

## 5. Confirm the mod loaded

Check the Luanti log for "EVAAA Luanti bridge loaded". If the HTTP API is unavailable, the log will say so. Enter the world with a player; the mod only controls the configured/first eligible player and ignores other players.

## Current limitations

- No live Luanti or Mineclonia runtime is available to this repository authoring session, so the mod has not been built or smoke-tested against your installation.
- The first reset resets episode state and moves the controlled player to spawn; it does not reset terrain, inventory, mobs, or the full world.
- Resource values and synthetic sensor rasters are an experimental mapping, not a claim of exact equivalence with the Unity EVAAA sensors.
- Training configs must not be treated as proof of successful training until a live smoke test passes.
