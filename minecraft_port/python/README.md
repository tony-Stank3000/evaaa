# Python adapter

This directory contains the Python-side Gymnasium adapter. It uses gymnasium
and numpy, already used by EVAAA's training project; no new Python package
dependency is added.

## Current state

- TCP client, protocol handshake, observation/action validation, reset/step,
  and finite socket timeouts are implemented.
- The Java-side Minecraft bridge is not implemented yet. The adapter cannot
  connect to a real Minecraft world until that bridge exists.
- Rewards and terminal flags are supplied by the bridge; this code does not
  fabricate Minecraft mechanics or rewards.
- Default endpoint: 127.0.0.1:8765. Non-loopback hosts are rejected.

## Use

With the repository's existing dependencies installed:

    from evaaa_minecraft_env import MinecraftBridgeEnv

    env = MinecraftBridgeEnv(host="127.0.0.1", port=8765)
    try:
        observation, info = env.reset(seed=7)
        observation, reward, terminated, truncated, info = env.step(1)
    finally:
        env.close()

The process implementing the protocol is described in ../PROTOCOL.md. This is
an interface example, not a claim that a bridge server is already running.
