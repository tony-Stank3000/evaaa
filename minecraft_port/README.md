# EVAAA Minecraft port (experimental)

This directory is an **adapter boundary**, not a claim that EVAAA has already been ported or reproduced in Minecraft.

## Non-negotiable reproduction rules

1. Keep upstream `evaaa_train/` and `evaaa_unity/` unchanged unless a change is required to make the original code runnable; document every such change.
2. Keep Minecraft-specific code in this directory.
3. Do not silently map Minecraft values to EVAAA physiological variables. Record each mapping, unit, update frequency, and any approximation in `MAPPING.md`.
4. Keep the original training algorithm/configuration as the baseline. Any algorithm changes belong in a separately named experimental configuration.
5. Do not claim paper reproduction until the original baseline has been run and its metrics compared with the publication.

## Target architecture

```text
Minecraft Java (private local world/server)
        |
        v
Minecraft adapter (observations, actions, episode reset)
        |
        v
EVAAA-compatible environment contract
        |
        v
Original EVAAA training/evaluation code (unchanged baseline)
        |
        v
Recorded transitions, checkpoints, metrics, provenance
```

The adapter must expose observations and actions only after the exact source environment's interface has been audited. Until then, this is an integration plan and contract, not a drop-in SheepRL environment.

## Next steps

- Audit `evaaa_train/README.md`, its configs, environment registration and agent implementations.
- Audit Unity observation/action definitions and internal-variable update rules.
- Record the original observation/action/reward/state contract.
- Implement the smallest Minecraft equivalent and write contract tests.
- Run original Unity baseline first where feasible, then run Minecraft experiments with the same algorithm and clearly report environment-induced differences.

## Runtime boundary

Use a private test world. The Minecraft client/bridge should bind to loopback by default, validate incoming actions, apply per-step timeouts, and stop safely on disconnect. Never expose a command-execution endpoint to the network.
