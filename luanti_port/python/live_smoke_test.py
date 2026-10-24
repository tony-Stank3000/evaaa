"""One-reset/one-step live smoke test for an already-running Luanti world."""
from __future__ import annotations

import sys

from luanti_port.python.evaaa_luanti_env import LuantiBridgeEnv


def main() -> int:
    env = LuantiBridgeEnv(timeout=30.0)
    try:
        print("Waiting for the EVAAA Luanti mod at 127.0.0.1:8766 ...")
        observation, info = env.reset(seed=0)
        print("RESET OK")
        print("Observation shapes:", {key: value.shape for key, value in observation.items()})
        print("Info:", info)
        observation, reward, terminated, truncated, info = env.step(0)
        print("STEP OK")
        print("Reward:", reward)
        print("Terminated:", terminated, "Truncated:", truncated)
        print("Observation shapes:", {key: value.shape for key, value in observation.items()})
        print("Info:", info)
        return 0
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "Check that Mineclonia is open, the player has joined the world, "
            "the mod is enabled, and secure.http_mods includes evaaa_luanti.",
            file=sys.stderr,
        )
        return 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
