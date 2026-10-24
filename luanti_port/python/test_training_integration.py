"""Checks that EVAAA's configured trainer path resolves to the real Luanti adapter."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "evaaa_train"
sys.path.insert(0, str(TRAIN_ROOT))

from envs.luanti_bridge import LuantiBridgeEnv  # noqa: E402


class LuantiTrainingIntegrationTests(unittest.TestCase):
    def test_training_import_resolves_to_bridge_class(self):
        from luanti_port.python.evaaa_luanti_env import LuantiBridgeEnv as BridgeClass

        self.assertIs(LuantiBridgeEnv, BridgeClass)

    def test_experiment_configs_target_real_bridge(self):
        for name in ("ppo_luanti.yaml", "dreamer_v3_luanti.yaml"):
            with self.subTest(config=name):
                config = yaml.safe_load((TRAIN_ROOT / "configs" / "exp" / name).read_text(encoding="utf-8"))
                self.assertEqual(config["defaults"][1], {"override /env": "luanti_bridge"})

        env_config = yaml.safe_load(
            (TRAIN_ROOT / "configs" / "env" / "luanti_bridge.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(env_config["wrapper"]["_target_"], "envs.luanti_bridge.LuantiBridgeEnv")
        self.assertEqual(env_config["wrapper"]["host"], "127.0.0.1")
        self.assertEqual(env_config["wrapper"]["port"], 8766)
        self.assertEqual(env_config["num_envs"], 1)


if __name__ == "__main__":
    unittest.main()
