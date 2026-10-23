"""Contract tests for the EVAAA Minecraft Gymnasium adapter."""
import unittest
import numpy as np
from evaaa_minecraft_env import (
    ACTION_NAMES, BridgeProtocolError, MinecraftBridgeEnv, OBSERVATION_SHAPES
)

def valid_observation():
    return {
        "rgb": np.zeros((64, 64, 3), dtype=np.uint8),
        **{name: np.zeros(shape, dtype=np.float32)
           for name, shape in OBSERVATION_SHAPES.items()},
    }

class AdapterValidationTests(unittest.TestCase):
    def setUp(self):
        self.env = MinecraftBridgeEnv()

    def tearDown(self):
        self.env.close()

    def test_action_mapping_matches_upstream_discrete_actions(self):
        self.assertEqual(ACTION_NAMES, {
            0: "none", 1: "forward", 2: "left", 3: "right", 4: "eat"
        })
        self.assertEqual(self.env.action_space.n, 5)

    def test_baseline_observation_shapes_fit_declared_space(self):
        self.assertTrue(self.env.observation_space.contains(valid_observation()))

    def test_missing_observation_key_is_rejected(self):
        data = valid_observation()
        del data["thermo"]
        with self.assertRaises(BridgeProtocolError):
            self.env._parse_observation(data)

    def test_wrong_vector_shape_is_rejected(self):
        data = valid_observation()
        data["ev"] = np.zeros((3,), dtype=np.float32)
        with self.assertRaises(BridgeProtocolError):
            self.env._parse_observation(data)

    def test_non_finite_sensor_value_is_rejected(self):
        data = valid_observation()
        data["collision"][2] = np.nan
        with self.assertRaises(BridgeProtocolError):
            self.env._parse_observation(data)

    def test_rgb_out_of_range_is_rejected(self):
        data = valid_observation()
        data["rgb"] = data["rgb"].astype(np.int16)
        data["rgb"][0, 0, 0] = 256
        with self.assertRaises(BridgeProtocolError):
            self.env._parse_observation(data)

    def test_non_loopback_host_is_rejected(self):
        with self.assertRaises(ValueError):
            MinecraftBridgeEnv(host="0.0.0.0")

if __name__ == "__main__":
    unittest.main()
