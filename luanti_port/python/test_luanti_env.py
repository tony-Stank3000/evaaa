"""Contract tests for the Luanti Gymnasium bridge; no Luanti installation required."""
from __future__ import annotations

import json
import socket
import threading
import time
import unittest
from urllib.request import Request, urlopen

import numpy as np

from luanti_port.python.evaaa_luanti_env import (
    ACTION_NAMES, BridgeProtocolError, LuantiBridgeEnv, PROTOCOL_VERSION,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def valid_payload():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "observation": {
            "rgb": [[[0, 12, 255] for _ in range(64)] for _ in range(64)],
            "ev": [1, 1, 0, 1],
            "olfactory": [0] * 10,
            "thermo": [0] * 8,
            "collision": [0] * 10,
            "touch": [0],
        },
        "reward": -0.1,
        "terminated": False,
        "truncated": False,
        "info": {"simulator": "luanti"},
    }


class LuantiBridgeContractTests(unittest.TestCase):
    def setUp(self):
        self.env = LuantiBridgeEnv(port=free_port(), timeout=2.0)

    def tearDown(self):
        self.env.close()

    def respond_to_next_command(self, payload):
        # Exercise the real HTTP protocol used by the Lua mod, not private broker methods.
        base = f"http://127.0.0.1:{self.env.port}"
        deadline = time.monotonic() + 1.0
        command = None
        while time.monotonic() < deadline:
            with urlopen(base + "/api/next", timeout=0.5) as response:
                candidate = json.loads(response.read().decode("utf-8"))
            if candidate.get("type") in ("reset", "step"):
                command = candidate
                break
            time.sleep(0.005)
        if command is None:
            self.fail("Gymnasium command was not available over HTTP.")
        request = Request(
            base + "/api/observation",
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=1.0) as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(json.loads(response.read().decode("utf-8"))["ok"])

    def test_action_mapping(self):
        self.assertEqual(ACTION_NAMES, {0: "none", 1: "forward", 2: "left", 3: "right", 4: "eat"})

    def test_reset_returns_declared_shapes(self):
        output = {}
        worker = threading.Thread(target=lambda: output.setdefault("value", self.env.reset(seed=7)))
        worker.start()
        self.respond_to_next_command(valid_payload())
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        obs, _info = output["value"]
        self.assertEqual(obs["rgb"].shape, (3, 64, 64))
        self.assertEqual(obs["rgb"][0, 0, 0], 0)
        self.assertEqual(obs["rgb"][1, 0, 0], 12)
        self.assertEqual(obs["rgb"][2, 0, 0], 255)
        self.assertEqual(obs["ev"].shape, (4,))
        self.assertEqual(obs["olfactory"].shape, (10,))
        self.assertTrue(self.env.observation_space.contains(obs))

    def test_step_returns_real_bridge_reward_and_flags(self):
        output = {}
        worker = threading.Thread(target=lambda: output.setdefault("value", self.env.step(1)))
        worker.start()
        self.respond_to_next_command(valid_payload())
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        _obs, reward, terminated, truncated, _info = output["value"]
        self.assertEqual(reward, -0.1)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_timeout_clears_unconsumed_command(self):
        env = LuantiBridgeEnv(port=free_port(), timeout=0.05)
        try:
            with self.assertRaises(TimeoutError):
                env.reset()
            with env._broker.condition:
                self.assertIsNone(env._broker.pending)
        finally:
            env.close()

    def test_rejects_non_loopback_host(self):
        with self.assertRaises(ValueError):
            LuantiBridgeEnv(host="0.0.0.0", port=free_port())

    def test_rejects_invalid_action(self):
        with self.assertRaises(ValueError):
            self.env.step(9)


if __name__ == "__main__":
    unittest.main()
