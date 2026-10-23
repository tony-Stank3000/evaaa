"""Gymnasium adapter for an EVAAA-compatible Minecraft Java bridge.

This module validates a real bridge contract; it does not simulate Minecraft.
It uses gymnasium and numpy, dependencies already used by EVAAA training.
"""
from __future__ import annotations
import json
import socket
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROTOCOL_VERSION = 1
ACTION_NAMES = {0: "none", 1: "forward", 2: "left", 3: "right", 4: "eat"}
OBSERVATION_SHAPES = {
    "ev": (4,), "olfactory": (10,), "thermo": (8,),
    "collision": (10,), "touch": (1,),
}
DEFAULT_RGB_SHAPE = (64, 64, 3)

class BridgeProtocolError(RuntimeError):
    """Malformed or incompatible bridge message."""

class MinecraftBridgeEnv(gym.Env):
    """Gymnasium client for a loopback Minecraft Java bridge using JSON lines.

    Every message includes protocol_version=1. Reset replies with type=observation;
    step replies with type=step_result and actual reward/terminal flags.
    """

    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 timeout: float = 5.0, rgb_shape: tuple[int, int, int] = DEFAULT_RGB_SHAPE,
                 max_line_bytes: int = 8 * 1024 * 1024) -> None:
        super().__init__()
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Bridge connections must use loopback by default.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if len(rgb_shape) != 3 or any(int(v) <= 0 for v in rgb_shape):
            raise ValueError("rgb_shape must contain three positive dimensions")
        if max_line_bytes < 1024:
            raise ValueError("max_line_bytes must be at least 1024 bytes")
        self.host, self.port, self.timeout = host, int(port), float(timeout)
        self.rgb_shape = tuple(int(v) for v in rgb_shape)
        self.max_line_bytes = int(max_line_bytes)
        self._socket: socket.socket | None = None
        self._reader = None
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, shape=self.rgb_shape, dtype=np.uint8),
            **{key: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
               for key, shape in OBSERVATION_SHAPES.items()},
        })

    def _connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        self._socket = sock
        self._reader = sock.makefile("rb")
        self._send({"type": "hello", "protocol_version": PROTOCOL_VERSION,
                    "client": "evaaa_minecraft_env"})
        response = self._receive()
        if response.get("type") != "hello_ack":
            self.close()
            raise BridgeProtocolError("Bridge did not return hello_ack.")
        if response.get("protocol_version") != PROTOCOL_VERSION:
            self.close()
            raise BridgeProtocolError("Bridge protocol version mismatch.")

    def _send(self, message: dict[str, Any]) -> None:
        if self._socket is None:
            raise BridgeProtocolError("Bridge socket is not connected.")
        payload = (json.dumps(message, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(payload) > self.max_line_bytes:
            raise BridgeProtocolError("Outgoing message exceeds max_line_bytes.")
        self._socket.sendall(payload)

    def _receive(self) -> dict[str, Any]:
        if self._reader is None:
            raise BridgeProtocolError("Bridge reader is not connected.")
        line = self._reader.readline(self.max_line_bytes + 1)
        if not line:
            raise BridgeProtocolError("Bridge closed the connection.")
        if len(line) > self.max_line_bytes or not line.endswith(b"\n"):
            raise BridgeProtocolError("Bridge response is oversized or missing newline.")
        try:
            message = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BridgeProtocolError("Bridge response is not valid UTF-8 JSON.") from exc
        if not isinstance(message, dict):
            raise BridgeProtocolError("Bridge response must be a JSON object.")
        if message.get("protocol_version") != PROTOCOL_VERSION:
            raise BridgeProtocolError("Missing or incompatible protocol_version.")
        return message

    def _parse_observation(self, value: Any) -> dict[str, np.ndarray]:
        if not isinstance(value, dict):
            raise BridgeProtocolError("observation must be a JSON object.")
        expected = {"rgb", *OBSERVATION_SHAPES.keys()}
        if set(value) != expected:
            raise BridgeProtocolError(
                f"Observation keys mismatch; missing={sorted(expected-set(value))}, "
                f"extra={sorted(set(value)-expected)}"
            )
        try:
            rgb = np.asarray(value["rgb"])
            if rgb.shape != self.rgb_shape:
                raise BridgeProtocolError(f"rgb shape must be {self.rgb_shape}, got {rgb.shape}")
            if not np.issubdtype(rgb.dtype, np.number):
                raise BridgeProtocolError("rgb must contain numeric pixels.")
            if not np.isfinite(rgb).all() or np.any(rgb < 0) or np.any(rgb > 255):
                raise BridgeProtocolError("rgb values must be finite and in [0,255].")
            if not np.equal(rgb, np.floor(rgb)).all():
                raise BridgeProtocolError("rgb values must be integer-valued.")
            result = {"rgb": rgb.astype(np.uint8, copy=False)}
            for key, shape in OBSERVATION_SHAPES.items():
                arr = np.asarray(value[key], dtype=np.float32)
                if arr.shape != shape:
                    raise BridgeProtocolError(f"{key} shape must be {shape}, got {arr.shape}")
                if not np.isfinite(arr).all():
                    raise BridgeProtocolError(f"{key} contains NaN or infinity.")
                result[key] = arr
        except (TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, BridgeProtocolError):
                raise
            raise BridgeProtocolError("Observation contains invalid numeric data.") from exc
        if not self.observation_space.contains(result):
            raise BridgeProtocolError("Observation does not fit the declared Gymnasium space.")
        return result

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._connect()
        request: dict[str, Any] = {"type": "reset", "protocol_version": PROTOCOL_VERSION}
        if seed is not None:
            request["seed"] = int(seed)
        if options:
            request["options"] = options
        self._send(request)
        response = self._receive()
        if response.get("type") != "observation":
            raise BridgeProtocolError("Reset must return an observation message.")
        observation = self._parse_observation(response.get("observation"))
        info = response.get("info", {})
        if not isinstance(info, dict):
            raise BridgeProtocolError("info must be a JSON object.")
        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action; expected integer ID 0..4.")
        if self._socket is None:
            raise BridgeProtocolError("Call reset() before step().")
        self._send({"type": "step", "protocol_version": PROTOCOL_VERSION,
                    "action": int(action), "action_name": ACTION_NAMES[int(action)]})
        response = self._receive()
        if response.get("type") != "step_result":
            raise BridgeProtocolError("Step must return a step_result message.")
        observation = self._parse_observation(response.get("observation"))
        try:
            reward = float(response["reward"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BridgeProtocolError("step_result must contain a numeric reward.") from exc
        if not np.isfinite(reward):
            raise BridgeProtocolError("reward must be finite.")
        terminated, truncated = response.get("terminated"), response.get("truncated")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise BridgeProtocolError("terminated and truncated must be booleans.")
        info = response.get("info", {})
        if not isinstance(info, dict):
            raise BridgeProtocolError("info must be a JSON object.")
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        reader, sock = self._reader, self._socket
        self._reader = None
        self._socket = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
