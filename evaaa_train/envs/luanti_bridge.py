"""Import the separately maintained Luanti Gymnasium bridge for EVAAA training."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from luanti_port.python.evaaa_luanti_env import LuantiBridgeEnv  # noqa: E402

__all__ = ["LuantiBridgeEnv"]
