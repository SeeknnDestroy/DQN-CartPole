from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch


def resolve_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_env(environment_id: str, seed: int, render_mode: str | None = None) -> gym.Env[Any, Any]:
    env = gym.make(environment_id, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def rolling_average(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    active_window = min(window, len(values))
    return float(np.mean(values[-active_window:]))


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2))
