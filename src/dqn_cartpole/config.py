from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DQNConfig:
    environment_id: str = "CartPole-v1"
    seed: int = 7
    episodes: int = 300
    max_steps: int = 500
    learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 1e-3
    batch_size: int = 64
    replay_buffer_size: int = 100_000
    update_every: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    hidden_sizes: tuple[int, int] = (64, 64)
    moving_average_window: int = 20
    solved_threshold: float = 475.0
    eval_episodes: int = 20
    checkpoint_path: Path = field(default_factory=lambda: Path("artifacts/checkpoints/cartpole_dqn.pt"))
    metrics_path: Path = field(default_factory=lambda: Path("artifacts/train_metrics.json"))
    device: str = "auto"
    log_every: int = 10

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["metrics_path"] = str(self.metrics_path)
        payload["hidden_sizes"] = list(self.hidden_sizes)
        return payload
