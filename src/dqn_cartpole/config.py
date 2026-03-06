from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DQNConfig:
    environment_id: str = "CartPole-v1"
    seed: int = 7
    episodes: int = 1500
    max_steps: int = 500
    learning_rate: float = 5e-4
    gamma: float = 0.99
    tau: float = 5e-3
    batch_size: int = 64
    replay_buffer_size: int = 100_000
    update_every: int = 4
    warmup_steps: int = 1_000
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.997
    hidden_sizes: tuple[int, int] = (128, 128)
    success_episode_threshold: float = 200.0
    moving_average_window: int = 20
    solved_threshold: float = 475.0
    validation_interval: int = 50
    validation_episodes: int = 20
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
