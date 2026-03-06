from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ExperienceBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=buffer_size)
        self._random = random.Random(seed)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        experience = (
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )
        self.memory.append(experience)

    def sample(self, device: torch.device) -> ExperienceBatch:
        experiences = self._random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return ExperienceBatch(
            states=torch.as_tensor(np.stack(states), dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.int64, device=device).unsqueeze(1),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            next_states=torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.memory)
