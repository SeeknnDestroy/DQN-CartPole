from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

from .config import DQNConfig
from .model import QNetwork
from .replay_buffer import ExperienceBatch, ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: DQNConfig,
        device: torch.device,
    ) -> None:
        self.action_size = action_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.update_every = config.update_every
        self.device = device
        self._step_count = 0

        self.qnetwork_local = QNetwork(state_size, action_size, config.hidden_sizes).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, config.hidden_sizes).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.learning_rate)
        self.memory = ReplayBuffer(
            buffer_size=config.replay_buffer_size,
            batch_size=config.batch_size,
            seed=config.seed,
        )

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(self.action_size))

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()
        return int(torch.argmax(action_values, dim=1).item())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        self.memory.add(state, action, reward, next_state, done)
        self._step_count = (self._step_count + 1) % self.update_every

        if len(self.memory) < self.memory.batch_size or self._step_count != 0:
            return None

        batch = self.memory.sample(self.device)
        return self.learn(batch)

    def learn(self, batch: ExperienceBatch) -> float:
        next_action_values = self.qnetwork_target(batch.next_states).detach().max(dim=1, keepdim=True)[0]
        targets = batch.rewards + (self.gamma * next_action_values * (1 - batch.dones))
        expected = self.qnetwork_local(batch.states).gather(1, batch.actions)

        loss = F.mse_loss(expected, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return float(loss.item())

    def soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
