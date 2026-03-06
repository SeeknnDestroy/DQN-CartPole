from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_sizes: tuple[int, int]) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_sizes
        self.layers = nn.Sequential(
            nn.Linear(state_size, first_hidden),
            nn.ReLU(),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Linear(second_hidden, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
