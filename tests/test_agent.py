import numpy as np
import torch

from dqn_cartpole.agent import DQNAgent
from dqn_cartpole.config import DQNConfig


def make_agent() -> DQNAgent:
    config = DQNConfig(
        seed=5,
        batch_size=4,
        replay_buffer_size=32,
        update_every=1,
        hidden_sizes=(16, 16),
    )
    return DQNAgent(state_size=4, action_size=2, config=config, device=torch.device("cpu"))


def test_agent_action_within_bounds() -> None:
    agent = make_agent()
    action = agent.act(np.zeros(4, dtype=np.float32), epsilon=0.0)
    assert action in {0, 1}


def test_agent_learn_smoke() -> None:
    agent = make_agent()
    for step in range(8):
        state = np.full(4, step, dtype=np.float32)
        next_state = state + 0.5
        agent.memory.add(state, step % 2, 1.0, next_state, False)

    batch = agent.memory.sample(torch.device("cpu"))
    loss = agent.learn(batch)

    assert isinstance(loss, float)
    assert loss >= 0.0
