import numpy as np
import torch

from dqn_cartpole.agent import DQNAgent
from dqn_cartpole.config import DQNConfig
from dqn_cartpole.replay_buffer import ExperienceBatch


def make_agent() -> DQNAgent:
    config = DQNConfig(
        seed=5,
        batch_size=4,
        replay_buffer_size=32,
        update_every=1,
        hidden_sizes=(16, 16),
        warmup_steps=0,
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


def test_double_dqn_targets_use_local_argmax_and_target_values() -> None:
    agent = make_agent()
    batch = ExperienceBatch(
        states=torch.zeros((2, 4), dtype=torch.float32),
        actions=torch.tensor([[0], [1]], dtype=torch.int64),
        rewards=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        next_states=torch.zeros((2, 4), dtype=torch.float32),
        dones=torch.tensor([[0.0], [1.0]], dtype=torch.float32),
    )

    local_outputs = torch.tensor([[1.0, 5.0], [7.0, 3.0]], dtype=torch.float32)
    target_outputs = torch.tensor([[11.0, 13.0], [17.0, 19.0]], dtype=torch.float32)

    agent.qnetwork_local.forward = lambda states: local_outputs
    agent.qnetwork_target.forward = lambda states: target_outputs

    targets = agent.compute_targets(batch)

    expected = torch.tensor([[1.0 + agent.gamma * 13.0], [2.0]], dtype=torch.float32)
    assert torch.allclose(targets, expected)


def test_step_respects_warmup_before_learning() -> None:
    config = DQNConfig(
        seed=5,
        batch_size=2,
        replay_buffer_size=8,
        update_every=1,
        warmup_steps=3,
        hidden_sizes=(16, 16),
    )
    agent = DQNAgent(state_size=4, action_size=2, config=config, device=torch.device("cpu"))
    learn_calls: list[int] = []
    agent.learn = lambda batch: learn_calls.append(1) or 0.0

    state = np.zeros(4, dtype=np.float32)
    next_state = np.ones(4, dtype=np.float32)

    assert agent.step(state, 0, 1.0, next_state, False) is None
    assert agent.step(state, 1, 1.0, next_state, False) is None
    assert agent.step(state, 0, 1.0, next_state, False) == 0.0
    assert len(learn_calls) == 1
