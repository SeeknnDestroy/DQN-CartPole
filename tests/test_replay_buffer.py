import numpy as np
import torch

from dqn_cartpole.replay_buffer import ReplayBuffer


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(buffer_size=32, batch_size=4, seed=3)
    for step in range(8):
        state = np.full(4, step, dtype=np.float32)
        next_state = state + 1
        buffer.add(state, step % 2, 1.0, next_state, done=step % 3 == 0)

    batch = buffer.sample(torch.device("cpu"))

    assert batch.states.shape == (4, 4)
    assert batch.actions.shape == (4, 1)
    assert batch.rewards.shape == (4, 1)
    assert batch.next_states.shape == (4, 4)
    assert batch.dones.shape == (4, 1)
    assert batch.states.dtype == torch.float32
