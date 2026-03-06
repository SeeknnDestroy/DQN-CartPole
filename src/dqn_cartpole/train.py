from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dataclasses import replace
import torch

from .agent import DQNAgent
from .config import DQNConfig
from .utils import ensure_parent_dir, make_env, resolve_device, rolling_average, set_global_seed, write_json


def build_parser() -> argparse.ArgumentParser:
    defaults = DQNConfig()
    parser = argparse.ArgumentParser(description="Train a DQN agent on CartPole-v1.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--tau", type=float, default=defaults.tau)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--replay-buffer-size", type=int, default=defaults.replay_buffer_size)
    parser.add_argument("--update-every", type=int, default=defaults.update_every)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=defaults.epsilon_decay)
    parser.add_argument("--hidden-sizes", type=int, nargs=2, metavar=("H1", "H2"), default=defaults.hidden_sizes)
    parser.add_argument("--moving-average-window", type=int, default=defaults.moving_average_window)
    parser.add_argument("--solved-threshold", type=float, default=defaults.solved_threshold)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--checkpoint-path", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--metrics-path", type=Path, default=defaults.metrics_path)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    return parser


def config_from_args(args: argparse.Namespace) -> DQNConfig:
    config = DQNConfig()
    return replace(
        config,
        seed=args.seed,
        episodes=args.episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        replay_buffer_size=args.replay_buffer_size,
        update_every=args.update_every,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_sizes=tuple(args.hidden_sizes),
        moving_average_window=args.moving_average_window,
        solved_threshold=args.solved_threshold,
        eval_episodes=args.eval_episodes,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        device=args.device,
        log_every=args.log_every,
    )


def save_checkpoint(
    path: Path,
    agent: DQNAgent,
    config: DQNConfig,
    summary: dict[str, Any],
) -> None:
    ensure_parent_dir(path)
    checkpoint = {
        "config": config.to_dict(),
        "summary": summary,
        "model_state_dict": agent.qnetwork_local.state_dict(),
        "target_state_dict": agent.qnetwork_target.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def train(config: DQNConfig) -> dict[str, Any]:
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    env = make_env(config.environment_id, config.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    agent = DQNAgent(state_size=state_size, action_size=action_size, config=config, device=device)

    episode_rewards: list[float] = []
    moving_averages: list[float] = []
    losses: list[float] = []
    epsilon = config.epsilon_start
    best_moving_average = float("-inf")

    for episode in range(1, config.episodes + 1):
        state, _ = env.reset(seed=config.seed + episode)
        total_reward = 0.0

        for _ in range(config.max_steps):
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)
            total_reward += float(reward)
            state = next_state
            if done:
                break

        episode_rewards.append(total_reward)
        moving_average = rolling_average(episode_rewards, config.moving_average_window)
        moving_averages.append(moving_average)
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)

        if moving_average >= best_moving_average:
            best_moving_average = moving_average
            save_checkpoint(
                config.checkpoint_path,
                agent,
                config,
                {
                    "checkpoint_type": "best",
                    "episode": episode,
                    "episode_reward": total_reward,
                    "moving_average_reward": moving_average,
                    "epsilon": epsilon,
                },
            )

        if episode == 1 or episode % config.log_every == 0 or episode == config.episodes:
            print(
                f"Episode {episode:04d} | reward={total_reward:.1f} "
                f"| moving_avg={moving_average:.1f} | epsilon={epsilon:.3f}"
            )

    env.close()

    metrics = {
        "config": config.to_dict(),
        "device": str(device),
        "state_size": state_size,
        "action_size": action_size,
        "episodes_completed": len(episode_rewards),
        "episode_rewards": episode_rewards,
        "moving_average_rewards": moving_averages,
        "best_episode_reward": max(episode_rewards),
        "best_moving_average_reward": best_moving_average,
        "mean_reward": mean(episode_rewards),
        "final_epsilon": epsilon,
        "loss_count": len(losses),
        "last_loss": losses[-1] if losses else None,
        "checkpoint_path": str(config.checkpoint_path),
    }
    write_json(config.metrics_path, metrics)
    print(f"Saved best checkpoint to {config.checkpoint_path}")
    print(f"Saved metrics to {config.metrics_path}")
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
