from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from .agent import DQNAgent
from .config import DQNConfig
from .utils import make_env, resolve_device, set_global_seed, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint on CartPole-v1.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics-path", type=Path, default=None)
    return parser


def load_agent(checkpoint_path: Path, device: torch.device) -> tuple[DQNAgent, DQNConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint["config"]
    config = DQNConfig(
        environment_id=checkpoint_config["environment_id"],
        seed=checkpoint_config["seed"],
        episodes=checkpoint_config["episodes"],
        max_steps=checkpoint_config["max_steps"],
        learning_rate=checkpoint_config["learning_rate"],
        gamma=checkpoint_config["gamma"],
        tau=checkpoint_config["tau"],
        batch_size=checkpoint_config["batch_size"],
        replay_buffer_size=checkpoint_config["replay_buffer_size"],
        update_every=checkpoint_config["update_every"],
        warmup_steps=checkpoint_config["warmup_steps"],
        gradient_clip_norm=checkpoint_config["gradient_clip_norm"],
        epsilon_start=checkpoint_config["epsilon_start"],
        epsilon_end=checkpoint_config["epsilon_end"],
        epsilon_decay=checkpoint_config["epsilon_decay"],
        hidden_sizes=tuple(checkpoint_config["hidden_sizes"]),
        success_episode_threshold=checkpoint_config["success_episode_threshold"],
        moving_average_window=checkpoint_config["moving_average_window"],
        solved_threshold=checkpoint_config["solved_threshold"],
        validation_interval=checkpoint_config["validation_interval"],
        validation_episodes=checkpoint_config["validation_episodes"],
        eval_episodes=checkpoint_config["eval_episodes"],
        checkpoint_path=Path(checkpoint_config["checkpoint_path"]),
        metrics_path=Path(checkpoint_config["metrics_path"]),
        device=checkpoint_config["device"],
        log_every=checkpoint_config["log_every"],
    )

    env = make_env(config.environment_id, config.seed)
    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)
    env.close()

    agent = DQNAgent(state_size=state_size, action_size=action_size, config=config, device=device)
    agent.qnetwork_local.load_state_dict(checkpoint["model_state_dict"])
    agent.qnetwork_target.load_state_dict(checkpoint["target_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.qnetwork_local.eval()
    return agent, config


def evaluate_policy(
    agent: DQNAgent,
    environment_id: str,
    episodes: int,
    seed: int,
    success_episode_threshold: float,
    solved_threshold: float,
    render_mode: str | None = None,
) -> dict[str, Any]:
    env = make_env(environment_id, seed, render_mode=render_mode)
    rewards: list[float] = []
    rendered_frames: list[np.ndarray] = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        done = False

        while not done:
            if render_mode == "rgb_array":
                rendered_frames.append(env.render())
            action = agent.act(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            state = next_state

        rewards.append(total_reward)

    env.close()

    metrics = {
        "episodes": episodes,
        "mean_reward": mean(rewards),
        "std_reward": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "success_rate": sum(reward >= success_episode_threshold for reward in rewards) / len(rewards),
        "success_episode_threshold": success_episode_threshold,
        "solved": mean(rewards) >= solved_threshold,
        "solved_threshold": solved_threshold,
        "episode_rewards": rewards,
    }
    if render_mode == "rgb_array":
        metrics["frames"] = rendered_frames
    return metrics


def evaluate_checkpoint(
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    device_name: str = "auto",
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    set_global_seed(seed)
    device = resolve_device(device_name)
    agent, config = load_agent(checkpoint_path, device)
    metrics = {
        "checkpoint_path": str(checkpoint_path),
        **evaluate_policy(
            agent=agent,
            environment_id=config.environment_id,
            episodes=episodes,
            seed=seed,
            success_episode_threshold=config.success_episode_threshold,
            solved_threshold=config.solved_threshold,
        ),
    }
    if metrics_path is not None:
        write_json(metrics_path, metrics)
    print(
        f"Evaluation over {episodes} episodes | mean={metrics['mean_reward']:.1f} "
        f"| std={metrics['std_reward']:.1f} | success_rate={metrics['success_rate']:.2%}"
    )
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        device_name=args.device,
        metrics_path=args.metrics_path,
    )


if __name__ == "__main__":
    main()
