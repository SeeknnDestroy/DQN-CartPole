from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from dqn_cartpole.evaluate import evaluate_policy, load_agent
from dqn_cartpole.utils import ensure_parent_dir, resolve_device, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a trained CartPole checkpoint to GIF.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser


def export_gif(
    checkpoint_path: Path,
    output_path: Path,
    seed: int,
    episode: int,
    fps: int,
    frame_skip: int,
    device_name: str = "auto",
) -> dict[str, float]:
    set_global_seed(seed)
    device = resolve_device(device_name)
    agent, config = load_agent(checkpoint_path, device)
    metrics = evaluate_policy(
        agent=agent,
        environment_id=config.environment_id,
        episodes=episode,
        seed=seed,
        success_episode_threshold=config.success_episode_threshold,
        solved_threshold=config.solved_threshold,
        render_mode="rgb_array",
    )
    frames = metrics.pop("frames")
    sampled_frames = frames[:: max(1, frame_skip)]
    gif_frames = [np.asarray(frame) for frame in sampled_frames]
    ensure_parent_dir(output_path)
    imageio.mimsave(output_path, gif_frames, fps=fps)
    print(
        f"Exported {len(gif_frames)} frames to {output_path} "
        f"(mean_reward={metrics['mean_reward']:.1f})"
    )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    export_gif(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        episode=args.episode,
        fps=args.fps,
        frame_skip=args.frame_skip,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
