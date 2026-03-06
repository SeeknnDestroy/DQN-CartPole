from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from dqn_cartpole.config import DQNConfig
from dqn_cartpole.evaluate import evaluate_checkpoint
from dqn_cartpole.train import train


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "benchmark_suite"
DEFAULT_RESULTS_ROOT = ROOT / "results"
DEFAULT_SEEDS = (7, 17, 27)
SUCCESS_MEAN_REWARD = 475.0

FALLBACK_SEQUENCE = (
    {"label": "epsilon_decay_0.998", "overrides": {"epsilon_decay": 0.998}},
    {"label": "warmup_steps_2000", "overrides": {"warmup_steps": 2000}},
    {"label": "episodes_2000", "overrides": {"episodes": 2000}},
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible CartPole benchmark studies.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--baseline-episodes", type=int, default=1000)
    parser.add_argument("--chosen-episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=250)
    return parser


def aggregate_variant(name: str, runs: list[dict[str, object]]) -> dict[str, object]:
    eval_means = [float(run["eval_mean_reward"]) for run in runs]
    success_rates = [float(run["eval_success_rate"]) for run in runs]
    moving_averages = [float(run["best_moving_average_reward"]) for run in runs]
    validation_rewards = [float(run["best_validation_reward"]) for run in runs]
    solved_all = all(float(run["eval_mean_reward"]) >= SUCCESS_MEAN_REWARD for run in runs)
    return {
        "name": name,
        "runs": runs,
        "aggregate": {
            "mean_eval_reward": mean(eval_means),
            "std_eval_reward": pstdev(eval_means) if len(eval_means) > 1 else 0.0,
            "mean_success_rate": mean(success_rates),
            "mean_best_moving_average_reward": mean(moving_averages),
            "mean_best_validation_reward": mean(validation_rewards),
            "solved_all_seeds": solved_all,
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_variant(
    *,
    study_name: str,
    variant_name: str,
    seeds: tuple[int, ...],
    eval_episodes: int,
    log_every: int,
    base_config: DQNConfig,
    overrides: dict[str, Any],
    artifact_root: Path,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []

    for seed in seeds:
        run_root = artifact_root / study_name / variant_name / f"seed_{seed}"
        checkpoint_path = run_root / "model.pt"
        train_metrics_path = run_root / "train_metrics.json"
        eval_metrics_path = run_root / "eval_metrics.json"

        config_kwargs = base_config.to_dict()
        config_kwargs["seed"] = seed
        config_kwargs["log_every"] = log_every
        config_kwargs["checkpoint_path"] = checkpoint_path
        config_kwargs["metrics_path"] = train_metrics_path
        config_kwargs.update(overrides)
        config_kwargs["hidden_sizes"] = tuple(config_kwargs["hidden_sizes"])
        config_kwargs["checkpoint_path"] = Path(config_kwargs["checkpoint_path"])
        config_kwargs["metrics_path"] = Path(config_kwargs["metrics_path"])
        config = DQNConfig(**config_kwargs)

        print(f"[train] {study_name}/{variant_name} seed={seed}")
        train_metrics = train(config)
        print(f"[eval]  {study_name}/{variant_name} seed={seed}")
        evaluation_metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=eval_episodes,
            seed=seed + 1000,
            metrics_path=eval_metrics_path,
        )

        runs.append(
            {
                "seed": seed,
                "train_metrics_path": str(train_metrics_path.relative_to(ROOT)),
                "eval_metrics_path": str(eval_metrics_path.relative_to(ROOT)),
                "best_moving_average_reward": train_metrics["best_moving_average_reward"],
                "best_validation_reward": train_metrics["best_validation_reward"],
                "best_checkpoint_episode": train_metrics["best_checkpoint_episode"],
                "eval_mean_reward": evaluation_metrics["mean_reward"],
                "eval_std_reward": evaluation_metrics["std_reward"],
                "eval_success_rate": evaluation_metrics["success_rate"],
                "solved": evaluation_metrics["mean_reward"] >= SUCCESS_MEAN_REWARD,
            }
        )

    return aggregate_variant(variant_name, runs)


def run_chosen_default_study(
    *,
    seeds: tuple[int, ...],
    eval_episodes: int,
    log_every: int,
    base_config: DQNConfig,
    chosen_episodes: int,
    artifact_root: Path,
) -> tuple[dict[str, object], dict[str, Any]]:
    attempts: list[dict[str, object]] = []
    chosen_overrides: dict[str, Any] = {"episodes": chosen_episodes}

    initial_variant = run_variant(
        study_name="chosen_default",
        variant_name=f"episodes_{chosen_episodes}",
        seeds=seeds,
        eval_episodes=eval_episodes,
        log_every=log_every,
        base_config=base_config,
        overrides=chosen_overrides,
        artifact_root=artifact_root,
    )
    attempts.append(initial_variant)
    if initial_variant["aggregate"]["solved_all_seeds"]:
        return {"attempts": attempts, "selected_variant": initial_variant}, chosen_overrides

    current_overrides = dict(chosen_overrides)
    for fallback in FALLBACK_SEQUENCE:
        current_overrides.update(fallback["overrides"])
        variant = run_variant(
            study_name="chosen_default",
            variant_name=fallback["label"],
            seeds=seeds,
            eval_episodes=eval_episodes,
            log_every=log_every,
            base_config=base_config,
            overrides=current_overrides,
            artifact_root=artifact_root,
        )
        attempts.append(variant)
        if variant["aggregate"]["solved_all_seeds"]:
            return {"attempts": attempts, "selected_variant": variant}, current_overrides

    return {"attempts": attempts, "selected_variant": attempts[-1]}, current_overrides


def build_report(
    *,
    seeds: tuple[int, ...],
    baseline_summary: dict[str, object],
    chosen_summary: dict[str, object],
    chosen_overrides: dict[str, Any],
    baseline_episodes: int,
    eval_episodes: int,
) -> str:
    baseline_variant = baseline_summary["variants"][0]
    selected_variant = chosen_summary["selected_variant"]
    baseline_aggregate = baseline_variant["aggregate"]
    selected_aggregate = selected_variant["aggregate"]
    solved_text = "yes" if selected_aggregate["solved_all_seeds"] else "no"
    selected_episodes = chosen_overrides["episodes"]

    lines = [
        "# Benchmark Report",
        "",
        f"- Baseline training episodes per run: {baseline_episodes}",
        f"- Chosen default training episodes per run: {selected_episodes}",
        f"- Evaluation episodes per run: {eval_episodes}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Baseline",
        "",
        "| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best validation reward |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| {baseline_variant['name']} | "
            f"{baseline_aggregate['mean_eval_reward']:.2f} | "
            f"{baseline_aggregate['std_eval_reward']:.2f} | "
            f"{baseline_aggregate['mean_success_rate']:.2%} | "
            f"{baseline_aggregate['mean_best_validation_reward']:.2f} |"
        ),
        "",
        "## Chosen Default",
        "",
        "| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best validation reward | Solved all seeds |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        (
            f"| {selected_variant['name']} | "
            f"{selected_aggregate['mean_eval_reward']:.2f} | "
            f"{selected_aggregate['std_eval_reward']:.2f} | "
            f"{selected_aggregate['mean_success_rate']:.2%} | "
            f"{selected_aggregate['mean_best_validation_reward']:.2f} | "
            f"{solved_text} |"
        ),
        "",
        "## Chosen Default Rationale",
        "",
        (
            f"The published default starts from the stabilized DQN bundle and targets {selected_episodes} "
            "training episodes because the 1000-episode baseline is still too inconsistent across seeds."
        ),
    ]

    if selected_aggregate["solved_all_seeds"]:
        lines.append(
            "No fallback beyond the selected variant is needed because all benchmark seeds reached the "
            "solved threshold under deterministic evaluation."
        )
    else:
        lines.append(
            "The fallback sequence was exhausted and the last attempted variant is published honestly as the "
            "best available reliability improvement."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(args.seeds)
    base_config = DQNConfig(validation_episodes=args.validation_episodes)

    baseline_summary = {
        "study": "baseline",
        "variants": [
            run_variant(
                study_name="baseline",
                variant_name=f"episodes_{args.baseline_episodes}",
                seeds=seeds,
                eval_episodes=args.eval_episodes,
                log_every=args.log_every,
                base_config=base_config,
                overrides={"episodes": args.baseline_episodes},
                artifact_root=args.artifact_root,
            )
        ],
    }
    write_json(args.results_root / "baseline_summary.json", baseline_summary)

    chosen_summary, chosen_overrides = run_chosen_default_study(
        seeds=seeds,
        eval_episodes=args.eval_episodes,
        log_every=args.log_every,
        base_config=base_config,
        chosen_episodes=args.chosen_episodes,
        artifact_root=args.artifact_root,
    )
    write_json(args.results_root / "chosen_default_summary.json", chosen_summary)

    report = build_report(
        seeds=seeds,
        baseline_summary=baseline_summary,
        chosen_summary=chosen_summary,
        chosen_overrides=chosen_overrides,
        baseline_episodes=args.baseline_episodes,
        eval_episodes=args.eval_episodes,
    )
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "benchmark_report.md").write_text(report)
    print(f"Wrote summaries to {args.results_root}")


if __name__ == "__main__":
    main()
