from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

from dqn_cartpole.config import DQNConfig
from dqn_cartpole.evaluate import evaluate_checkpoint
from dqn_cartpole.train import train


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "benchmark_suite"
RESULTS_ROOT = ROOT / "results"
SEEDS = (7, 17, 27)
TRAIN_EPISODES = 1000
EVAL_EPISODES = 100

STUDIES = {
    "baseline": [
        {
            "name": "default",
            "overrides": {},
        }
    ],
    "epsilon_decay_comparison": [
        {
            "name": "epsilon_decay_0.995",
            "overrides": {"epsilon_decay": 0.995},
        },
        {
            "name": "epsilon_decay_0.99",
            "overrides": {"epsilon_decay": 0.99},
        },
    ],
}


def aggregate_variant(name: str, runs: list[dict[str, object]]) -> dict[str, object]:
    eval_means = [float(run["eval_mean_reward"]) for run in runs]
    success_rates = [float(run["eval_success_rate"]) for run in runs]
    moving_averages = [float(run["best_moving_average_reward"]) for run in runs]
    return {
        "name": name,
        "runs": runs,
        "aggregate": {
            "mean_eval_reward": mean(eval_means),
            "std_eval_reward": pstdev(eval_means) if len(eval_means) > 1 else 0.0,
            "mean_success_rate": mean(success_rates),
            "mean_best_moving_average_reward": mean(moving_averages),
        },
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_variant(study_name: str, variant_name: str, overrides: dict[str, object]) -> dict[str, object]:
    runs: list[dict[str, object]] = []

    for seed in SEEDS:
        run_root = ARTIFACT_ROOT / study_name / variant_name / f"seed_{seed}"
        checkpoint_path = run_root / "model.pt"
        train_metrics_path = run_root / "train_metrics.json"
        eval_metrics_path = run_root / "eval_metrics.json"

        config = DQNConfig(
            seed=seed,
            episodes=TRAIN_EPISODES,
            checkpoint_path=checkpoint_path,
            metrics_path=train_metrics_path,
            log_every=100,
            **overrides,
        )

        print(f"[train] {study_name}/{variant_name} seed={seed}")
        train_metrics = train(config)
        print(f"[eval]  {study_name}/{variant_name} seed={seed}")
        evaluation_metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            episodes=EVAL_EPISODES,
            seed=seed + 1000,
            metrics_path=eval_metrics_path,
        )

        runs.append(
            {
                "seed": seed,
                "train_metrics_path": str(train_metrics_path.relative_to(ROOT)),
                "eval_metrics_path": str(eval_metrics_path.relative_to(ROOT)),
                "best_moving_average_reward": train_metrics["best_moving_average_reward"],
                "eval_mean_reward": evaluation_metrics["mean_reward"],
                "eval_std_reward": evaluation_metrics["std_reward"],
                "eval_success_rate": evaluation_metrics["success_rate"],
            }
        )

    return aggregate_variant(variant_name, runs)


def build_report(baseline_summary: dict[str, object], comparison_summary: dict[str, object]) -> str:
    baseline_variant = baseline_summary["variants"][0]
    comparison_variants = comparison_summary["variants"]
    baseline_aggregate = baseline_variant["aggregate"]

    lines = [
        "# Benchmark Report",
        "",
        f"- Training episodes per run: {TRAIN_EPISODES}",
        f"- Evaluation episodes per run: {EVAL_EPISODES}",
        f"- Seeds: {', '.join(str(seed) for seed in SEEDS)}",
        "",
        "## Baseline",
        "",
        "| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best moving avg |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| {baseline_variant['name']} | "
            f"{baseline_aggregate['mean_eval_reward']:.2f} | "
            f"{baseline_aggregate['std_eval_reward']:.2f} | "
            f"{baseline_aggregate['mean_success_rate']:.2%} | "
            f"{baseline_aggregate['mean_best_moving_average_reward']:.2f} |"
        ),
        "",
        "## Epsilon Decay Comparison",
        "",
        "| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best moving avg |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for variant in comparison_variants:
        aggregate = variant["aggregate"]
        lines.append(
            (
                f"| {variant['name']} | "
                f"{aggregate['mean_eval_reward']:.2f} | "
                f"{aggregate['std_eval_reward']:.2f} | "
                f"{aggregate['mean_success_rate']:.2%} | "
                f"{aggregate['mean_best_moving_average_reward']:.2f} |"
            )
        )

    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            (
                "This comparison changes only the epsilon decay rate while keeping the rest of the "
                "training configuration fixed."
            ),
            (
                "Use the higher mean evaluation reward and lower cross-seed variance together to decide "
                "which default feels more reliable."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_variants = [
        run_variant("baseline", variant["name"], variant["overrides"])
        for variant in STUDIES["baseline"]
    ]
    baseline_summary = {
        "study": "baseline",
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "seeds": list(SEEDS),
        "variants": baseline_variants,
    }
    write_json(RESULTS_ROOT / "baseline_summary.json", baseline_summary)

    comparison_variants = [
        run_variant("epsilon_decay_comparison", variant["name"], variant["overrides"])
        for variant in STUDIES["epsilon_decay_comparison"]
    ]
    comparison_summary = {
        "study": "epsilon_decay_comparison",
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "seeds": list(SEEDS),
        "variants": comparison_variants,
    }
    write_json(RESULTS_ROOT / "epsilon_decay_comparison.json", comparison_summary)

    report = build_report(baseline_summary, comparison_summary)
    (RESULTS_ROOT / "benchmark_report.md").write_text(report)
    print(f"Wrote summaries to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
