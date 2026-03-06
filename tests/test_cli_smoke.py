from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_PATH) if not existing else f"{SRC_PATH}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def test_training_and_evaluation_cli(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "cartpole.pt"
    train_metrics_path = tmp_path / "metrics" / "train.json"
    eval_metrics_path = tmp_path / "metrics" / "eval.json"

    train_result = run_module(
        "-m",
        "dqn_cartpole.train",
        "--episodes",
        "4",
        "--max-steps",
        "50",
        "--batch-size",
        "4",
        "--replay-buffer-size",
        "32",
        "--update-every",
        "1",
        "--warmup-steps",
        "0",
        "--hidden-sizes",
        "16",
        "16",
        "--validation-interval",
        "2",
        "--validation-episodes",
        "2",
        "--log-every",
        "1",
        "--checkpoint-path",
        str(checkpoint_path),
        "--metrics-path",
        str(train_metrics_path),
    )

    assert "Saved best checkpoint" in train_result.stdout
    assert checkpoint_path.exists()
    assert train_metrics_path.exists()

    training_metrics = json.loads(train_metrics_path.read_text())
    assert training_metrics["episodes_completed"] == 4
    assert len(training_metrics["episode_rewards"]) == 4
    assert len(training_metrics["validation_history"]) == 2
    assert training_metrics["best_checkpoint_episode"] in {2, 4}

    evaluate_result = run_module(
        "-m",
        "dqn_cartpole.evaluate",
        "--checkpoint",
        str(checkpoint_path),
        "--episodes",
        "5",
        "--seed",
        "9",
        "--metrics-path",
        str(eval_metrics_path),
    )

    assert "Evaluation over 5 episodes" in evaluate_result.stdout
    assert eval_metrics_path.exists()

    evaluation_metrics = json.loads(eval_metrics_path.read_text())
    assert evaluation_metrics["episodes"] == 5
    assert "mean_reward" in evaluation_metrics
