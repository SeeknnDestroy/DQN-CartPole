from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_PATH) if not existing else f"{SRC_PATH}{os.pathsep}{existing}"
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_export_agent_gif_smoke(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints" / "cartpole.pt"
    metrics_path = tmp_path / "metrics" / "train.json"
    gif_path = tmp_path / "gifs" / "agent.gif"

    run_python(
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
        "--checkpoint-path",
        str(checkpoint_path),
        "--metrics-path",
        str(metrics_path),
    )
    result = run_python(
        "scripts/export_agent_gif.py",
        "--checkpoint",
        str(checkpoint_path),
        "--output",
        str(gif_path),
        "--frame-skip",
        "8",
    )

    assert "Exported" in result.stdout
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
