# DQN CartPole

[![CI](https://github.com/SeeknnDestroy/DQN-CartPole/actions/workflows/ci.yml/badge.svg)](https://github.com/SeeknnDestroy/DQN-CartPole/actions/workflows/ci.yml)

Minimal, reproducible Deep Q-Network training and evaluation for `CartPole-v1` using PyTorch and Gymnasium.

This project started as a notebook experiment and was refactored into a small RL codebase with stable entrypoints, deterministic seeding, tests, artifact generation, and a stabilized DQN training loop that solves CartPole reliably across benchmark seeds.

![Agent Performance](agent_performance.gif)

## Why this repo exists

`CartPole-v1` is a small but useful benchmark for demonstrating reinforcement learning fundamentals:

- value-function approximation with a neural network
- experience replay
- target network updates
- epsilon-greedy exploration
- reproducible training and evaluation loops

For a portfolio project, the value is less about the benchmark itself and more about whether the implementation is coherent, runnable, and easy to review.

## Project structure

```text
src/dqn_cartpole/
  agent.py         DQN policy, learning step, target updates
  config.py        typed configuration for training/eval
  evaluate.py      checkpoint loading and deterministic evaluation CLI
  model.py         Q-network definition
  replay_buffer.py replay sampling utilities
  train.py         training loop and checkpoint/metrics writing
  utils.py         seeding, environment creation, JSON helpers
tests/
  unit + smoke tests for replay, agent, and CLI workflows
notebooks/
  demo notebook that imports package code instead of defining it
```

## Quickstart

### 1. Install

Recommended: use `uv`.

```bash
uv sync --extra test
```

If you prefer standard library tooling, use `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Train

```bash
uv run python -m dqn_cartpole.train \
  --episodes 1500 \
  --checkpoint-path artifacts/checkpoints/cartpole_dqn.pt \
  --metrics-path artifacts/train_metrics.json
```

If you used `venv`, run the same command after activating `.venv`, without `uv run`.

Training writes:

- the best validation checkpoint at `artifacts/checkpoints/cartpole_dqn.pt`
- machine-readable training metrics at `artifacts/train_metrics.json`
- console logs with episode reward, moving average reward, epsilon, and validation summaries

### 3. Evaluate

```bash
uv run python -m dqn_cartpole.evaluate \
  --checkpoint artifacts/checkpoints/cartpole_dqn.pt \
  --episodes 100 \
  --metrics-path artifacts/eval_metrics.json
```

If you used `venv`, run the same command after activating `.venv`, without `uv run`.

Evaluation reports:

- mean reward
- reward standard deviation
- per-episode rewards
- success rate for episodes with reward >= 200
- whether the mean evaluation reward reaches the official solved threshold

## Results

Benchmarks below were generated with:

- 1000 training episodes per run
- 1500 training episodes per run for the published default
- 100 evaluation episodes per run
- seeds `7`, `17`, and `27`

### Baseline

| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best validation reward |
| --- | ---: | ---: | ---: | ---: |
| episodes_1000 | 460.78 | 55.46 | 99.33% | 472.47 |

### Published default

| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best validation reward | Solved all seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| episodes_1500 | 500.00 | 0.00 | 100.00% | 500.00 | yes |

Takeaway: the stabilized DQN default keeps the implementation small but moves the repo from “sometimes works” to a consistent solved benchmark. The 1000-episode budget is close, but still misses one benchmark seed; 1500 episodes solves all three.

Raw summaries:

- [results/benchmark_report.md](results/benchmark_report.md)
- [results/baseline_summary.json](results/baseline_summary.json)
- [results/chosen_default_summary.json](results/chosen_default_summary.json)

## Design notes

- The training loop has a single source of truth: `agent.step(...)` owns replay insertion, learning cadence, and replay warmup behavior.
- The learning rule uses Double DQN targets, Huber loss, gradient clipping, and validation-based checkpoint selection to improve stability without turning the project into a larger RL framework.
- The repo uses `gymnasium` instead of legacy `gym` and handles `terminated` / `truncated` correctly.
- Checkpoints store model weights, optimizer state, config, and summary metadata so evaluation can run independently from the training process.
- Tests cover RL plumbing, validation checkpoint flow, and GIF export in addition to the CLI smoke path.

## Reproducibility

- Python, NumPy, Torch, and environment seeding are set explicitly.
- The default artifact paths are stable and suitable for CI smoke tests.
- `pytest` covers unit-level tensor shapes and a small end-to-end train/evaluate run.

Run the test suite with:

```bash
uv run pytest
```

If you used `venv`, run `pytest` after activating `.venv`.

Regenerate the benchmark summaries with:

```bash
uv run python scripts/run_benchmark_suite.py
```

Refresh the demo GIF from a trained checkpoint with:

```bash
uv run python scripts/export_agent_gif.py \
  --checkpoint artifacts/benchmark_suite/chosen_default/episodes_1500/seed_7/model.pt \
  --output agent_performance.gif
```

## Demo notebook

The original notebook-heavy structure has been reduced to a thin demo notebook that imports the package code:

- [notebooks/cartpole_dqn_demo.ipynb](notebooks/cartpole_dqn_demo.ipynb)

Use it for exploration or quick inspection after installing the package in editable mode.

## Current limitations

- This is intentionally scoped to a single classic-control environment.
- There is no experiment tracker, hyperparameter sweeper, or GPU-specific optimization layer.
- The published default is reliable at the benchmarked seeds, but it still depends on a non-trivial training budget rather than a more advanced RL method family.
