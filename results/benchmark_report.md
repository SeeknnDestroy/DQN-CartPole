# Benchmark Report

- Baseline training episodes per run: 1000
- Chosen default training episodes per run: 1500
- Evaluation episodes per run: 100
- Seeds: 7, 17, 27

## Baseline

| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best validation reward |
| --- | ---: | ---: | ---: | ---: |
| episodes_1000 | 460.78 | 55.46 | 99.33% | 472.47 |

## Chosen Default

| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best validation reward | Solved all seeds |
| --- | ---: | ---: | ---: | ---: | --- |
| episodes_1500 | 500.00 | 0.00 | 100.00% | 500.00 | yes |

## Chosen Default Rationale

The published default starts from the stabilized DQN bundle and targets 1500 training episodes because the 1000-episode baseline is still too inconsistent across seeds.
No fallback beyond the selected variant is needed because all benchmark seeds reached the solved threshold under deterministic evaluation.
