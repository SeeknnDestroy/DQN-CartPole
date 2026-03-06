# Benchmark Report

- Training episodes per run: 1000
- Evaluation episodes per run: 100
- Seeds: 7, 17, 27

## Baseline

| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best moving avg |
| --- | ---: | ---: | ---: | ---: |
| default | 166.17 | 33.12 | 20.00% | 173.07 |

## Epsilon Decay Comparison

| Variant | Mean eval reward | Std across seeds | Mean success rate | Mean best moving avg |
| --- | ---: | ---: | ---: | ---: |
| epsilon_decay_0.995 | 166.17 | 33.12 | 20.00% | 173.07 |
| epsilon_decay_0.99 | 140.46 | 98.95 | 32.33% | 140.97 |

## Takeaway

This comparison changes only the epsilon decay rate while keeping the rest of the training configuration fixed.
Use the higher mean evaluation reward and lower cross-seed variance together to decide which default feels more reliable.
