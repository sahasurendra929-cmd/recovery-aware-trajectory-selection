# QLoRA v2 comparison

Offline held-out next-tool-call imitation only; this is not executable Agent success.

| arm | status | JSON valid | tool acc. | full call EM | task-macro EM | task-cluster 95% CI | recovery EM | agent-initiated EM |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| base_model | complete | 17.52% | 6.15% | 0.63% | 1.26% | [0.10, 1.40] | 7.55% | 7.69% |
| random_success | complete | 99.79% | 61.73% | 33.06% | 35.42% | [28.77, 38.53] | 58.49% | 0.00% |
| shortest_success | complete | 99.27% | 59.96% | 28.15% | 31.87% | [23.41, 33.83] | 58.49% | 15.38% |
| recovery_coverage | complete | 99.79% | 57.14% | 25.23% | 28.70% | [21.47, 29.84] | 56.60% | 15.38% |
