# Observed pilot v1 summary

This is an offline, legacy τ-bench retail pilot. It used 915 raw successful records, 588 exact-sequence-deduplicated trajectories, 78 train task IDs, and 22 test task IDs.

| Method | Estimated tokens | Traces | Recovery traces | Recovery events | Tool sequences |
|---|---:|---:|---:|---:|---:|
| Random-success | 711,329 | 131 | 49 (37.4%) | 59 | 111 |
| Shortest-success | 707,887 | 179 | 79 (44.1%) | 110 | 108 |
| Recovery-balanced | 710,443 | 131 | 74 (56.5%) | 95 | 120 |

The budget gap between arms was at most 0.484%. Recovery-balanced materially changed retained error-resolution coverage, but a transparent offline mode/retrieval diagnostic did not show a recovery-call exact-match improvement over random sampling. This is a selection-coverage result, not an end-to-end Agent result.
