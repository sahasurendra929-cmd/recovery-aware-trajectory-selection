# QLoRA V4 paired comparison

All values are recomputed from complete prediction JSONL files. Intervals are paired 10,000-draw task-cluster bootstrap intervals.

## clean_sft_minus_standard_v3

| group / metric | treatment | control | delta | treatment-only true | control-only true | task-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 0.312826 | 0.310740 | +0.002086 | 11 | 9 | [-0.006965, +0.010368] |
| outcome_success | 0.292683 | 0.288248 | +0.004435 | 11 | 7 | [-0.005015, +0.012752] |
| failed_gold / failed_action_imitation_rate | 0.631579 | 0.666667 | -0.035088 | 0 | 2 | [-0.074088, +0.000000] |
| non_recovery_success | 0.275822 | 0.272300 | +0.003521 | 10 | 7 | [-0.006593, +0.012890] |
| recovery_success | 0.580000 | 0.560000 | +0.020000 | 1 | 0 | [+0.000000, +0.078947] |
| agent_initiated | 0.000000 | 0.000000 | +0.000000 | 0 | 0 | [+0.000000, +0.000000] |
| user_assisted | 0.763158 | 0.736842 | +0.026316 | 1 | 0 | [+0.000000, +0.103448] |

Rejected-repeat rate on the 48 strict preference-pair examples: treatment=0.125000, control=0.145833, delta=-0.020833 (lower is better).

## continued_sft_minus_clean_sft

| group / metric | treatment | control | delta | treatment-only true | control-only true | task-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 0.274244 | 0.312826 | -0.038582 | 13 | 50 | [-0.055127, -0.022803] |
| outcome_success | 0.250554 | 0.292683 | -0.042129 | 12 | 50 | [-0.059476, -0.025901] |
| failed_gold / failed_action_imitation_rate | 0.649123 | 0.631579 | +0.017544 | 1 | 0 | [+0.000000, +0.044118] |
| non_recovery_success | 0.231221 | 0.275822 | -0.044601 | 10 | 48 | [-0.060956, -0.028610] |
| recovery_success | 0.580000 | 0.580000 | +0.000000 | 2 | 2 | [-0.100000, +0.075000] |
| agent_initiated | 0.166667 | 0.000000 | +0.166667 | 2 | 0 | [+0.000000, +0.444444] |
| user_assisted | 0.710526 | 0.763158 | -0.052632 | 0 | 2 | [-0.166667, +0.000000] |

Rejected-repeat rate on the 48 strict preference-pair examples: treatment=0.104167, control=0.125000, delta=-0.020833 (lower is better).

## dpo_minus_continued_sft

| group / metric | treatment | control | delta | treatment-only true | control-only true | task-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 0.248175 | 0.274244 | -0.026069 | 45 | 70 | [-0.055725, +0.002215] |
| outcome_success | 0.250554 | 0.250554 | +0.000000 | 45 | 45 | [-0.031523, +0.032476] |
| failed_gold / failed_action_imitation_rate | 0.210526 | 0.649123 | -0.438596 | 0 | 25 | [-0.666667, -0.291667] |
| non_recovery_success | 0.234742 | 0.231221 | +0.003521 | 44 | 41 | [-0.029788, +0.038349] |
| recovery_success | 0.520000 | 0.580000 | -0.060000 | 1 | 4 | [-0.126984, +0.027027] |
| agent_initiated | 0.083333 | 0.166667 | -0.083333 | 0 | 1 | [-0.272727, +0.000000] |
| user_assisted | 0.657895 | 0.710526 | -0.052632 | 1 | 3 | [-0.138889, +0.058824] |

Rejected-repeat rate on the 48 strict preference-pair examples: treatment=0.000000, control=0.104167, delta=-0.104167 (lower is better).

## Required chosen/rejected log-probability scoring

| arm | status | summed-logp accuracy | mean summed margin | normalized accuracy | mean normalized margin |
| --- | --- | ---: | ---: | ---: | ---: |
| standard_v3 | not_available | — | — | — | — |
| clean_sft | complete | 0.708333 | -2.834618 | 0.708333 | +0.069266 |
| continued_sft | complete | 0.791667 | -1.907001 | 0.708333 | +0.103953 |
| dpo | complete | 0.875000 | +37.658790 | 0.833333 | +1.701206 |

DPO minus continued-SFT pair-score deltas:

- pair_accuracy_summed_logp: +0.083333
- mean_summed_logp_margin: +39.565791
- per_token_normalized_pair_accuracy: +0.125000
- mean_per_token_normalized_margin: +1.597253

## Frozen exploratory screening gate

Decision: **do_not_claim_positive_mechanism_from_v4**. This gate only decides whether to run fresh three-seed confirmation; it is not a paper-final claim.

Only the two preregistered primary recovery-success full-call tests receive Holm adjustment. All other p-values are uncorrected exploratory diagnostics.

## Claim boundary

Offline next-tool-call evaluation on the already inspected frozen V2/V3 test set. Single-seed task-cluster intervals do not cover training randomness and cannot establish end-to-end Agent success, cross-seed robustness, or paper-final confirmation.
