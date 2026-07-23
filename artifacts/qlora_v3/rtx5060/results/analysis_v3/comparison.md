# QLoRA V3 constrained-recovery diagnostic

Offline held-out next-tool-call imitation on an already inspected V2 test set; this is neither executable Agent success nor confirmatory paper evidence.

- V3 audit: **PASS**
- V2 random reference: **compatible**
- Direction label: **non_recovery_preserved_without_recovery_gain**
- Interpretation: Non-recovery EM is preserved, but the predeclared recovery gain is absent.

| group | n | V3 tool acc. | V3 full-call EM | V2 random tool acc. | V2 random full-call EM | Δ tool | Δ full-call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 959 | 59.65% | 31.07% | 61.73% | 33.06% | -2.09% | -1.98% |
| non_recovery | 906 | 59.49% | 29.69% | 61.70% | 31.57% | -2.21% | -1.88% |
| recovery | 53 | 62.26% | 54.72% | 62.26% | 58.49% | 0.00% | -3.77% |
| agent_initiated | 13 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| user_assisted | 40 | 82.50% | 72.50% | 82.50% | 77.50% | 0.00% | -5.00% |

Reference note: Compatible V2 random_success predictions were rescored and paired by example ID.
