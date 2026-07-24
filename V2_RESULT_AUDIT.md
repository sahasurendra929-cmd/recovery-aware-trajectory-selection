# QLoRA V2 result audit

Date: 2026-07-23  
Result branch: `results/v2-rtx5060-20260723`  
Audited result commit: `4b331d7`  
Scope: offline held-out next-tool-call imitation only

## Audit conclusion

The uploaded RTX 5060 V2 bundle is complete and internally consistent for the
frozen V2 protocol. All four controls contain 959 unique predictions for the
same held-out examples, use the same model revision, NF4 loading, prebuilt
prompts of at most 1,664 tokens, greedy generation, and 128 generated tokens.
All 32 uploaded payload hashes and byte counts were rechecked successfully.

The result branch README originally described 512-token left truncation and
called the fixed schedule “one epoch.” Those statements were documentation
errors: the machine-readable contracts and evaluator consistently specify no
runtime truncation, and training used exactly 1,088 scheduled microbatches /
68 optimizer steps with a small number of repeated examples. The README and
its upload manifest were corrected in closeout commit `99f50ec`.

The bundle does not contain adapters, processed JSONL files, or raw data, so it
supports artifact-level result auditing but not fully independent retraining.

## Main results

| Control | JSON valid | Tool accuracy | Full-call EM | Recovery EM | Agent-initiated EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base model | 17.52% | 6.15% | 0.63% | 7.55% | 7.69% |
| Random success | **99.79%** | **61.73%** | **33.06%** | **58.49%** | 0.00% |
| Shortest success | 99.27% | 59.96% | 28.15% | **58.49%** | **15.38%** |
| Recovery coverage | **99.79%** | 57.14% | 25.23% | 56.60% | **15.38%** |

The recovery subset contains only 53 examples: 40 user-assisted and 13
agent-initiated. It therefore cannot establish broad autonomous recovery.

## What V2 established

All trained arms used exactly 1,690,929 selected SFT tokens, including 523,182
GPT-4o tokens and 1,167,747 Sonnet tokens, and the same 1,088-microbatch padded
compute schedule. Nevertheless, the training signal changed materially:

| Arm | Trajectories | Unique tasks | Examples | Recovery targets | Agent targets | Scheduled loss tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random success | 141 | **63** | 1,066 | 78 | 13 | 36,706 |
| Shortest success | 195 | 50 | 1,074 | 87 | 8 | 37,268 |
| Recovery coverage | 140 | 44 | 1,062 | **188** | **32** | 39,577 |

Recovery coverage increased the intended supervision but reduced task diversity
and changed the target-tool distribution. It did not improve recovery EM and
reduced overall full-call EM by 7.82 percentage points relative to Random.
A paired task-cluster bootstrap estimated the Random-minus-Recovery difference
as +7.82 points with a 95% interval of `[+3.98, +11.89]`. Recovery-subset
arm-to-arm intervals crossed zero; training-seed uncertainty is not covered
because V2 has one seed.

The largest observed failure mechanism was ordinary tool routing. Among 71
non-recovery targets for `find_user_id_by_name_zip`, Random was correct on 56,
whereas Recovery Coverage was correct on 13 and selected
`find_user_id_by_email` 55 times. V2 therefore supports a negative result:
maximizing coarse recovery coverage can shift the learned action prior and
cause negative transfer.

## Claim boundary and V3 implication

V2 supports these claims:

1. Completion-only QLoRA SFT improves canonical next-tool-call prediction over
   the unadapted 0.5B base model.
2. Coarse recovery-volume or coverage maximization is not sufficient.
3. Recovery selection must also preserve task coverage, ordinary targets,
   target-tool marginals, source budgets, and effective supervised tokens.

V2 does not prove end-to-end Agent success, executable tool success, reliable
autonomous recovery, or performance across training seeds.

V3 is therefore a one-variable diagnostic: it retains V2's prompts, SFT labels,
model, schedule, source quotas, and evaluator, and changes only selection. It
adds hard retention constraints and saturates distinct recovery signatures
instead of maximizing raw recovery volume. Failed tool calls remain V2 SFT
labels and are explicitly counted; removing them or adding DPO is a later
factorial experiment that requires its own clean Random control.
