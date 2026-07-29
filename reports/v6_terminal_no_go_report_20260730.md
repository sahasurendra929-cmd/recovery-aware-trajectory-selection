# V6 Terminal Report — Pilot Candidate-Supply NO-GO

Date: 2026-07-30 (Asia/Shanghai)

Final status: **NO-GO before candidate-pool construction**

## What completed

- Three-GPU environment and provenance preflight passed.
- Exact model revisions and tau2 commit were reachable and matched.
- All 83 V6 repository tests passed.
- The outcome-independent registry passed:
  24 Pilot tasks and 72 registered candidate pairs.
- The official test remained sealed and unused.

## Terminal result

The frozen V6 generator began with `airline:12`. It allows two clean attempts,
using the registered task seed and the next integer seed. Both executions made
real tool calls and completed normally, but both received official tau2 reward
`0.0`. The required successful clean source trajectory therefore did not
exist under the frozen generation contract.

The generator correctly exited before writing a candidate pair:

```text
V6GenerationError: airline:12: no successful clean rollout with an observed tool call
```

Observed frozen result:

| Item | Value |
|---|---:|
| Registered Pilot tasks | 24 |
| Registered candidate pairs | 72 |
| Tasks reached | 1 |
| Frozen clean attempts | 2 |
| Successful clean attempts | 0 |
| Candidate pairs emitted | 0 |
| Official test accesses | 0 |

## Diagnostic evidence

After preserving the original failure, an isolated diagnostic increased only
the clean-attempt ceiling to eight. It was not preregistered before additional
outcomes were observed and is therefore excluded from V6 evidence. All eight
attempts for `airline:12` also received official reward `0.0`; no candidate was
emitted.

## Gate consequences

Because there is no candidate pool:

- exact token and hardness measurement is impossible;
- Pilot execution audit and the preregistered GO/NO-GO population checks are
  impossible;
- selector scoring and manifests are unauthorized;
- SFT materialization and all training arms are unauthorized;
- checkpoint registry, official-test unseal, end-to-end evaluation, bootstrap
  confidence intervals, and clean non-inferiority are unauthorized.

The correct scientific conclusion is:

> The frozen V6 experiment is engineering-complete through registry
> construction but fails its candidate-supply prerequisite. It provides no
> evidence for or against `full_proposed` relative to random selection.

## Code-gap audit

The source commit also lacks the handoff's downstream:

- `scripts/build_v6_checkpoint_registry.py`
- `scripts/run_v6_end_to_end_eval.py`
- `scripts/summarize_v6_directional_screen.py`

Those gaps did not cause the terminal outcome: Stage 3 failed before any of
them could be authorized. Implementing them after observing this failure would
not repair the missing frozen candidate pool.

## Reproducibility hashes

- Registry file:
  `7e7ea772f44a109e818ce8de73812b4d4ba42da6e0595cdd92ff1770a093a136`
- Frozen generation run contract:
  `e571842eb359821e33686dfcfa7182d8a0f793120702ae9b4e30b906778bb36e`
- Frozen generation log:
  `a789793cd66ec61b429efabf24ef94fdaa77c016e6731562a2f0abe6266c347b`
- Quarantined diagnostic contract:
  `3a9e59b7eadf9d75e2740838e55a5030a5cf5073c133f51acb60586e9845211c`
- Quarantined diagnostic log:
  `e0be5a68cdf4e92d34235b1a93617b877163f835d00ab56d5467adc24ffae02a`

## Required future protocol

Any future continuation must be explicitly preregistered as V6.1 before new
outcomes are observed. It must state whether it changes the Pilot task set,
clean-attempt population, teacher policy, or generation seeds. V6 and V6.1
results must never be merged.

