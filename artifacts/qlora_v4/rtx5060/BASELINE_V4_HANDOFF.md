# QLoRA V4 clean-SFT and DPO handoff

V4 studies how a fixed recovery-oriented trajectory selection should be
supervised. It does not reselect trajectories. All new models use the 167
trajectories frozen by V3 `constrained_recovery`.

The experiment has three new checkpoints:

1. **Clean-SFT** reallocates failed tool-call target slots to matched clean
   targets while retaining failures as context for later verified repairs.
2. **continued-SFT** starts from Clean-SFT and continues chosen-only SFT on
   the preference-pair schedule.
3. **DPO** starts from the byte-identical Clean-SFT checkpoint and
   learns that the verified repair is preferred to repeating the observed
   failed call.

The existing Standard V3 checkpoint is the fourth result point. Pure
DPO-from-base and a second selection arm are deliberately out of scope.

The existing Standard V3 959-row prediction artifact can be reused only after
its test hash, NF4 loading, 1,664-token prompt bound, 128-token greedy decode,
and `limited=false` fields match V4. It supports recomputing outcome-success,
non-recovery-success, recovery-success, and repeat-error metrics. It does not
contain token
log-probabilities, so Standard V3 pair-ranking must be shown as `N/A` unless a
new scoring-only forward pass is run under the frozen scorer.

A positive result is not guaranteed. The 959-example V2/V3 test set has
already been inspected, so all V4 results on it are exploratory. A paper claim
requires fresh unseen tasks and three training seeds.

## 1. Frozen causal comparisons

| Comparison | Changed factor | Question |
| --- | --- | --- |
| Standard V3 vs. Clean-SFT | failed-to-clean supervision allocation | Does reallocating label budget away from known failures help? |
| Clean-SFT vs. continued-SFT | additional chosen exposure | Does more positive SFT help? |
| continued-SFT vs. DPO | second-stage objective | Does preference learning outperform continued SFT? |

continued-SFT and DPO use the same 144-entry pair schedule, the same prompt
and chosen exposure, the same 18 optimizer steps, and the same optimizer
hyperparameters. DPO additionally processes rejected responses and frozen
reference log-probabilities. It therefore uses more computation. The proper
description is **chosen-exposure-matched SFT control**, not exact
compute-matching. Actual policy tokens, reference tokens, wall time, GPU time,
and peak VRAM must be reported.

## 2. Source and release freeze

The implementation must fill and freeze these values after deterministic
preparation and before any formal GPU run:

```text
V4 repository tag:                 v4-frozen-20260724-p2
Implementation parent commit:      7e0419d9b0941902ae149a68498ce9a19b1ea2f1
Clean train JSONL SHA256:           1bfa3d9df8e38e6a97237aa6efb47cc4bfacd9a15a1a396acdbf213b3f7ca1e8
Clean train schedule SHA256:        4b28da48082ef5bd3396e7df4b5b723c4efffe4b2e5438f47c8c2ca9d709f386
Preference pairs JSONL SHA256:      f6b967d5decb4741e3b1fbee2c0a0b3ac4760dfdfb76d910fd0b6cf7d0adefe5
Preference schedule SHA256:         f3cd0565cab0fd12252512b018a749dbe2c42a89d15ec92efd6b03a18f521341
Preference smoke schedule SHA256:   86fd923875ba3d11c50d635c409246e6f09437c771a4b4881c02cbba47190eb4
Global longest preference pair ID:  sonnet-35-new-retail:task106:trial4:action28:prefer_repair_over_action26
Test outcome annotations SHA256:     9a4ec2b1e25ee512e5946e8ac770b0fbf6b0ed0d5b61f994c643fc04cd227b57
```

These values are frozen. The RTX 5060 operator must not edit or replace them.

Patch release `v4-frozen-20260724-p2` supersedes
`v4-frozen-20260724-p1` at commit
`ab7a5439680eed75967dfcdfbaf6b14014ab54b4`. It updates both sequential
Trainer overrides to the pinned Transformers 4.52.4 sampler signature,
explicitly enables the intended `/8` gradient-accumulation scaling for the
continued-SFT custom loss, and initializes the CUDA allocator before
preference-training peak-memory reset. P1 had already added the equivalent
allocator initialization to generation evaluation. The scientific protocol,
data, hashes, model, seed, intended training schedules, generation
parameters, and metrics are unchanged. No preference optimizer step completed
under P1. Artifacts from a superseded source commit cannot be mixed into a
formal patch-release run.

The config deliberately does not contain the SHA of the commit that contains
the config itself; that would be an impossible self-reference. At runtime the
runner must verify `HEAD == repository_tag`, then record the resolved HEAD SHA
in every run manifest. `implementation_parent_commit` identifies the frozen
V3 parent commit from which the V4 implementation branch was created. The
tagged V4 implementation itself is identified by resolving `repository_tag`.

The Clean-SFT adapter hash is runtime-generated and therefore is `null` in the
static config. `train-clean-sft` must record it; `smoke-preference`, both
stage-2 arms, aggregation, and packaging must then require the same non-null
hash. This runtime value must not be manually copied into the source config.

Already frozen:

- parent source commit: `7e0419d`;
- parent selection trace SHA256:
  `a65bba64baf7c9a6e816e721b382511211aa9df6f5204e7c4cce74f78b992cc5`;
- τ-bench:
  `59a200c6d575d595120f1cb70fea53cef0632f6b`;
- Qwen2.5-0.5B-Instruct:
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- seed: `20260722`;
- current exploratory test SHA256:
  `0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96`.
- Standard V3 result commit:
  `aedf77a5784a364bd76bad42aa0a6cb6fad555b6`;
- Standard V3 checkpoint fingerprint:
  `3cdfa858353e8f7ea6da0d5558c21014bacbd58b2092f7095d6b5925f147825c`;
- Standard V3 metrics / contract / predictions SHA256:
  `880db45dcb6dc6eea497aa32dff26c5d59a4ab3b570c458c6f132757ea9d61f4`,
  `2b941573f85b9c1d33622c4e6fde42d10af194981fe028baa7e210d39a455471`,
  and
  `491e1613b20eb11b176d9aac61e19b4e3472257d3a2576125c76c6e03cb24de3`.

## 3. Clean-SFT construction

For every assistant tool-call target in the frozen V3 selection, inspect its
immediate tool response using the frozen V3 error rules.

```text
immediate response is an error  -> never a Clean-SFT label
ordinary nonfailed action       -> Clean-SFT label
verified repair after an error  -> Clean-SFT label
```

Removing a label does not delete history. A failed action and its error
response remain in the prompt of a later repair example. Only messages before
the target may enter a prompt.

Because every removed slot is filled by a matched clean target, this comparison
does not identify the effect of label deletion alone. It tests the practical
fixed-budget intervention of moving supervision from known failed actions to
source- and target-tool-matched clean actions.

The 1,088-slot Clean-SFT schedule must be rebuilt deterministically:

- use the same 167 V3 trajectories;
- expose every unique retained clean target at least once;
- fill removed slots using clean replacements from those trajectories;
- match source and target-tool distributions as closely as the frozen
  preparation contract specifies;
- never use validation or test outcomes to select a replacement;
- retain 1,088 microbatches, gradient accumulation 16, and 68 optimizer
  steps;
- retain fixed 2,048-token padding for the stage-1 comparison;
- keep scheduled completion/loss tokens within 1% of the V3 value 36,599.

Preparation must report, by source and target tool:

- retained targets;
- removed failed targets;
- replacement targets;
- unique examples and repeat counts;
- selected and scheduled completion tokens;
- future-leakage and split-overlap counts;
- all output hashes.

Clean-SFT requires `failed_action_labels=0` and `future_leakage=0`.

Its completion-only objective is

\[
\mathcal{L}_{clean}
=-\frac{1}{\sum_t m_t}\sum_t m_t
\log \pi_\theta(y_t\mid h,y_{<t}),
\]

where \(m_t=1\) only on target tool-call tokens and the target EOS. Prompt
tokens have \(m_t=0\), and every known failed-action target has no Clean-SFT
training row. The runner must fail on a zero-loss-token microbatch rather than
silently training on prompt tokens.

## 4. High-confidence preference pairs

The prepared training set contains exactly 79 unique pairs:

| Recovery mode | Unique pairs |
| --- | ---: |
| agent-initiated | 18 |
| user-assisted | 61 |
| total | 79 |

Each pair is:

\[
(h,a^+,a^-)
\]

where \(h\) is the post-failure history, \(a^+\) is the immediate verified
repair call, and \(a^-\) is a canonical repetition of the observed failed
call.

A pair is valid only when:

- the failed action is followed by a deterministic, nonretryable error;
- prompt \(h\) ends after the failure information and contains no future
  message;
- chosen is the immediate repair and does not itself immediately fail;
- chosen and rejected differ after canonical JSON normalization;
- prompt, chosen, and rejected satisfy the 1,664/384/2,048 token contract;
- chosen and rejected explicitly include the tokenizer EOS in training;
- the task belongs to the training split.

Timeouts, rate limits, network failures, generic ambiguous errors, and cases
where retry can be correct are excluded. A final trajectory reward alone is
not sufficient evidence that an intermediate action is a valid repair.

The frozen second-stage schedule has 144 entries:

```text
72 agent-initiated slots
72 user-assisted slots
batch size 1
gradient accumulation 8
18 optimizer steps
```

The mode balancing is deliberate. Both second-stage arms consume the exact
same pair IDs and order. continued-SFT consumes only `prompt + chosen`; DPO
consumes `prompt + chosen + rejected`.

Training code must not open the test preference-pair file. Every stage-2 run
manifest must record `held_out_test_accessed=false`; construction and scoring
of the held-out pair set belong only to the evaluation path.

For continued-SFT, the objective is the same completion-only cross-entropy
above, evaluated on \(a^+\). For DPO, with the frozen Clean-SFT policy
\(\pi_{ref}\),

\[
\mathcal{L}_{DPO}
=-\log \sigma\left(
\beta\left[
\log\frac{\pi_\theta(a^+\mid h)}{\pi_{ref}(a^+\mid h)}
-\log\frac{\pi_\theta(a^-\mid h)}{\pi_{ref}(a^-\mid h)}
\right]\right),\qquad \beta=0.1.
\]

Response log-probabilities sum only completion tokens, including EOS; prompt
tokens are excluded.

## 5. Frozen software and model loading

Use Python 3.11 and:

```text
torch==2.7.1+cu128
transformers==4.52.4
trl==0.18.2
peft==0.15.2
bitsandbytes==0.46.0
datasets==3.6.0
accelerate==1.7.0
```

Stage 1 retains the V3 QLoRA setup:

```text
NF4 4-bit + double quantization
BF16 compute
LoRA r=16, alpha=32, dropout=0.05
all q/k/v/o and gate/up/down projection modules
paged AdamW 8-bit
learning rate 1e-4
```

All arms run the frozen number of optimizer steps. Validation is audit-only:
there is no early stopping, best-checkpoint selection, or hyperparameter
choice based on validation/test outcomes. A loss over known failed-action
labels must not be used to select Clean-SFT, because those labels are precisely
what the intervention excludes.

Both stage-2 arms start from separate copies of the exact Clean-SFT adapter.
They reset optimizer and scheduler state, use learning rate `1e-5`, a constant
schedule with zero warmup, a sequential sampler, max gradient norm 1.0, and
disable dropout. The schedule file order must not be shuffled.

### Memory-safe DPO reference

Do not keep a second base model on the GPU. Load one NF4 base and the same
Clean-SFT adapter twice:

```text
default adapter:   trainable policy role
reference adapter: frozen reference role
```

Pass the resulting `PeftModel` to TRL without a new `peft_config`, use
`ref_model=None`, set `model_adapter_name=default` and
`ref_adapter_name=reference`, and enable:

```text
precompute_ref_log_probs=true
precompute_ref_batch_size=1
use_logits_to_keep=true
max_prompt_length=1664
max_completion_length=384
max_length=2048
batch_size=1
gradient_checkpointing=true
padding_free=false
```

Before training, the policy and reference adapters must have identical
fingerprints. With dropout disabled, the maximum absolute policy/reference
log-probability difference on smoke pairs must be at most `1e-4`, and initial
DPO loss must be within 0.02 of \(-\log 0.5=0.693147\). The reference adapter
must remain byte-identical after training.

`default` is an internal PEFT name, not a scientific arm name. It is required
so the trained policy saves to the root of `checkpoint_final` and reloads
uniformly. Manifests must record both the internal adapter name and its policy
role.

## 6. RTX 5060 preflight

Use a fresh V4 checkout and fresh output paths. A previously created virtual
environment may be reused only when every frozen Python/package/CUDA version
passes the exact preflight; otherwise create a new environment. Reusing a
verified environment, pip cache, and Hugging Face cache does not change the
scientific protocol. The laptop must:

- report an RTX 5060 Laptop GPU with approximately 8,151 MiB;
- support CUDA and BF16;
- have at least 6 GiB free VRAM before model loading;
- have no other Python/CUDA model process;
- be connected to power, kept awake, and adequately cooled.

Do not set `expandable_segments:True`; the prior Windows build reported that
allocator option as unsupported. Do not enable FlashAttention, Liger,
`torch.compile`, automatic batch-size discovery, CPU model fallback, or an
unfrozen library version.

## 7. Mandatory smoke gates

Preparation and CPU audit must pass before CUDA work. Clean-SFT smoke runs
before formal Clean-SFT training. Preference smoke runs only after the formal
Clean-SFT checkpoint exists, because both smoke arms must initialize from that
checkpoint.

### Clean-SFT smoke

Run at least two optimizer steps and include the longest retained Clean-SFT
sequence. Require:

- exact token and loss-mask agreement with the prepared file;
- finite train and validation loss and gradient norm;
- no OOM;
- a nonempty adapter that reloads successfully.

### Stage-2 smoke

Use a small set containing the longest preference pair and representative
agent/user-assisted pairs. Run:

1. reference-log-probability precomputation;
2. two optimizer steps of chosen-only continued SFT;
3. two optimizer steps of DPO;
4. checkpoint save and reload;
5. NF4 generation of 128 tokens from a 1,664-token prompt.

Require:

- no runtime truncation;
- finite loss, reward margin, log-probabilities, and gradient norm;
- initial policy/reference agreement and DPO-loss checks;
- unchanged reference adapter;
- changed policy adapter;
- identical prompt/chosen schedule hashes between the two arms;
- `torch.cuda.max_memory_reserved() <= 7.5 GiB`.

The two-step requirement is important because optimizer memory may not be
fully allocated until the first update.

## 8. Formal stage order

The frozen implementation should expose these separate gates:

```powershell
$v4Python = ".\.venv\Scripts\python.exe"
$v4Data = "data\raw\tau-bench\historical_trajectories"

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage prepare
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage audit
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage smoke-clean
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage train-clean-sft
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage smoke-preference
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage train-sft-long
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage train-dpo
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage evaluate
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage score
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage aggregate
& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage package
```

Do not combine them into an unattended command until each stage has passed
individually once. `smoke-preference` before a completed formal Clean-SFT
checkpoint is invalid. A preparation, hash, smoke, or reference-policy failure
is a hard stop.

The runner's frozen preference-trainer delegation is:

```powershell
& $v4Python scripts\train_preference_v4.py `
  --arm continued_sft `
  --pair-file data\processed\qlora_v4\preference\train_schedule.jsonl `
  --expected-pair-file-sha256 f3cd0565cab0fd12252512b018a749dbe2c42a89d15ec92efd6b03a18f521341 `
  --clean-sft-adapter results\qlora_v4\clean_sft\checkpoint_final `
  --output-dir results\qlora_v4\continued_sft

& $v4Python scripts\train_preference_v4.py `
  --arm dpo `
  --pair-file data\processed\qlora_v4\preference\train_schedule.jsonl `
  --expected-pair-file-sha256 f3cd0565cab0fd12252512b018a749dbe2c42a89d15ec92efd6b03a18f521341 `
  --clean-sft-adapter results\qlora_v4\clean_sft\checkpoint_final `
  --output-dir results\qlora_v4\dpo
```

Defaults are part of the frozen contract: 144 rows, 18 optimizer steps,
gradient accumulation 8, learning rate `1e-5`, constant scheduler, and zero
warmup. The runner must assert them rather than relying silently on defaults.
The angle-bracket value above is supplied by the runner from the frozen
config; it is not an operator-entered value.

Preference smoke uses a separate frozen 16-row schedule:

```powershell
& $v4Python scripts\train_preference_v4.py `
  --arm <continued_sft-or-dpo> `
  --pair-file data\processed\qlora_v4\preference\smoke_pairs.jsonl `
  --expected-pair-file-sha256 86fd923875ba3d11c50d635c409246e6f09437c771a4b4881c02cbba47190eb4 `
  --expected-longest-pair-id sonnet-35-new-retail:task106:trial4:action28:prefer_repair_over_action26 `
  --clean-sft-adapter results\qlora_v4\clean_sft\checkpoint_final `
  --output-dir <new-smoke-output-path> `
  --smoke-test
```

Both smoke arms must use that same file, which contains 16 distinct pairs,
and run two optimizer steps.

Pair scoring is a separate, read-only trainer mode:

```powershell
& $v4Python scripts\train_preference_v4.py `
  --mode score `
  --score-adapter <checkpoint_final> `
  --pair-file data\processed\qlora_v4\evaluation\test_preference_pairs.jsonl `
  --expected-pair-file-sha256 b85548e9f1c041032358172e10b7f7f53f91710d1d15f26dfaa606a07799cf74 `
  --score-split test `
  --output-dir <new-empty-score-directory>
```

Each scoring directory path must not exist before launch and must contain
`metrics.json`, `pair_scores.jsonl`, and `score_manifest.json` on success.
Score each new checkpoint separately; do not reuse cached policy
log-probabilities across adapters.

Expected formal artifacts:

```text
data/processed/qlora_v4/build_summary.json
data/processed/qlora_v4/contract_audit.json
data/processed/qlora_v4/clean_sft/train_unique.jsonl
data/processed/qlora_v4/clean_sft/train_schedule.jsonl
data/processed/qlora_v4/preference/train_pairs.jsonl
data/processed/qlora_v4/preference/train_schedule.jsonl
data/processed/qlora_v4/preference/smoke_pairs.jsonl
data/processed/qlora_v4/evaluation/test_preference_pairs.jsonl
data/processed/qlora_v4/evaluation/test_outcomes.jsonl
results/qlora_v4/clean_sft/
results/qlora_v4/continued_sft/
results/qlora_v4/dpo/
results/qlora_v4/pair_scores/clean_sft/
results/qlora_v4/pair_scores/continued_sft/
results/qlora_v4/pair_scores/dpo/
results/analysis_v4/comparison.json
artifacts/qlora_v4/rtx5060/UPLOAD_MANIFEST.json
```

## 9. OOM and interruption policy

Training is not considered safely resumable. Keep a failed output directory as
evidence and restart from the same Clean-SFT checkpoint into a new output
directory only after identifying the cause.

The runner implements this as an explicit, recoverable retry. First confirm
that no matching Python/CUDA process is alive, then rerun the failed stage with
`--archive-partial-output`. The runner moves the fixed partial output under
`results/qlora_v4/failed_attempts/<timestamp>_<stage>` before starting. Never
use this flag while the original process is still alive.

Evaluation must save one prediction at a time under a frozen contract. After a
disconnect:

1. check `nvidia-smi` and Windows process command lines;
2. if the original process is alive, do not start a duplicate;
3. if it exited, resume the identical evaluation command;
4. never alter or manually complete predictions or metrics.

For CUDA OOM:

1. close unrelated GPU programs and retry the exact smoke in a fresh process;
2. verify NF4, batch 1, dynamic padding, gradient checkpointing,
   precomputed reference log-probabilities, and `use_logits_to_keep`;
3. if the maximum-length smoke still OOMs, do not shorten prompts, drop long
   pairs, alter the pair schedule, or silently change one arm;
4. stop the 5060 DPO run and move the frozen protocol to the 4090.

A mathematically equivalent sequential chosen/rejected backend is permissible
only if implemented, numerically compared with the concatenated TRL backend,
tested before formal execution, and frozen for every DPO run. It is not an
after-the-fact emergency edit.

## 10. Evaluation and interpretation

All three new checkpoints receive the exact same 959-example NF4 generation
evaluation as V3:

```text
prebuilt prompt <= 1664 tokens
no runtime truncation
greedy decoding
max_new_tokens=128
batch size 1
959/959 required
```

Evaluate each checkpoint in a fresh Python process and release CUDA state
before launching the next one. Never keep multiple base-model copies or all
three adapters resident on the 8 GB GPU.

Report:

- overall tool accuracy and full-call exact match;
- **outcome-success**: full-call exact match on all 902 targets whose observed
  tool outcome succeeded;
- **non-recovery-success**: full-call exact match on the 852 successful
  ordinary targets with no preceding error;
- **recovery-success**: full-call exact match on the 50 verified repair
  targets after a prior error;
- **failed-action imitation rate**: canonical exact match to the known failed
  gold call on the 57 failed-gold rows, where lower is better;
- exact failed-call replay rate on the 48 strict preference-pair rows;
- pair-ranking accuracy;
- chosen-minus-rejected log-probability margin;
- parseable canonical tool-call validity;
- task-cluster paired confidence intervals;
- actual train tokens, wall time, GPU time, and peak VRAM.

Outcome-success, non-recovery-success, and recovery-success are offline
next-action metrics. They are not executable task success.

These three subsets are mutually exclusive and exhaust all 959 rows
(`852 + 50 + 57`). Overall full-call exact match is retained only for
continuity with V2/V3: it rewards copying the 57 failed-gold labels and must
not be interpreted as the primary V4 success metric. Outcome-success uses the
902-row union of the 852 non-recovery and 50 recovery subsets.

Pair ranking must use a separately built high-confidence pair set from the
test split, under the same construction rules as training. The current audit
contains 48 strict test pairs. Its pair count and
SHA256 must be frozen, its training-task overlap must be zero, and it must
never be used for training or scheduling. Ranking the 79 training pairs is a
diagnostic only and cannot be reported as held-out pair-ranking accuracy.
Here “held-out” means training-disjoint only; this test has already been
inspected and is not an independent paper-confirmation set.

On \(N_R=50\) recovery rows, define

\[
\mathrm{RecoverySuccess}
=\frac{1}{N_R}\sum_i \mathbf{1}[\hat a_i=a_i^+].
\]

Only \(N_P=48\) of those rows satisfy the frozen high-confidence preference
pair rule. On that strict subset, define

\[
\mathrm{ExactFailedCallReplayRate}
=\frac{1}{N_P}\sum_i \mathbf{1}[\hat a_i=a_i^-].
\]

The remaining two recovery rows are included in RecoverySuccess but excluded
from replay and pair-ranking denominators. Offline generation cannot determine
whether a novel predicted call would trigger the same environmental error
family, so V4 does not report a `same_error_family_rate`; that requires
executable tool evaluation.

after canonical tool-call normalization. On the \(N_F=57\) failed-gold rows,

\[
\mathrm{FailedActionImitationRate}
=\frac{1}{N_F}\sum_i\mathbf{1}[\hat a_i=a_i^{failed}].
\]

On held-out preference pairs,

\[
\mathrm{PairRankAcc}
=\frac{1}{N_P}\sum_i
\mathbf{1}[
\log\pi_\theta(a_i^+\mid h_i)>
\log\pi_\theta(a_i^-\mid h_i)].
\]

The primary ranking score uses summed completion log-probabilities including
EOS, matching the DPO convention. Also report a per-token normalized margin
as a length-bias diagnostic; do not substitute it for the frozen primary
metric after seeing results.

The central DPO test is:

\[
\operatorname{RecoverySuccess}_{DPO}>
\operatorname{RecoverySuccess}_{continued-SFT}
\]

and

\[
\operatorname{ExactFailedCallReplayRate}_{DPO}<
\operatorname{ExactFailedCallReplayRate}_{continued-SFT}
\]

without material degradation in non-recovery-success. Pair-ranking improvement
alone is insufficient to claim better generated recovery behavior.

The frozen directional gate for advancing beyond V4 requires all four point
estimates: DPO minus continued-SFT recovery-success at least `+0.04`, strict
failed-call replay at most `-2/48`, summed-log-prob pair-ranking at least
`+3/48`, and non-recovery-success at least `-0.02`. Passing this gate only
triggers fresh three-seed confirmation. A statistically supported exploratory
signal additionally requires the paired task-cluster 95% interval for
recovery-success to be above zero and the non-recovery lower bound to be at
least `-0.02`.

No result from the already inspected 959 examples is paper-final. The final
method selected by this screen must be evaluated on fresh uninspected tasks
with three training seeds and task-cluster paired uncertainty.

## 11. Expected RTX 5060 time

Assuming a warm model cache:

| Stage | Expected time |
| --- | ---: |
| CPU preparation and audit | 15–60 minutes |
| all smoke gates | 20–45 minutes |
| Clean-SFT formal training | 15–25 minutes |
| continued-SFT formal training | 5–20 minutes |
| reference precompute + DPO training | 25–80 minutes |
| three 959-example evaluations | 6–12 hours |
| three 48-pair scoring passes | 15–60 minutes |
| aggregate, package, upload | 15–40 minutes |
| total | about 8–18 hours |

Cold installation/downloads may add 1–2 hours, for about 9–20 hours total. An
eight-hour night is not a safe promise; reserve up to 20 hours or split
evaluation across two nights.

## 12. Result integrity

Aggregation must reject:

- unresolved source or artifact hash placeholders;
- a failed-action label in Clean-SFT;
- a pair/schedule count other than 79/144;
- a mode schedule other than 72/72;
- a second-stage run other than 18 optimizer steps;
- different Clean-SFT initialization hashes across second-stage arms;
- a modified DPO reference adapter;
- a missing, overlapping, or unfrozen held-out preference-pair set;
- prompt, EOS, tokenizer, model, or evaluator drift;
- limited or partial evaluation;
- manually created metrics.

Only a complete audit package may be uploaded. Preserve negative and null
results exactly as generated.
