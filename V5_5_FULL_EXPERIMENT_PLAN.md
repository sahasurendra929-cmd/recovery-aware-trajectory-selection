# V5.5 Full Experiment: Recovery-Data Dose Response

## 1. Status and purpose

V5.5 Phase A has established that counterfactual recovery pairs can be
constructed and independently replay-audited. The local integration run
produced 48/48 valid pairs over 24 tasks and two domains. That result is a data
engineering result, not a model result.

The complete experiment asks:

> Under matched training and supervised-token budgets, what fraction of
> masked-failure recovery trajectories maximizes post-error end-to-end task
> success without materially reducing clean task success?

This preserves the original research direction: **recovery-aware trajectory
selection**. It does not change the project into a new objective-function
paper.

## 2. Why the uploaded 48 pairs are not sufficient by themselves

The Phase-A pairs use tau2 reference actions. A reference action path is one
valid route to the target database state; it is not necessarily a natural
multi-turn conversation. Directly treating all 48 pairs as paper-quality
dialogue SFT data would introduce a train/evaluation format mismatch.

The complete study therefore separates two evidence levels:

1. **Reference-grounded mechanism screen.** Cheaply checks whether a model can
   learn the masked-error/correct-recovery pattern. It cannot support a broad
   natural-agent claim.
2. **Natural-conversation confirmatory study.** Uses successful tau2
   agent–user conversations and counterfactual branches as the actual paper
   training pool.

Phase A remains useful as an executable oracle and code test. It is not
discarded or relabeled.

## 3. Hypotheses

Let \(r\in\{0,.25,.50,.75,1\}\) be the fraction of supervised-token mass drawn
from recovery trajectories.

### Primary hypothesis

\[
H_1:\quad
SR_{\mathrm{error}}(r^\*)-
SR_{\mathrm{error}}(0)>0,
\]

where \(r^\*\) is selected only on derived validation and \(SR\) is official
tau2 end-to-end task success.

### Clean non-inferiority

\[
H_2:\quad
SR_{\mathrm{clean}}(r^\*)-
SR_{\mathrm{clean}}(0)\ge -0.05.
\]

The five-point margin is frozen before training.

### Dose-response hypothesis

An intermediate mixture is expected to dominate the endpoints:

\[
r^\* =
\arg\max_r
\left[
SR_{\mathrm{error}}(r)
- \max(0,SR_{\mathrm{clean}}(0)-SR_{\mathrm{clean}}(r)-0.05)
\right].
\]

This is a validation selection rule, not a claim that 50% must win.

## 4. Data construction

### 4.1 Frozen partitions

| Partition | Tasks | Use |
|---|---:|---|
| inner train | 83 | data generation and training only |
| derived validation | 21 | arm selection and debugging |
| sealed official test | 60 | one final confirmation |

No validation or official-test outcome may select training examples.

### 4.2 Natural clean trajectory

For each structurally eligible inner-train task:

1. run a frozen, ground-truth-guided strong teacher and a separately frozen
   user simulator;
2. require official task reward \(=1\);
3. require no tool error in the clean trajectory;
4. require exactly one assistant tool call per turn;
5. require an independently executable, read-only identifier lookup;
6. cap accepted clean sources at two per task.

### 4.3 Counterfactual recovery branch

For an accepted clean trajectory:

1. freeze a read-only injection position before observing recovery outcome;
2. copy the exact conversational prefix \(h\);
3. replace one identifier in the selected correct call \(a^\*\) with a
   deterministic, schema-valid wrong identifier \(\tilde a\);
4. execute \(\tilde a\) in the real environment and require a real error \(e\);
5. keep the clean future out of the recovery **prompt**;
6. use the original correct call and successful clean suffix as the positive
   counterfactual target;
7. independently replay the failed read, correction and suffix, requiring
   official task reward \(=1\);
8. optionally generate a teacher-replanned suffix as a separate realism
   ablation, never as a replacement for a failed registered row.

The clean and recovery paths are:

\[
h\rightarrow a^\*\rightarrow s_c
\]

and

\[
h\rightarrow\tilde a\rightarrow e\rightarrow
a_r\rightarrow s_r.
\]

Because the injected read-only failure does not mutate state, reusing the
successful clean suffix as a label is a valid counterfactual insertion. This
is not future leakage: the suffix is unavailable in the input and is precisely
the target being learned. A separately generated suffix may take another valid
route, but belongs to an ablation.

### 4.4 Full data gate

The natural-conversation training screen is prohibited unless the pool
contains:

- at least 48 eligible pairs;
- at least 24 distinct task IDs;
- no more than two pairs per task;
- both airline and retail;
- zero official-test access;
- zero failed actions selected as positive labels;
- zero future leakage;
- independently verified task success and environment replay.

The independent statistical unit remains `task_id`, not trajectory count.
The stronger target for opening the official test is 60 pairs over 30 tasks.

## 5. Labels and loss

For a recovery row, the injected failed call and error response are visible
context, but never label tokens. Let \(M\) contain only successful assistant
actions after the error:

\[
\mathcal L_{\mathrm{recovery}}(\theta)
=-\sum_{t\in M}
\log p_\theta(y_t\mid h,\tilde a,e,y_{<t}).
\]

For a clean row:

\[
\mathcal L_{\mathrm{clean}}(\theta)
=-\sum_{t\in M_c}
\log p_\theta(y_t\mid h,y_{<t}).
\]

For recovery dose \(r\):

\[
\mathcal L_r =
(1-r)\mathcal L_{\mathrm{clean}}
+r\mathcal L_{\mathrm{recovery}}.
\]

The ratio is measured by **supervised-token mass**, not number of JSONL rows.

## 6. Training arms

| Arm | Perfect tokens | Recovery tokens | Purpose |
|---|---:|---:|---|
| R0 | 100% | 0% | standard perfect-only SFT |
| R25 | 75% | 25% | small recovery dose |
| R50 | 50% | 50% | balanced mixture |
| R75 | 25% | 75% | recovery-heavy mixture |
| R100 | 0% | 100% | recovery-only endpoint |

All arms use Qwen2.5-7B-Instruct QLoRA with:

- the same task pool;
- the same total supervised-token budget;
- the same optimizer steps and microbatch schedule;
- the same non-padding-token budget within 1%;
- the same LoRA configuration, learning rate and sequence limit;
- three training seeds: 20260805, 20260806 and 20260807.

R0 versus R100 answers whether recovery context changes behavior. The
intermediate arms identify the clean/recovery trade-off.

DPO, PPO, TRPO and ETO are intentionally excluded from V5.5. Adding an
objective change would confound the data-selection variable. A later study may
apply the best V5.5 mixture to an objective comparison.

## 7. Evaluation

Each checkpoint is evaluated on the same task IDs and evaluation seeds under
four conditions.

### 7.1 Clean

No external error is inserted. This measures ordinary task competence and
catastrophic specialization.

### 7.2 Controlled error, in-family

Insert an unseen wrong-identifier perturbation using a frozen task/location
schedule. The primary metric is end-to-end task success, not next-call exact
match.

### 7.3 Controlled error, out-of-family

Use held-out error families such as a wrong read-only tool with compatible
schema or a stale identifier. This tests whether the model learned a recovery
principle rather than one typo template.

### 7.4 Natural agent errors

Run without intervention and describe trajectories where the model itself
makes a tool error. This is diagnostic because the set of naturally failing
tasks differs by arm and cannot be treated as the same randomized intervention.

### Metrics

Primary:

\[
SR=\frac{\#\text{tasks with official tau2 reward }1}
{\#\text{evaluated tasks}}.
\]

Secondary:

- recovery success conditional on an injected error;
- repeated-same-error rate;
- clean task success;
- invalid tool-call rate;
- tool calls per successful task;
- wall time, GPU-hours and pairs per GPU-hour.

Tool-call exact match may be reported only as a code diagnostic.

## 8. Statistical analysis

- Independent unit: task ID.
- Comparisons: paired on task, condition and evaluation seed.
- Uncertainty: 10,000-replicate task-cluster bootstrap, 95% CI.
- Primary test: selected \(r^\*\) versus R0 on sealed-test in-family error
  success.
- Clean analysis: five-percentage-point non-inferiority margin.
- Secondary arm/error-family tests: Holm correction.
- Report every seed and task-level outcome; do not count multiple trajectories
  from one task as independent samples.

With 21 validation tasks, uncertainty will be wide. Validation selects the arm
and detects gross failures; the 60-task official test provides the primary
course-paper confirmation.

## 9. Staged execution and stop rules

### Stage A — completed mechanism/data gate

Reference-grounded 24-task/48-pair construction and independent replay.

### Stage B — guaranteed reference-grounded mechanism screen

Build R0, R50 and R100 from the already audited 48-pair pool and run one
training seed on derived validation. This stage is designed to reach an
evaluation result even if natural-pool generation is delayed. Its result is
diagnostic because reference actions are not natural conversations.

### Stage C — natural data pilot

Generate successful clean conversations on inner-train only, then insert the
failed read before the already successful suffix. The training gate is
24 tasks/48 pairs; the paper target is 30 tasks/60 pairs. Do not weaken either
after observing model outcomes.

### Stage D — natural three-arm screen

Train R0, R50 and R100 with one seed. Evaluate all 21 derived-validation tasks.

Continue when either R50 or R100 improves controlled-error success over R0 and
R50 clean success is no more than five points below R0. A null result is still
reported; it is not replaced with another metric.

### Stage E — full dose response

Train all five arms with three seeds. Select one non-zero recovery arm using
the frozen validation utility. Freeze checkpoints, evaluator and analysis.

### Stage F — one-time confirmation

Open the 60-task official test once and compare only:

- R0;
- validation-selected \(r^\*\).

Run clean, in-family and out-of-family conditions. The natural-error analysis
remains diagnostic. Official-test access additionally requires the stronger
30-task/60-pair natural-data target.

### Stage G — optional model-size replication

Repeat R0 versus \(r^\*\) with Qwen2.5-14B-Instruct. This strengthens a paper
but is not required for the first course result.

## 10. Interpretation matrix

| Observed result | Supported conclusion |
|---|---|
| error success rises, clean is non-inferior | recovery-aware selection improves controlled robustness |
| error rises but clean drops >5 points | recovery specialization has a measurable cost |
| only in-family error rises | model learned a narrow error template |
| no arm beats R0 | audited recovery exposure is insufficient under this model/budget |
| reference screen rises, natural study does not | gain is caused by the synthetic representation |
| natural and out-of-family both rise | evidence for broader recovery behavior |

No outcome licenses the statement “the model learned reflection” without an
additional mechanism analysis.

## 11. Compute and four-GPU schedule

The fast screen uses three GPUs for R0/R50/R100 and one GPU for evaluation
services or data generation. The full five-arm × three-seed grid contains 15
training runs and should be queued across four GPUs.

Approximate 4×4090 planning budget:

- natural data pilot: 2–8 hours, dominated by teacher/user generation;
- three-arm screen: 1–3 hours training plus 2–6 hours evaluation;
- full 15-run grid: 4–10 wall-clock hours training;
- full validation and sealed-test evaluation: potentially 8–24 hours,
  depending on rollout length and service concurrency.

These are planning ranges, not guaranteed runtimes.

## 12. Claim ladder

1. **Already supported:** counterfactual pairs can be built and
   independently audited.
2. **After Stage B:** a first, diagnostic controlled validation result.
3. **After Stage D:** preliminary natural-conversation recovery signal.
4. **After Stage E:** dose-response and clean/recovery trade-off on
   development data.
5. **After Stage F:** paper-level evidence for or against improved held-out
   controlled-error task success.
6. **After Stage G:** preliminary evidence that the effect is not unique to
   one model size.
