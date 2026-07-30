# V6.10 official-evaluation freeze preregistration

Status: design frozen before any V6.10 official-test task content is loaded by
an evaluator or exported to a model worker.

This document closes the evaluator and task-cluster summarizer blockers named
in `V6_10_RUN_HANDOFF.md`. It does not authorize Pilot by itself. Pilot remains
conditional on a source-bound compatibility authorization produced after the
implementation and tests described here are committed.

## 1. Scope and claim boundary

The fast directional screen compares the three already preregistered arms:

```text
flawless_only
random_stratified
full_proposed
```

The primary contrast is:

```text
full_proposed - random_stratified
```

The independent unit is `task_id`. Evaluation seeds are repeated
measurements, not independent tasks. The official test remains sealed during
compatibility, Pilot, formal candidate generation, measurement, selection,
training, and checkpoint selection.

The evaluator and summarizer are implementation infrastructure. Their
code-only tests must not load official-test task content.

## 2. Frozen executables

The release must contain and hash:

```text
scripts/build_v6_official_unseal_receipt.py
scripts/run_v6_end_to_end_eval.py
scripts/summarize_v6_task_clusters.py
```

The unseal builder refuses to overwrite an existing receipt. It binds the
source commit, Tau2 commit, release manifest, formal-pool freeze, selector
manifests, materialization audits, training manifests, checkpoint registry,
split manifest, evaluator bytes, summarizer bytes, task population, and all
evaluation/statistical seeds and thresholds.

The evaluator refuses to run unless the supplied unseal receipt and every
bound file hash recompute exactly. The summarizer accepts only complete,
hash-bound evaluator rows produced by the frozen evaluator.

## 3. Official task population and conditions

The official population is exactly the 60 sealed IDs in the immutable split
manifest:

```text
retail:  40 tasks
airline: 20 tasks
```

Each arm is evaluated under both conditions for every task and each frozen
evaluation seed:

```text
clean
controlled_error
```

The evaluation seeds are:

```text
20260722
20260723
20260724
```

No task, condition, seed, arm, or failed rollout may be dropped. A missing,
duplicate, malformed, or extra grid row makes the evaluation incomplete.

On the exact three-GPU screen host, evaluation uses two identical 7B agent
services (GPU 0 and GPU 2), each exposing all three frozen LoRA aliases, plus
one shared frozen 14B user/judge service (GPU 1). Tasks are assigned by the
canonical task-identity order to exactly two shards and the two shards run
concurrently. Sharding changes throughput only: both shards use identical
model revisions, adapter hashes, prompts, decoding, task manifests, and
evaluation seeds. Every row records its shard and endpoint receipt.

## 4. Controlled-error construction

The controlled error is selected only after the one-time unseal, before any
arm rollout, by the existing frozen
`v5_multifamily_readonly_faults_v1` construction:

1. load all 60 official tasks and pinned domain databases from the exact Tau2
   commit;
2. enumerate each task's reference-tool names only to determine
   operation-aligned read-only families;
3. balance the preregistered family counts in the deterministic
   `20260722:official_test:<domain>:<task_id>:<family>` hash order;
4. construct one database-absent parameter with the frozen family-specific
   generator and prove absence against the pinned database;
5. freeze the family, tool name, arguments, task identity, and canonical hash
   in one evaluation task manifest before any arm rollout; and
6. inject exactly that read-only call once, require its immediate result to be
   an error, then let the evaluated policy continue normally.

The registered retail families are missing user email, user ID, order ID, and
product ID. The registered airline families are missing reservation ID, user
ID, and flight number. All are read-only and must produce zero state mutation.
Every official task receives exactly one family; there is no task
substitution, outcome-conditioned assignment, or post-rollout adaptation.

The clean condition contains no injected call. The controlled-error condition
must prove that exactly one registered call was executed, its immediate tool
result was an error, and the injected call was not silently repeated by the
injection mechanism.

## 5. End-to-end outcome

Task success is the frozen Tau2 official composite reward after the rollout
ends. Evaluation criteria may be used by the frozen judge only after the
trajectory is produced; they are never included in the agent prompt.

For every row the evaluator records at least:

```text
arm
training_seed
evaluation_seed
domain
task_id
condition
official_reward
task_success
injection evidence
checkpoint identity
model-service identity
unseal-receipt hash
evaluator source hash
official_test_used = true
```

The evaluator uses the registered checkpoint only. There is no checkpoint,
prompt, decoding, retry, task, or metric selection after unseal.

## 6. Statistics

For each task and arm, repeated evaluation seeds are averaged first. The
primary controlled-error task-level value is:

```text
mean_seed(success_full_proposed)
-
mean_seed(success_random_stratified)
```

The reported primary point estimate is the mean of those paired task-level
deltas. Its 95% confidence interval uses exactly 10,000 task-cluster bootstrap
replicates. The bootstrap seed is `20260722`.

Clean non-inferiority uses the same paired task-level construction and the
preregistered margin:

```text
-0.05
```

Clean non-inferiority passes only when the lower endpoint of the 95% cluster
bootstrap interval for `full_proposed - random_stratified` is at least
`-0.05`.

A positive V6.10 directional-screen claim requires all of:

1. complete official end-to-end task-success rows;
2. a positive primary controlled-error point estimate;
3. a primary 95% task-cluster interval whose lower endpoint is above zero;
4. clean non-inferiority; and
5. zero provenance, seal, checkpoint, injection, or coverage failures.

Otherwise the result is preserved as null, negative, or incomplete according
to the typed failing gate. Training loss and next-call exact match are not
substitutes for this end-to-end decision.

## 7. Fail-closed tests

Code-only tests must cover:

- official-test content cannot be loaded without a valid one-time unseal
  receipt;
- every bound artifact and executable hash is recomputed;
- a dirty or wrong source commit is rejected;
- task-manifest construction is deterministic, family-balanced,
  database-absent, and outcome-blind;
- clean rows reject injected faults;
- controlled-error rows reject a missing, changed, repeated, or non-error
  injection;
- checkpoint/model aliases must match the registry;
- the exact arm/task/condition/seed grid is required;
- evaluation seeds are averaged within task before resampling;
- the bootstrap resamples task clusters, not rows or seeds;
- exactly 10,000 replicates and the `-0.05` margin are enforced;
- duplicated, missing, extra, non-finite, or post-unseal-drifted evidence
  fails closed; and
- output files are atomic and never silently overwritten.

## 8. Release consequence

Adding these executables is a tracked source change. Therefore any
compatibility evidence produced by an earlier source commit remains useful
engineering evidence but cannot authorize Pilot. After implementation,
tests, and commit, the release must rebuild:

```text
static reference preflight
-> executable registry
-> runtime receipts
-> release manifest
-> compatibility generation
-> frozen-student measurement
-> compatibility audit
```

Only a source-bound
`COMPATIBILITY_RELEASE_AUTHORIZED` audit plus a `FROZEN` freeze manifest from
that rebuilt release can authorize the prospective Pilot.
