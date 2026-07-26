# Prompt for the RunPod 4×RTX 4090 Codex agent

Replace only `<V5_3_COMMIT>`, then give the text below to the Codex agent on
the single four-GPU RunPod host.

---

You are the sole operator of the V5.3 train-only feasibility pilot for:

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

Required detached commit:

```text
<V5_3_COMMIT>
```

Read these files completely before acting:

```text
V5_3_SINGLE_HOST_HANDOFF.md
configs/v5_3_sft_causal.yaml
scripts/v5_3_protocol.py
scripts/run_v5_3_train_only_pilot.py
scripts/v5_judge_audit_contract.py
scripts/v5_strict_nl_judge.py
```

Your scope ends after a complete `pilot_go_no_go.json` and an auditable pilot
bundle. Do not start formal generation, SFT/QLoRA training, validation
evaluation, or official-test evaluation.

Scientific invariants:

- repository checkout is detached at `<V5_3_COMMIT>` with no tracked changes;
- tau2 commit is
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`;
- Stage-0 split SHA-256 is
  `a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a`;
- the structural GT filter excludes exactly `retail:24`, `airline:0`,
  `airline:10`, `airline:28`, and `airline:34` before any V5.3 outcome;
- the resulting 78 tasks are frozen into 70 arm-train and 8 loss-validation
  tasks with partition SHA-256
  `c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818`;
- pilot is exactly 24 preregistered arm-train tasks: 18 retail and 6 airline,
  all using the homogeneous ground-truth teacher;
- each pilot task has exactly 12 clean and 12 controlled-error attempts;
- pilot trajectories and seeds never enter formal data;
- failed attempts are not rescued, repaired, relabelled, replaced, or extended;
- the official test remains sealed and no metric is fabricated or imputed.

Frozen services:

- GPU 0: `Qwen/Qwen2.5-14B-Instruct-AWQ`, revision
  `539535859b135b0244c91f3e59816150c8056698`, user simulator and strict
  judge, port 8001, tensor parallel size 1;
- GPUs 1–2: `Qwen/Qwen2.5-32B-Instruct-AWQ`, revision
  `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`, trajectory teacher, port
  8011, tensor parallel size 2;
- GPU 3: preflight/monitoring reserve during pilot.

Both endpoints use AWQ, float16, `max_model_len=32768`,
`max_num_seqs=3`, `gpu_memory_utilization=0.90`, vLLM generation config,
Hermes tool parsing, and auto tool choice. Do not move the 32B teacher to one
24GB GPU or introduce a fallback model.

Inspect and save hardware evidence first:

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader
python3 --version
df -h /workspace
```

Fail closed unless exactly four RTX 4090-class GPUs with at least 24,000 MiB
reported memory are visible and `/workspace` is writable.

Use `scripts/run_v5_3_single_host.py` as the sole executor. Run these stages
in order and do not duplicate their internal service/client commands:

```bash
/workspace/venvs/v5-2-train/bin/python \
  scripts/run_v5_3_single_host.py --stage preflight
/workspace/venvs/v5-2-train/bin/python \
  scripts/run_v5_3_single_host.py --stage self-test
/workspace/venvs/v5-2-train/bin/python \
  scripts/run_v5_3_single_host.py --stage prefetch
/workspace/venvs/v5-2-train/bin/python \
  scripts/run_v5_3_single_host.py --stage protocol
/workspace/venvs/v5-2-train/bin/python \
  scripts/run_v5_3_single_host.py --stage pilot
```

Follow `V5_3_SINGLE_HOST_HANDOFF.md` to verify that this controller:

1. create or reuse the persistent repository, tau2 checkout, serving
   environment, and Hugging Face cache;
2. verify exact commits and a clean tracked worktree;
3. run all listed V5.3 tests;
4. construct the 78-task V5.3 manifests and both dynamic audits;
5. construct the 70-task effective arm-train universe and 24-task pilot;
6. starts the two frozen vLLM services;
7. run near-longest and three-concurrent-request preflights for both roles;
8. launch exactly three pilot clients, shard indices 0, 1, and 2, against the
   shared endpoints;
9. wait for all three COMPLETE contracts;
10. runs the exact one-shot audit command and packages the result.

The manual commands later in the handoff are audit/reference material, not a
second workflow. Never start manual tmux services while the controller is
running or rerun the pilot audit outside the controller.

Sampling is frozen:

```text
teacher/user: temperature=0.2, top_p=0.95, max_tokens=512
strict judge: temperature=0.0, top_p=1.0, max two content attempts
max_steps=60, task timeout=900 seconds
pilot base seed=20260730
```

The strict judge must require a nonempty complete result set with exact
assertion coverage, nonempty reasoning, real JSON booleans, and raw response
hashes. A malformed first response may receive one content-format retry. A
second malformed response fails closed. Do not convert malformed or empty
judge output into success. Every simulation with nonempty natural-language
assertions must have exactly one PASS audit in its own task/simulation log
directory; missing, extra, duplicate, misplaced, or assertion-mismatched
audits fail closed.

During generation, monitor read-only:

- all three client processes and both vLLM servers;
- GPU utilization, memory, and temperature;
- output-file growth;
- COMPLETE/INCOMPLETE contracts;
- server OOM/restart/NaN and judge retry exhaustion.

Do not kill a healthy silent process because tau2 lacks progress lines. Stop
on an actual nonzero exit, OOM, server death, source/model/hash/seed drift,
prohibited split access, or contract violation. Never calculate partial
metrics from an incomplete shard.

The complete pilot decision rule is only:

```text
tasks with >=1 pair must be >=15 of 24
AND
tasks with >=2 pairs must be >=4 of 24
```

These imply first/second rates of `0.625` and `1/6`, and projections of 43.75
covered tasks and 55.4167 capped pairs over 70 formal arm-train tasks.
Projections are diagnostics, not extra adjustable gates. The future formal
gate stays 40 distinct tasks and 48 pairs, capped at two pairs per task.

Valid pilot terminal results:

```text
exit 0  / GO_FORMAL_GENERATION
exit 20 / NO_GO_STOP
exit 2  / malformed, prohibited, or incomplete data (fail closed)
```

Even after GO, do not start formal work in this task.

Create a SHA-256 inventory and a small bundle containing:

- effective task universe and pilot manifest;
- `pilot_go_no_go.json`;
- three COMPLETE generation contracts;
- exact commands and complete client/server logs;
- hardware, Python, package, driver, CUDA, PyTorch, vLLM, and endpoint evidence;
- model/revision receipts and preflight logs;
- strict-judge audit files or their hash-bound summary;
- user-stop/DB=0, repeated-user, and max-step diagnostics;
- all raw-result and audit hashes.

Keep raw trajectories, caches, and model weights out of Git. If Git push is
already authorized, push only the small bundle on a new results branch.
Otherwise leave it under `/workspace` and report its path and SHA-256. Never
request or expose a password or token.

Your final response must report:

1. repository and tau2 commits;
2. hardware and exact model revisions;
3. unit-test and preflight results;
4. completed simulations versus 576 expected;
5. tasks with at least one pair and their rate;
6. tasks with at least two pairs and their rate;
7. 70-task projections;
8. strict-judge reject/retry/exhaustion counts;
9. user-stop/DB=0, repeated-user, and max-step counts;
10. exact terminal status and exit code;
11. `training_started=false`;
12. `official_test_used=false`;
13. bundle path/SHA-256 and pushed branch/commit if applicable.

---
