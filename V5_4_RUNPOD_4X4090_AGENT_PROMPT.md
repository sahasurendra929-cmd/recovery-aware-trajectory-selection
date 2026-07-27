# Prompt for the V5.4 pilot operator on one 4×RTX 4090 RunPod

Replace `<V5_4_COMMIT>` with the reviewed full commit from the release branch.

---

You are the sole operator of the V5.4 counterfactual recovery-branch **pilot**
for:

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

Checkout detached at:

```text
<V5_4_COMMIT>
```

Read completely:

```text
V5_4_COUNTERFACTUAL_PILOT_HANDOFF.md
configs/v5_4_counterfactual_pilot.yaml
scripts/v5_4_pilot_protocol.py
scripts/prepare_v5_4_pilot.py
scripts/build_v5_4_branch_manifest.py
scripts/audit_v5_4_pilot.py
```

Scope: implement and validate the pinned tau2 live adapter, run the one-task
smoke barrier, and only after it passes run the bounded 24-task pilot. Do not
train, evaluate a student model, access the official test, relax a gate, add
replacement attempts, or reuse V5.3 generated trajectory bytes.

Reuse `/workspace/venvs` and `/workspace/cache/huggingface`. Do not reinstall
CUDA, PyTorch, vLLM, or pinned packages. Verify four RTX 4090 GPUs, persistent
storage, exact repository/tau2 commits, exact model revisions, a clean tracked
checkout, and GitHub result-push access before paid generation.

Create the protocol registry first and run all V5.4 tests. Then implement the
live adapter by constructing a copied tau2 task whose `InitialState` preserves
the original initialization data/actions and whose `message_history` ends at
the actual injected tool-error result. The recovery run must therefore route
the next turn to the agent. Do not feed the removed clean suffix through any
prompt, cache, judge input, selection rule, or label builder.

Before the full run, demonstrate on one preregistered task:

```text
clean success
deterministic read-only injection
schema-valid task-incorrect call
actual consequential tool error
prefix replay without mismatch
fresh recovery generation
strict-judge receipt
zero future leakage
failed call and error result masked
all smoke slots terminal
```

Stop with `SMOKE_FAIL_NO_RUN` if any item fails. Do not patch scientific
thresholds in response.

After smoke PASS, execute the frozen per-task state machine: eight clean slots
followed by four recovery slots, with clean early stop and the two-pair cap.
Unused slots receive their specified terminal status and are never reassigned.
Store GPU seconds and wall seconds on every executed slot.

Run the final audit exactly once after all 288 slots are terminal. Publish a
small result branch containing protocol files, terminal slot ledger, eligible
pair evidence, cost report, decision, logs, and SHA-256 inventory. Keep raw
large trajectories and model files on `/workspace`.

Final response must state exact commits/models, smoke result, executed and
terminal slots, paired tasks, capped pairs, pairs per executed slot,
pairs/GPU-hour, audit rejection counts, official_test_used=false,
training_started=false, GO/NO-GO status, artifact path/hash, and pushed
branch/commit.

---
