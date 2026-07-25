# Project specification

## Research question

At a fixed training-token budget, does selecting successful tool-use trajectories for verified error-resolution coverage improve an agent's tool-use reliability and recovery on held-out tasks?

## Terminology boundary

- **Trajectory**: ordered user messages, agent actions, tool calls, tool outputs, and eventual outcome.
- **Error-resolution event**: an error output followed by a changed tool action in a final-success trajectory.
- **Agent-initiated recovery**: an error-resolution event with no intervening user message. This is the target concept for the main study.
- The legacy pilot must not call all error-resolution events “agent self-recovery.”

## Frozen pilot contract (v1)

- Data source: τ-bench historical retail trajectories only.
- Split unit: `task_id`.
- Seed: `20260722`.
- Split ratio: 70% train / 10% validation / 20% test.
- Candidate pool: final-success train trajectories after exact tool-sequence deduplication.
- Sampling arms: `random_success`, `shortest_success`, `recovery_balanced`.
- Evaluation claim: offline action prediction only; no end-to-end Agent-success claim.

## Evidence rule

Every reported number must trace to a configuration, a selected-trajectory manifest, a fixed evaluator version, and a results file. A change to split, budget, labels, prompt format, model, or evaluator creates a new experiment version.

## V5 Stage-1 controlled mechanism question

Stage 1 no longer assumes that more recovery trajectories are inherently
better. It asks three separable questions under matched training budgets:

1. Does observing one real, read-only tool failure followed by a successful
   post-fault continuation help relative to flawless-only demonstrations?
2. Is any effect caused by the failure context, or by incorrectly treating the
   failed action as a positive SFT label?
3. Is an intermediate clean/post-fault mixture better than either endpoint?

`failure_raw` is an intentionally harmful diagnostic control, never a proposed
deployment method. In every repair arm the failed assistant action remains in
the input context but receives no loss; only later tool calls with an adjacent
non-error result in a final-success trajectory may be supervised. The primary
screening measurement is paired end-to-end task success on derived validation
tasks after controlled error injection. The official test remains sealed until
the method, mixture, model, and hyperparameters are frozen.

Stage 1 uses the repository-tracked frozen split at
`artifacts/v5_stage0/manifests/split_manifest.json` and writes its isolated,
deterministic multi-fault protocol to `data/processed/v5_stage1_protocol`.
Generated `data/processed/v5_stage0` files are not Stage-1 split inputs. The
protocol assigns one database-absent parameter
for a pinned read-only tool to each of the 83 inner-train and 21 validation
tasks. Assignment prefers a reference-path or operation-aligned tool and then
balances the available fault families. All 104 injected calls must be executed
in a dynamic preflight and must both return an error and preserve the agent and
user database hashes. The historical Stage-0 files are read-only inputs: their
bytes and hashes must not change. In particular, the seed is `20260722` and the
published `split_manifest.json` SHA-256 is
`a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a`.
The 60 official-test tasks remain sealed.

The scientific claim is a **multi-fault-family post-fault task-robustness
screen**, not semantic error repair. A later non-error tool result does not by
itself prove that the call repaired the injected error. The existing
`repair_*` arm names are compatibility identifiers. Results grouped by
`fault_family` are descriptive only because each task receives one family; the
pre-registered 21-task primary gate is unchanged.

The comparison is a controlled data-composition screen, not a fully identified
causal effect: clean and repair examples can match task and compute budgets,
but the failure context and subsequent successful action sequence necessarily
differ. Legacy filenames containing `causal` are protocol identifiers only.
