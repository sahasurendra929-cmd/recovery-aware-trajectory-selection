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
