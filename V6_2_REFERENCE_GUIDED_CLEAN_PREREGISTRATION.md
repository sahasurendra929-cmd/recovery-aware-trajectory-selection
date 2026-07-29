# V6.2 Preregistration — Reference-guided clean source

Status: **frozen before any V6.2 model outcome is observed**

V6.1 is preserved as a candidate-supply NO-GO: its pinned 72B teacher failed
both registered clean attempts on the first fixed Pilot task, `airline:12`.
This repeated the original V6 failure and showed that teacher scaling alone
did not repair the stochastic clean-source prerequisite.

## Single change from V6.1

Only the agent used to construct the successful clean source trajectory is
changed to Tau2's `llm_agent_gt`. It receives the task's already-frozen
reference actions, including arguments. The source rollout must still earn
official reward `1.0`, contain an assistant tool action, and pass independent
replay and state-hash audits.

The observed clean future is deleted before any recovery rollout. The injected
error, frozen shared prefix, matched/crossed corrective action, fresh recovery
suffix, continuation seeds, and all candidate audits remain unchanged. The
72B teacher remains responsible for every fresh recovery suffix and is not
shown the deleted clean future.

## Unchanged contract

Tasks and task IDs, two clean attempts, two recovery attempts, model revisions,
candidate definition, Pilot/formal gates, selector definitions and weights,
student, training budgets, evaluation seeds, primary contrast and metric,
clean non-inferiority margin, 10,000-replicate task-cluster bootstrap, and the
official-test seal are identical to V6.1.

## Interpretation and stopping

V6.2 estimates selector effects conditional on a reference-guided successful
clean source; it is not pooled with V6 or V6.1. A V6.2 generator, replay, or
Pilot gate failure is reported as its own NO-GO. Engineering defects may be
fixed without changing this contract. Any scientific change requires another
explicitly preregistered version.
