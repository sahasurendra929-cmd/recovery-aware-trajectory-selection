# V6.3 Preregistration — Deterministic clean replay

Status: **frozen before any V6.3 model outcome is observed**

V6.2 is preserved as a candidate-supply NO-GO. Its reference-guided LLM
performed a policy-violating extra cabin change in both registered clean
attempts for `airline:12`; both official rewards were `0.0`.

## Single change from V6.2

The successful clean future is constructed by executing the task's already
frozen Tau2 reference actions deterministically and evaluating that replay
with the official evaluator. The conversational prefix is still sampled with
the frozen teacher, user simulator, seed, and decoding contract, and ends
immediately before the first assistant tool action.

Any frozen `communicate_info` and NL assertions are rendered only into the
deterministic clean future. That entire future is deleted before the injected
error or recovery model call. It cannot enter the recovery prompt or SFT
labels. A clean source is eligible only if its reference actions execute
without error, official reward equals `1.0`, and independent state replay and
hash audits pass.

## Unchanged contract

The 72B recovery teacher, 14B user/judge, task IDs, seeds, attempts, injected
errors, matched/crossed corrections, continuation generation, audits, gates,
selector definitions and weights, student, training budgets, evaluation
seeds, primary contrast and metric, non-inferiority margin, bootstrap, and
official-test seal are unchanged from V6.2.

## Estimand boundary

V6.3 estimates selector effects conditional on a successful, reference-derived
clean source. It does not claim that an unconstrained teacher can solve every
clean task. Results are not pooled with V6–V6.2.
