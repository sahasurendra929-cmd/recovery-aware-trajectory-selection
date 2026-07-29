# V6.4 Preregistration — Forced correction with reference tail

Status: **frozen before any V6.4 outcome is observed**

V6.3 is preserved as a recovery-supply NO-GO. Its deterministic clean replay
passed, but both registered 72B fresh continuations for the first candidate
branch changed unrelated flights, dates, and cabin state and received official
reward `0.0`.

## Single change from V6.3

Recovery now consists of:

1. the branch's frozen forced corrective action;
2. the frozen Tau2 reference actions strictly after that action's unique
   position in the task resolution;
3. frozen communication/NL facts rendered in the completion message.

The forced action must match exactly one reference action and execute without
error. Every remaining reference action must execute without error. The full
trajectory must receive official reward `1.0` for a matched candidate and pass
independent replay. Matched and crossed cells use the same tail rule. Registered
continuation seeds remain cell identifiers/repeated measurements but do not
introduce stochastic decoding under this deterministic policy.

## Unchanged contract

V6.3 clean construction, tasks, errors, pair registry, matched/crossed cell
layout, gates, selector definitions and weights, student, training budgets,
evaluation seeds, primary contrast and metric, clean non-inferiority margin,
bootstrap, and official-test seal are unchanged.

## Claim boundary

V6.4 studies selection among constrained/reference-completed recovery
trajectories. It does not claim unconstrained teacher recovery and is not
pooled with V6–V6.3. If deterministic reference-tail construction cannot
supply audited choice sets, V6.4 is reported as NO-GO rather than altering
tasks, thresholds, or selectors.
