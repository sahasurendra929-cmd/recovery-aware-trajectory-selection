# V6.5 forced-correction reference-completion preregistration

V6.5 is preregistered after, and separately from, the V6.4 Pilot NO-GO.
No V6.5 task outcome was observed before this document and its implementation
tests were committed.

## Reason

V6.4 forced the registered corrective action and replayed only reference
actions strictly after that action's reference index. With the frozen initial
prefix, this omitted required earlier actions. On `airline:14`, forcing
reference index 1 (`book_reservation`) omitted index 0
(`cancel_reservation`) and produced official reward zero.

## Sole scientific change

For matched and crossed recovery cells:

1. execute the exact registered corrective action first;
2. remove its unique semantic match from the frozen reference-action list;
3. execute every remaining frozen reference action exactly once;
4. preserve the original relative order among those remaining actions.

No task, threshold, model, revision, seed, decoding setting, selector,
selector weight, token budget, training hyperparameter, metric, evaluation
grid, or statistical method changes. The official test remains sealed.

## Gates

The existing V6 Pilot gates remain unchanged. A smoke run must first pass on
the V6.4 failure task (`airline:14`). The complete fixed 24-task Pilot is then
rerun in a fresh V6.5 directory. Any failure is recorded as V6.5 NO-GO before
another protocol is considered.

## Claim boundary

This construction measures selection among constrained,
reference-completed recovery trajectories. It is not evidence about
unconstrained teacher recovery. V6.5 results cannot be pooled with
V6–V6.4.

