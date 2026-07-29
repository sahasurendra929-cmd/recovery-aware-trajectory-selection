# V6.6 single-turn clean-prefix preregistration

Date frozen: 2026-07-30 (Asia/Shanghai)

Status: **preregistered before implementation and before any V6.6 outcome**

## Motivation and predecessor result

V6.5 is preserved as a Pilot candidate-supply NO-GO. Seven of 24 task
receipts passed, but both frozen clean-source attempts for `retail:104`
received official reward `0.0`. The stochastic teacher/user conversation
consumed the 60-step budget after the user simulator repeatedly supplied
invented mnemonic order IDs. No merged candidate pool was emitted and the
official test remained sealed.

V6.6 does not reinterpret or replace that result.

## Only scientific change

V6.6 changes only the source of the clean conversational prefix:

1. Start the unchanged Tau2 task with the unchanged user simulator model,
   revision, seed, decoding parameters, domain policy, and initial assistant
   greeting.
2. Run exactly through the first user response and stop before any assistant
   tool action.
3. Freeze that assistant-greeting/user-response history as the shared
   conversational prefix.
4. Starting from the unchanged task initial state, execute every frozen Tau2
   reference action in its registered order and append the frozen
   communication assertions in the clean future.
5. Require official clean reward `1.0` and independent replay.
6. Delete the entire clean future before constructing either recovery branch.

The retained prefix is generated only from the task's user scenario. The user
simulator and prefix constructor are not given task evaluation actions,
communicate-info values, natural-language assertions, official rewards, or
official-test content. Reference actions and assertions occur only in the
deleted clean future.

This replaces V6.5's requirement that a stochastic teacher/user conversation
reach its first assistant tool call and ultimately complete the full task
within the rollout budget. It does not change recovery construction.

## Frozen invariants

The following remain identical to V6.5:

- Pilot and formal task IDs and partition hashes;
- registry candidate-pair IDs and error branches;
- Tau2 commit;
- teacher, user/judge, student, and tokenizer revisions;
- user-simulator decoding parameters and registered seeds;
- V6.5 deterministic recovery completion:
  forced corrective action first, then every other frozen reference action
  exactly once in original relative order;
- clean and recovery success requirements;
- continuation seeds and forced-first four-cell causal score;
- token and hardness definitions;
- Pilot and formal GO/NO-GO thresholds;
- selector definitions and `full_proposed` weights
  (`causal=0.50`, `hardness=0.25`, `coverage=0.25`);
- task sets, training budgets, training seed, evaluation seeds, primary
  contrast, primary metric, clean non-inferiority margin, and bootstrap
  procedure;
- official-test seal and one-time unseal rule.

V6.6 results must not be pooled with V6 through V6.5.

## Implementation and audit requirements

Before a formal V6.6 Pilot:

1. Add a distinct clean mode named
   `single_turn_user_reference_replay`; do not silently change the semantics
   of `deterministic_reference_replay`.
2. Bind the mode and the single-turn budget into the semantic run contract.
3. Add unit fixtures proving:
   - the retained prefix ends with the first user response;
   - no assistant tool call occurs in the retained prefix;
   - reference actions and evaluation text occur only after the prefix;
   - the clean future receives official reward `1.0`;
   - the future is absent from recovery prompts;
   - retries/resume fail closed on contract drift.
4. Pass the full V6 code-only test suite.
5. Run a targeted `retail:104` smoke in a fresh V6.6 directory.
6. Audit the smoke's prefix, clean replay, three pairs, six matched branches,
   forced-first cells, independent replays, labels, hashes, and
   `official_test_used=false`.

## Decision rule

- If the targeted smoke fails because the single-turn prefix cannot be
  produced, contains an assistant tool action, leaks evaluation-only content,
  or the deterministic clean/recovery replay fails, V6.6 is NO-GO. Preserve
  and publish the result before any successor.
- If it passes, start the complete fixed 24-task / 72-pair Pilot in a fresh
  V6.6 directory.
- The existing Pilot gate is applied without modification after token and
  frozen-base hardness measurement.

No attempt count, threshold, task, seed, model, selector weight, or downstream
budget may be changed under the V6.6 label.
