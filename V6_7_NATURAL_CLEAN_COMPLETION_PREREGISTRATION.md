# V6.7 natural clean-completion preregistration

Date frozen: 2026-07-30 (Asia/Shanghai)

Status: **preregistered before implementation and before any V6.7 outcome**

## Predecessor result

V6.6 is preserved as a targeted-smoke NO-GO. Its `retail:104` single-turn
prefix and all five deterministic reference actions were correct. The DB,
action, and communicate checks passed, but the strict NL judge rejected both
clean attempts because the assistant's final message echoed a meta-level
evaluation assertion:

```text
Agent should provide the tracking number 286422338955.
```

No V6.6 Pilot was authorized and the official test remained sealed.

## Only scientific change

V6.7 changes only the renderer for the **deleted deterministic clean future's
final assistant message**.

The renderer:

1. emits each frozen `communicate_info` value in direct user-facing text;
2. converts each frozen `nl_assertion` from evaluator/meta language into a
   direct declarative confirmation using a fixed, deterministic,
   outcome-independent rewrite table;
3. never copies a remaining `Agent`, `agent`, `Check that`, `should`, or
   evaluator-instruction phrase into assistant speech;
4. fails closed on an assertion shape not covered by the frozen rewrite table;
5. does not expose the rendered message or any other clean future content to
   recovery prompts or SFT inputs.

The rewrite table is frozen from the syntax of `nl_assertions` in the
outcome-independent registered 24 Pilot and 50 formal task sets, inspected
before V6.7 outcomes. It includes:

- `Agent should tell the user X` -> `X`;
- `Agent should provide X` -> `Here is X`;
- `Agent should not approve/cancel/offer X` -> `I did not ... X`;
- `Agent should cancel/book/exchange/modify X` -> `I ...ed X`;
- `Agent should realize that X` -> `X`;
- `Agent communicates/communicated/mentions that X` -> `X`;
- present-tense action summaries (`Agent updates`, `assigns`, `add`,
  `cancels`, `books`, `charges`, `verifies`, `does not ...`) -> direct
  first-person past-tense confirmations;
- `Check that Agent clearly identifies that X` -> `X`;
- `Check that agent correctly adds X` -> `I added X`;
- already declarative assertions without evaluator language are preserved.

Punctuation and numeric/string values are preserved. The strict judge still
evaluates against the unchanged frozen `nl_assertions`; the renderer does not
change evaluation criteria or reward aggregation.

## Frozen invariants

Everything else remains identical to V6.6:

- single-turn assistant-greeting/first-user-response prefix;
- user simulator model, revision, seed, decoding, and stop boundary;
- deterministic clean reference actions and official reward requirement;
- V6.5 deterministic forced-correction/reference-completion recovery;
- task IDs, registry pairs, injected errors, and continuation seeds;
- teacher, judge, student, tokenizer, and all revisions;
- Pilot/formal gates, token and hardness definitions;
- selector manifests and `full_proposed` weights
  (`causal=0.50`, `hardness=0.25`, `coverage=0.25`);
- training budget, training seed, evaluation seeds, primary metric,
  non-inferiority margin, and bootstrap method;
- official-test seal and one-time unseal rule.

V6.7 results must not be pooled with V6 through V6.6.

## Required preflight and decision

Before a V6.7 Pilot:

1. implement the renderer as a pure, unit-tested function;
2. test all registered Pilot/formal assertion strings and fail on unknown
   meta-language;
3. pass the complete V6 test suite;
4. build a fresh V6.7 registry and run `retail:104` smoke in a fresh directory;
5. require clean reward `1.0`, three pairs, six branches, all forced-first
   cells and repeated trials, all independent replays and label audits, and
   `official_test_used=false`.

Smoke failure closes V6.7 as NO-GO and must be reported before any successor.
Smoke success authorizes the unchanged fixed 24-task / 72-pair Pilot.
