# V6.6 targeted smoke NO-GO

Date: 2026-07-30 (Asia/Shanghai)

## Decision

V6.6 `v6_6_single_turn_user_reference_replay_v1` is **NO-GO** at its
preregistered `retail:104` smoke. No V6.6 Pilot is authorized.

The official test remained sealed and unused. No candidate pair, training
data, checkpoint, or evaluation result was emitted.

## Frozen identity

- final smoke implementation commit:
  `5043998da320500103480b6be8cff9387a73e472`
- Tau2 commit:
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- teacher revision:
  `698703eae6604af048a3d2f509995dc302088217`
- user/judge revision:
  `539535859b135b0244c91f3e59816150c8056698`
- target task: `retail:104`
- clean attempts: 2
- clean-prefix source: assistant greeting plus the first frozen user-simulator
  response, stopped before any assistant tool call
- recovery mode: unchanged V6.5 deterministic reference completion

The V6.6 registry contains the unchanged 24 Pilot tasks / 72 Pilot pairs and
50 formal tasks / 150 formal pairs.

## Result

Both single-turn prefixes were produced correctly. They contained exactly the
assistant greeting and first user response, with no assistant tool call,
tool result, evaluation action, or official-test content.

For each clean attempt:

- all five frozen `retail:104` reference actions executed successfully;
- all five action checks were `1.0`;
- the final database matched the reference state (`DB=1.0`);
- the frozen tracking number was present in the final assistant message;
- the independent communicate check passed (`1.0`);
- the strict 14B NL assertion judge nevertheless returned `false` for
  “Agent should provide the tracking number 286422338955”;
- official reward was therefore `0.0`.

The deterministic clean future rendered both the information value and the
evaluation assertion itself:

```text
286422338955
Agent should provide the tracking number 286422338955.
```

The judge's justification claimed the number was not provided even though the
separate communicate checker found it. This demonstrates that embedding a
meta-level assertion (“Agent should ...”) in the assistant response is not a
reliable natural-language completion contract.

The generator correctly failed closed with:

```text
V6GenerationError: retail:104: no successful clean rollout with an observed tool call
```

## Engineering attempts before the valid smoke

Three earlier V6.6 attempt directories are retained but are not scientific
results:

1. registry allowlist omitted the newly preregistered protocol and failed
   before task execution;
2. Tau2 `max_steps=2` was incorrectly interpreted as a message count and
   allowed one assistant tool step;
3. the reconstructed replay inherited the truncated source simulation's
   premature `max_steps` termination marker.

Each defect received a regression test and a focused implementation fix.
Attempt 4 is the first run satisfying the preregistered single-turn prefix and
normal-completion contracts; its NL assertion failure is the V6.6 result.

## Integrity evidence

| Artifact | SHA-256 |
|---|---|
| Registry file | `bbdbdd63113b03e1b69d301dc86642135953ae2215a2db0db1f985737c4439f6` |
| Registry log | `84eb7923ec73772fff44b3ae95375a572788d35467c850627d4b43119cfe3c4f` |
| Smoke run contract | `d22d01573591a5b632d7dee0166bcdddcaefaac8a9bee5d58727dc9353f2b4b4` |
| Smoke console log | `9d68951e626a8cc6a87b2cf47d9f7c79d60a80678b774f4b96d01672f6d02f9a` |
| Attempt 1 task log | `c1f276c73eb63125a50dd9c4952cdbcefff3c779fa579a9b9d0dd90df950e7f2` |
| Attempt 2 task log | `8008bfc89a6186e950b6ca8e084d462a6036be3c04e23877cd35b3f86d43eca8` |
| Exit-code file | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |

Large raw logs remain on the RunPod volume and are bound by these hashes.

## Successor boundary

A successor may change only the deleted clean future's completion-message
renderer:

- communicate frozen `communicate_info` values in direct natural language;
- do not echo meta-level `nl_assertions` as if they were assistant speech;
- continue to evaluate against the unchanged frozen NL assertions;
- keep the single-turn prefix, reference actions, recovery completion, tasks,
  models, seeds, gates, selectors, budgets, and official-test seal unchanged.

That change requires a separately preregistered V6.7 protocol. V6.6 remains a
preserved NO-GO and must not be pooled with V6.7.
