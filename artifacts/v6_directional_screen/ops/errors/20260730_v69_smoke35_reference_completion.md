# V6.9 retail:35 reference-completion smoke failure

Status: `NO_GO`

The first expected-reference-error repair correctly retained the deliberately
failed email lookup during clean reconstruction. The fresh smoke then reached
the matched-recovery reference-completion path and failed closed because that
second path independently retained the obsolete invariant that every remaining
reference action must return a non-error result.

Failure:

```text
V6GenerationError: reference completion action 35_0 returned a tool error
```

This is a follow-on implementation defect in the same error-replay invariant,
not a scientific protocol change. The next repair must use one tested helper
for clean replay, reference-tail replay, and reference-completion replay.
Frozen corrective actions remain required to succeed.

- Run-contract SHA-256:
  `265af55e4b523d8143ba739bc2ea720d6e164b627f5904a2ce87506b3c26da33`
- Log SHA-256:
  `4db7d8aa6b6e9698bcf6fbd8cac00bf0f3e5005519b1df86529a1120a6d29fa6`
- Candidate output emitted: no
- Official test used: no
