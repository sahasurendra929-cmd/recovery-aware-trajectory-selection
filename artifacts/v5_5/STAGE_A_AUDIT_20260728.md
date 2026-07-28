# V5.5 Stage A terminal audit

Stage A is accepted as `PASS_TRAINING_AUTHORIZED` and its existing 48
reference-grounded pairs are reused without regeneration.

The audit covers 48 unique pairs over 24 tasks in airline and retail. The
frozen tau2 commit is
`fc0055dc4e0a316c3f83133267fbd6faaa770992`; the split manifest hash is
`a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a`.
No official-test task was accessed.

The independent environment replay audit was rerun on 2026-07-28 against the
unchanged manifest and pairs. It produced a byte-identical audit:

```text
af7f09a4b9d7d1075c7f700507be58944f51a3177e812f0abc32292aeee075f0
```

All 48 injections produced the registered real error, every correction
succeeded, environment end states and stored hashes matched replay, pair IDs
were complete, and there were zero replay failures.

The legacy `sha256_manifest.txt` remains preserved. Its Stage A core hashes
match the current Git blobs. Six entries under later preflight/smoke outputs
changed after the legacy manifest was created because those checks were
rerun; these dynamic files do not define pair identity and are covered by the
later no-weight result-package manifests. The scoped terminal audit JSON
records this distinction instead of rewriting historical evidence.

This pass authorizes reuse of the registered reference pairs. It does not
override Stage B's subsequent `NO_GO_REFERENCE_SCREEN` scientific result.
