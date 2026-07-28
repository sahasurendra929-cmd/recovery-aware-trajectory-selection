# V5.5 reference-screen evaluation startup failure

- Time: 2026-07-28 07:09 UTC
- Phase: `evaluate`
- Source commit: `c2266386f7f580a07f9f5f9aeffea3e38ea88347`
- Status: fail-closed before any evaluation task ran
- Impact: none on the completed training checkpoints or checkpoint registry; all GPUs remained idle

## Symptom

All four vLLM service processes exited immediately. Each service log reported:

```text
/usr/bin/python3.12: Error while finding module specification for
'vllm.entrypoints.openai.api_server' (ModuleNotFoundError: No module named 'vllm')
```

## Root cause

The supplied serving interpreter,
`/workspace/venvs/v5_4_4500/bin/python`, is a symbolic link. The controller
normalizes executable paths, resolving it to `/usr/bin/python3.12`. That system
interpreter does not contain the vLLM environment.

## Recovery plan

1. Preserve this report and the four service logs on the GitHub result branch.
2. Create a regular executable copy inside the serving virtual environment and
   verify that it imports `vllm` and reports the expected environment prefix.
3. Restart the same sealed reference-screen evaluation with the regular
   executable path. No data, checkpoint, registry, seed, or evaluation task
   selection will change.

