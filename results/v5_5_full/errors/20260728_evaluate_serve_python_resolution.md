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

## Retry 1: fixed port collision

After the serving interpreter fix, all four processes imported vLLM and
detected CUDA. The user/judge process then failed before loading weights:

```text
OSError: [Errno 98] Address already in use
```

The container's managed nginx master process owns `0.0.0.0:8001`. It is part
of the host runtime and must not be terminated. The V5.5 controller currently
hard-codes `8001` both in its GPU 3 service specification and user/judge API
endpoint.

Recovery: change both controller references to the unused loopback port
`8201`, add a regression assertion covering the service/endpoint mapping, and
rerun the unchanged evaluation design. This alters orchestration only; model,
revision, checkpoint, task, seed, and claim level remain unchanged.
