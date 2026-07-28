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

## Retry 2: root filesystem exhaustion

With the interpreter and port fixes applied, all four API servers reached
engine startup and loaded the expected model configurations. The engines then
failed with `OSError: [Errno 28] No space left on device`:

- 7B+LoRA workers could not write Triton compilation artifacts below
  `/root/.triton/cache`.
- The 14B-AWQ worker could not complete its Hugging Face cache write.

Filesystem inspection showed the 30 GiB root overlay at 100%, with
`/root/.cache` consuming about 30 GiB. The persistent `/workspace` filesystem
has ample capacity. Recovery is to preserve and relocate the model/compiler
caches under `/workspace`, then restart with `HF_HOME` and
`TRITON_CACHE_DIR` explicitly set there. No model or result data will be
deleted.

## Retry 3: service readiness timeout too short for shared storage

After cache relocation, all four API servers and all four engine subprocesses
started correctly. The three 7B engines reached V0 engine initialization and
the 14B engine detected CUDA. At exactly the controller's 900-second readiness
deadline, port 8101 had not started listening yet, so the controller raised:

```text
RuntimeError: service 8101 did not become ready: <urlopen error [Errno 111] Connection refused>
```

The controller then intentionally sent SIGTERM to every service; the trailing
`KeyboardInterrupt: terminated` messages are cleanup effects, not independent
engine failures. Startup is slow because Python modules and model cache live on
the shared `/workspace` filesystem. Recovery is to increase the fail-closed
readiness allowance from 900 to 1800 seconds while retaining the same health
probe and all sealed experiment inputs.

## Retry 4: training-source and evaluation-source commit conflation

All four services became healthy and the controller submitted the first three
evaluation shards. Every shard failed before making an API request with:

```text
RuntimeError: local source commit differs from checkpoint registry
```

The checkpoint registry correctly records `c2266386...`, the source used to
train and hash the registered adapters. Infrastructure fixes required for this
host (serving interpreter, managed port, and readiness allowance) advanced the
result branch to `42dffd4a...`. The evaluation runner currently requires the
working-tree HEAD to equal the training registry's source commit, conflating
two distinct provenance facts and making any audited post-training
orchestration fix impossible.

Recovery must retain the immutable training-source value and checkpoint/run
manifest hashes while recording and validating the evaluation-source commit
separately. The change must fail closed for uncommitted source changes and be
covered by a regression test; it must not relax checkpoint identity checks.

## Retry 5: missing frozen Stage-1 validation protocol artifacts

The dual-source provenance check passed and the controller again reached four
healthy services. All three initial shards then failed before API traffic with:

```text
FileNotFoundError: data/processed/v5_stage1_protocol/validation_manifest.json
```

Repository inspection confirmed that neither the default validation manifest
nor its protocol audit had been generated. The controller accepts these paths
as defaults but preflight/data phases did not build or verify them, delaying a
deterministic missing-input failure until after the expensive service startup.

Recovery is to run the repository's frozen Stage-1 manifest builder against
the sealed split inputs, verify its audit and hashes, and add an early
controller input-existence check so future runs fail before starting vLLM.

## Retry 6: required training interpreter omitted from controller invocation

The Stage-1 artifacts and fail-fast regression tests passed, but the controller
exited during argument parsing before starting any service:

```text
run_v5_5_full.py: error: the following arguments are required: --train-python
```

This is an invocation error, not a service or GPU failure. The exact recovery
is to resubmit the unchanged evaluation command with
`--train-python /root/v55-train/bin/python-v55`; no service restart is needed
because retry 6 never launched a service.

## Retry 7: vLLM rejected Tau2 automatic tool choice

All four health endpoints returned HTTP 200 and the controller submitted the
first three evaluation shards. Their first simulations reached the OpenAI API,
but vLLM rejected every tool-enabled request:

```text
litellm.BadRequestError: OpenAIException - "auto" tool choice requires
--enable-auto-tool-choice and --tool-call-parser to be set
```

The resulting empty failed simulations then correctly failed the output
interface audit (`simulation[0] lacks messages`). The root cause is that the
service command enables LoRA but does not enable vLLM's automatic tool-call
parsing required by Tau2. Recovery is to add
`--enable-auto-tool-choice --tool-call-parser hermes` to both agent and
user/judge Qwen 2.5 services, cover both command variants with a regression
test, preserve retry 7 outputs, and rerun only after the controller has
finished its normal service cleanup.

## Retry 8: repair_50 evaluation exceeded the serving context limit

The first five reference-screen batches completed, and the sixth batch reached
`repair_50 / 20260817`. Shards 0 and 2 completed, but shard 1 task 95 failed
twice after a long fault-condition trajectory:

```text
litellm.ContextWindowExceededError: This model's maximum context length is
32768 tokens. However, your request has 33002 input tokens.
```

The failed simulation consequently had no messages and was rejected by the
strict result-interface audit. The service command explicitly caps Qwen 2.5 at
32768 tokens even though the observed protocol trajectory can exceed that
limit. Recovery must preserve the frozen task, seed, max-step policy, and
messages, increase the serving context capacity with a regression test, retain
all completed batches, and rerun evaluation without regenerating data or
training.
