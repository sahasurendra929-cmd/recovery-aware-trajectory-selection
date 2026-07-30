# V6.10-CX1 context-remediation preregistration

Status: frozen before the first V6.10-CX1 model request.

This is a post-NO_GO engineering-remediation experiment. It does not replace
or reinterpret the original V6.10 compatibility result.

## Triggering evidence

V6.10 compatibility attempt
`v6_10-compat-a03-20260730T135100Z` produced three atomically published
task-local rejections:

- `airline:12`: `CONTEXT_WINDOW_EXCEEDED`;
- `airline:14`: `CONTEXT_WINDOW_EXCEEDED`;
- `airline:21`: `CONTEXT_WINDOW_EXCEEDED`.

The deterministic futility rule stopped the attempt after the third
rejection because the frozen minimum of 22 accepted tasks out of 24 was then
mathematically unreachable. The three failing requests contained 7,889,
7,681, and 8,010 input tokens while reserving the frozen 512-token output
cap against an 8,192-token serving limit.

## Single remediation delta

V6.10-CX1 changes only the teacher and shared user/judge vLLM
`--max-model-len` value from 8,192 to the models' native 32,768 tokens.

The 72B model declares `max_position_embeddings=32768`. On the exact
3x RTX PRO 4500 runtime used by the failed attempt, vLLM reported a 46,784
token teacher KV cache and a 95,184 token shared-endpoint KV cache. Therefore
one 32,768-token request fits the observed cache on both endpoints without
changing model weights, quantization, tensor parallelism, or GPU memory
utilization.

## Frozen invariants

All other compatibility conditions remain byte-for-byte or semantically
identical:

- exact 24-task compatibility population and 300-pair executable registry;
- Qwen2.5 72B teacher and 14B shared user/judge AWQ revisions;
- teacher TP=2 on GPU 0/1 and shared endpoint TP=1 on GPU 2;
- `float16`, `safetensors`, GPU memory utilization 0.9;
- automatic tool choice with the Hermes parser;
- seed 20260806, temperature 0, max output 512, max steps 60;
- deterministic matched-positive completion, fresh-teacher causal cells,
  and unforced teacher first-action measurement;
- minimum 22 tasks with three accepted pairs and 66 accepted pairs;
- official test remains sealed.

No history truncation, summarization, prompt deletion, adaptive output cap,
task substitution, or post-hoc seed change is permitted.

## Decision rule

Run a fresh compatibility attempt bound to a new source commit, fresh static
preflight and registry, fresh live runtime receipts, and a fresh release
manifest.

- GO: the existing V6.10 compatibility audit authorizes release under its
  unchanged 22-task/66-pair thresholds.
- NO_GO: any existing fail-closed gate fails.
- Futility stop: after three atomically published rejected tasks, stop because
  the 22-of-24 threshold is unreachable.

Pilot and formal generation remain forbidden until the compatibility token
measurement and independent audit return the existing authorization state.
