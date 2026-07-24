# V5 Stage 0: end-to-end pipeline validation

Stage 0 replaces the V1–V4 offline next-tool-call boundary with a real
multi-turn τ²-bench run. It does not train a model and it does not support a
paper claim.

## What Stage 0 must establish

1. The frozen τ²-bench v1.0.1 retail and airline environments run end to end.
2. The official 60-task test split remains sealed.
3. Ten development tasks run under both clean and deterministic error-injected
   conditions, producing 20 complete trajectories.
4. Every injected call is a read-only nonexistent-identifier lookup, produces
   a real tool error, and leaves the database unchanged.
5. The official composite task reward is recorded alongside recovery
   diagnostics and runtime cost.

V1–V4 answer, “Given this history, what is the next tool call?” Stage 0
answers, “Can the agent finish the whole task, and can the same agent still
finish after a controlled tool failure?”

## Frozen data boundary

- benchmark: `sierra-research/tau2-bench`
- release: `v1.0.1`
- commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- official train: 74 retail + 30 airline = 104 tasks
- derived development validation: 15 retail + 6 airline = 21 tasks
- remaining inner train: 59 retail + 24 airline = 83 tasks
- sealed official test: 40 retail + 20 airline = 60 tasks
- deterministic seed: `20260722`

Only official-train tasks may be used for Stage 0. The official test IDs and
source-file hashes may be audited, but test task content, trajectories, and
outcomes must not influence development.

## Paired smoke design

Ten validation tasks are selected deterministically: seven retail and three
airline tasks whose reference plan contains a safe identity/reservation lookup.
Each task is run twice:

- `clean`: stock τ²-bench LLM agent;
- `error`: a wrapper injects one deterministic invalid read call on the first
  agent turn, then delegates every later decision to the same LLM agent.

The injected call must be one of:

- `find_user_id_by_email`;
- `find_user_id_by_name_zip`;
- `get_user_details`;
- `get_reservation_details`.

The clean and error conditions use the same tasks, model, user simulator,
decoding, seed, maximum steps, and evaluator.

## Stage-0 gates

Stage 0 passes only if:

- data audit is `PASS`;
- all four domain/condition result files exist;
- all 20 simulations are complete;
- all ten injected calls are present and match the manifest;
- all ten injected calls produce `ToolMessage.error=true`;
- infrastructure failure count is zero;
- clean and error task IDs are identical.

Task success is reported, but no significance or recovery-training claim is
made from ten development tasks. The same Qwen model is used as agent, local
user simulator, and qualitative judge only to validate the plumbing. Formal
evaluation must freeze a separately capable simulator and judge.

## Local deterministic preparation

```bash
python3 scripts/prepare_v5_stage0.py \
  --tau2-root data/raw/tau2-bench \
  --output-dir data/processed/v5_stage0

python3 -m unittest tests.test_v5_stage0 -v
```

## Persistent RunPod layout

Use the network volume for environments, model cache, code, and artifacts:

```text
/workspace/cache/huggingface
/workspace/cache/uv
/workspace/venvs/tau2-v1.0.1
/workspace/recovery-aware-trajectory-selection
/workspace/recovery-aware-trajectory-selection/results/v5_stage0
```

Never rebuild a working environment for a new arm. Create a new environment
only when the dependency lock changes.

## vLLM server

Qwen2.5 already contains a Hermes-compatible tool-use chat template. Start the
server with automatic tool selection and the Hermes parser:

```bash
export HF_HOME=/workspace/cache/huggingface
export UV_CACHE_DIR=/workspace/cache/uv

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.88 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm
```

Do not start the experiment until both `/health` and `/v1/models` respond.

## Run and summarize

```bash
python scripts/run_v5_stage0.py \
  --tau2-root data/raw/tau2-bench \
  --manifest data/processed/v5_stage0/smoke_manifest.json \
  --output-dir results/v5_stage0/raw \
  --condition both

python scripts/summarize_v5_stage0.py \
  --manifest data/processed/v5_stage0/smoke_manifest.json \
  --results-dir results/v5_stage0/raw \
  --output results/v5_stage0/stage0_summary.json
```

Stop immediately on a manifest mismatch, a missing injected error, an
infrastructure failure, or an incomplete result file. Do not substitute test
tasks, silently retry with different decoding, or report partial metrics as a
completed Stage 0.
