# V5.3 single-host 4×RTX 4090 handoff

## 1. Decision and claim boundary

V5.3 first runs a **train-only feasibility pilot**. It must not train an SFT
arm or read the official test split. The 24 pilot tasks are fixed before V5.3
outcomes, and their trajectories are permanently excluded from formal data.

The complete pilot decision rule is:

| Quantity | GO threshold | Implied projection to 70 formal arm-train tasks |
|---|---:|---:|
| Tasks with at least one pair | at least 15/24 | 43.75 tasks |
| Tasks with at least two pairs | at least 4/24 | 55.4167 capped pairs total |

Both integer thresholds must pass. The displayed rates (`0.625` and `1/6`)
and projections are interpretations, not additional movable gates.

`GO_FORMAL_GENERATION` only says that formal generation is worth its cost. It
is not a positive scientific result. `NO_GO_STOP` stops before formal
generation and before every SFT arm.

The later formal gate is unchanged:

```text
at least 40 distinct arm-train task IDs with a pair
AND at least 48 constructible pairs
AND at most 2 pairs contributed by one task
```

## 2. Frozen data design

The historical inner-train split contains 83 tasks. Five tasks have no
registered assistant tool action, so a homogeneous ground-truth teacher
cannot execute them:

```text
retail:24
airline:0, airline:10, airline:28, airline:34
```

They are excluded structurally before any V5.3 rollout, reward, validation,
or test outcome. V5.3 does not mix a standard teacher into this population.
The remaining 78 tasks are hash-partitioned before generation:

```text
70 arm-train tasks = 52 retail + 18 airline
8 loss-validation tasks = 6 retail + 2 airline
partition SHA-256 =
c90255a2cf4100d8cf5b5fcf3e684144ed1a393fad1c214058146721c4ed6818
```

Loss-validation IDs:

```text
retail: 22, 23, 52, 63, 96, 103
airline: 15, 47
```

The 24 pilot IDs are sampled only from the 70 arm-train tasks and are
stratified by domain and fault family:

```text
retail (18):
2, 8, 10, 15, 19, 25, 30, 35, 54,
67, 69, 72, 85, 92, 93, 104, 106, 110

airline (6):
1, 11, 14, 33, 38, 40
```

Each task receives 12 clean and 12 controlled-error attempts. Pilot and formal
raw roots, base seeds, trial seeds, and contracts are separate.

Pilot:

```text
base seed: 20260730
733980, 626956, 183324, 266579, 559726, 176907,
35514, 769350, 589887, 418549, 589641, 621567
```

Formal:

```text
base seed: 20260722
574293, 816256, 420309, 70872, 419659, 596026,
55413, 256204, 120794, 83442, 692054, 873496
```

No extra attempt, replacement attempt, favorable-seed substitution, hidden
rescue, repaired failure, or relabelled failure is allowed.

## 3. What V5.3 fixes

V5.2 failed because the registered pool was unattainable under its teacher,
user simulator, and judge interface. V5.3 freezes these corrections:

- exact 32,768-token serving context;
- stronger 32B-AWQ trajectory teacher;
- stronger 14B-AWQ user simulator and strict NL judge;
- bounded nonzero trajectory sampling (`temperature=0.2`, `top_p=0.95`);
- 12 fixed attempts per task and condition;
- strict judge accepts one bare JSON object or one complete JSON fence,
  requires nonempty exact assertion coverage and real JSON booleans, retries
  malformed content once, then fails closed;
- multiple tool calls are serialized by executing the first ordered call and
  replanning; mixed text is removed while its SHA-256 is retained;
- user-stop/DB=0, repeated-user, and max-step patterns are reported only as
  diagnostics and never repaired.

## 4. Frozen models and GPU topology

During pilot and formal trajectory generation, use one host with four RTX
4090 24GB GPUs:

| GPU | Role | Frozen model | Port |
|---:|---|---|---:|
| 0 | user simulator + strict judge | `Qwen/Qwen2.5-14B-Instruct-AWQ` | 8001 |
| 1–2 | shared TP=2 trajectory teacher | `Qwen/Qwen2.5-32B-Instruct-AWQ` | 8011 |
| 3 | pilot preflight/monitoring reserve | none | — |

Frozen revisions:

```text
32B: 5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c
14B: 539535859b135b0244c91f3e59816150c8056698
```

All three pilot clients use the same two endpoints, each with concurrency one.
Thus aggregate client concurrency is three. GPU 3 must not introduce a
scientifically different fallback model.

The later one-seed validation screen is a separate phase. Each evaluation GPU
serves the frozen 7B student base plus all four LoRA adapters. The unadapted
`openai/v5-3-base` alias is used as the user simulator and strict judge for
every arm, while the arm-specific alias is used only as the agent. This keeps
the evaluator identical across arms, but it is only a screening evaluator:
before an official-test claim, the selected method must be preregistered and
confirmed with a stronger fixed user simulator/judge. The 14B generation
service is not silently reused during this validation screen.

## 5. Immutable checkout and reusable environment

Publish and record one reviewed 40-character commit:

```bash
export V5_3_COMMIT='<published-40-character-commit>'
export REPO=/workspace/repos/recovery-aware-trajectory-selection
export TAU2="$REPO/data/raw/tau2-bench"
export SERVE_VENV=/workspace/venvs/v5-2-serve
export TRAIN_VENV=/workspace/venvs/v5-2-train
export HF_HOME=/workspace/cache/huggingface
```

Use a detached, clean checkout:

```bash
mkdir -p /workspace/repos /workspace/venvs /workspace/cache/huggingface
git clone \
  https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git \
  "$REPO"
cd "$REPO"
git fetch origin
git checkout --detach "$V5_3_COMMIT"
git diff --quiet --
git diff --cached --quiet --

mkdir -p data/raw
git clone https://github.com/sierra-research/tau2-bench.git "$TAU2"
git -C "$TAU2" checkout --detach \
  fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C "$TAU2" diff --quiet --
git -C "$TAU2" diff --cached --quiet --
```

Reuse the persistent V5.2 serving and training environments when their exact
package checks pass. This avoids downloading CUDA and model dependencies again.
Create only a missing environment:

```bash
test -x "$SERVE_VENV/bin/python" || python3 -m venv "$SERVE_VENV"
"$SERVE_VENV/bin/python" -m pip install --upgrade pip
"$SERVE_VENV/bin/python" -m pip install -r requirements-stage0-v5.txt
"$SERVE_VENV/bin/python" -m pip install -e "$TAU2"
"$SERVE_VENV/bin/python" -m pip check

test -x "$TRAIN_VENV/bin/python" || python3 -m venv "$TRAIN_VENV"
"$TRAIN_VENV/bin/python" -m pip install --upgrade pip
"$TRAIN_VENV/bin/python" -m pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$TRAIN_VENV/bin/python" -m pip install -r requirements-gpu-v5-sft.txt
"$TRAIN_VENV/bin/python" -m pip install -e "$TAU2"
"$TRAIN_VENV/bin/python" -m pip check
```

Run the targeted CPU tests before loading models:

```bash
"$TRAIN_VENV/bin/python" -m unittest \
  tests.test_v5_3_judge_protocol \
  tests.test_v5_3_train_only_pilot \
  tests.test_v5_strict_nl_judge \
  tests.test_v5_3_manifests \
  tests.test_v5_3_sft_causal_data
```

The controller validates the exact environment versions, commits, four-GPU
inventory, free disk, ports, and protocol files before it may start the pilot:

```bash
"$TRAIN_VENV/bin/python" scripts/run_v5_3_single_host.py --stage preflight
"$TRAIN_VENV/bin/python" scripts/run_v5_3_single_host.py --stage self-test
"$TRAIN_VENV/bin/python" scripts/run_v5_3_single_host.py --stage prefetch
"$TRAIN_VENV/bin/python" scripts/run_v5_3_single_host.py --stage protocol
"$TRAIN_VENV/bin/python" scripts/run_v5_3_single_host.py --stage pilot
```

The final command exits `0` for `GO_FORMAL_GENERATION`, `20` for a valid
`NO_GO_STOP`, and nonzero for a protocol/runtime failure. Do not invoke
`--stage generate` unless the immutable pilot decision is GO.

**Use this controller path as the only execution path.** Sections 6 onward
document the underlying commands and audit contract so an operator can inspect
what the controller did; do not start a second set of vLLM services, clients,
or an additional audit after the controller has run.

## 6. Build and audit the V5.3 protocol

This step creates a 78-task generation manifest and independent dynamic audit:

```bash
python scripts/prepare_v5_3_manifests.py \
  --tau2-root "$TAU2" \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --output-dir data/processed/v5_3_protocol \
  --seed 20260722

python scripts/verify_v5_stage0_injections.py \
  --tau2-root "$TAU2" \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest data/processed/v5_3_protocol/generation_manifest.json \
  --output \
    data/processed/v5_3_protocol/generation_dynamic_audit.json

python scripts/verify_v5_stage0_injections.py \
  --tau2-root "$TAU2" \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest data/processed/v5_3_protocol/validation_manifest.json \
  --output \
    data/processed/v5_3_protocol/validation_dynamic_audit.json
```

Then build the outcome-free pilot:

```bash
python scripts/run_v5_3_train_only_pilot.py manifest \
  --tau2-root "$TAU2" \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --generation-manifest \
    data/processed/v5_3_protocol/generation_manifest.json \
  --output-dir artifacts/v5_3_train_only_pilot/protocol
```

Outputs:

```text
artifacts/v5_3_train_only_pilot/protocol/effective_task_universe.json
artifacts/v5_3_train_only_pilot/protocol/pilot_manifest.json
```

They must report 70 arm-train tasks, 24 pilot tasks, all 24 using the
ground-truth route, and `official_test_used=false`.

## 7. Start the frozen endpoints

User/judge:

```bash
CUDA_VISIBLE_DEVICES=0 "$SERVE_VENV/bin/vllm" serve \
  Qwen/Qwen2.5-14B-Instruct-AWQ \
  --revision 539535859b135b0244c91f3e59816150c8056698 \
  --served-model-name Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 127.0.0.1 --port 8001 \
  --tensor-parallel-size 1 --quantization awq --dtype float16 \
  --max-model-len 32768 --max-num-seqs 3 \
  --gpu-memory-utilization 0.90 --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Teacher:

```bash
CUDA_VISIBLE_DEVICES=1,2 "$SERVE_VENV/bin/vllm" serve \
  Qwen/Qwen2.5-32B-Instruct-AWQ \
  --revision 5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c \
  --served-model-name Qwen/Qwen2.5-32B-Instruct-AWQ \
  --host 127.0.0.1 --port 8011 \
  --tensor-parallel-size 2 --quantization awq --dtype float16 \
  --max-model-len 32768 --max-num-seqs 3 \
  --gpu-memory-utilization 0.90 --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser hermes
```

Run them in persistent `tmux` sessions and save the exact command and full
log. Before generation, require:

1. exact model IDs from both `/v1/models` endpoints;
2. one near-longest representative request per endpoint;
3. three concurrent representative teacher requests;
4. three concurrent representative user requests;
5. no OOM, restart, NaN, or tool-parser error.

Failure means stop; it does not authorize a model, context, concurrency,
quantization, or threshold change under the same protocol name.

## 8. Generate the three pilot shards

The pilot raw root must be new and must never equal the formal raw root:

```bash
export PILOT_ROOT="$REPO/artifacts/v5_3_train_only_pilot"
mkdir -p "$PILOT_ROOT/raw" "$PILOT_ROOT/logs"
```

Run one command for each `SHARD=0`, `1`, and `2` concurrently:

```bash
SHARD=0
python scripts/run_v5_3_train_only_pilot.py generate \
  --tau2-root "$TAU2" \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --generation-manifest \
    data/processed/v5_3_protocol/generation_manifest.json \
  --dynamic-audit \
    data/processed/v5_3_protocol/generation_dynamic_audit.json \
  --pilot-manifest "$PILOT_ROOT/protocol/pilot_manifest.json" \
  --output-dir "$PILOT_ROOT/raw" \
  --shard-index "$SHARD" \
  --teacher-api-base http://127.0.0.1:8011/v1 \
  --user-api-base http://127.0.0.1:8001/v1 \
  --teacher-revision \
    5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c \
  --user-judge-revision \
    539535859b135b0244c91f3e59816150c8056698 \
  --expected-source-commit "$V5_3_COMMIT"
```

Expected terminal artifacts:

```text
3 COMPLETE run_contract.shard-*.json files
576 simulations across the clean/error result files
strict_nl_judge_audit_*.json files under raw/logs/
```

An incomplete contract, nonzero process exit, seed drift, schema-retry
exhaustion, or missing result is fail-closed. Do not report a partial metric.

## 9. Apply the one-shot pilot gate

After all three contracts are `COMPLETE`:

```bash
mapfile -t PILOT_RESULTS < <(
  find "$PILOT_ROOT/raw" -maxdepth 1 -type f \
    \( -name 'retail_clean.shard-*.json' \
       -o -name 'retail_error.shard-*.json' \
       -o -name 'airline_clean.shard-*.json' \
       -o -name 'airline_error.shard-*.json' \) | sort
)

mapfile -t JUDGE_AUDITS < <(
  find "$PILOT_ROOT/raw/logs" -type f \
    -name 'strict_nl_judge_audit_*.json' | sort
)

python scripts/run_v5_3_train_only_pilot.py audit \
  --pilot-manifest "$PILOT_ROOT/protocol/pilot_manifest.json" \
  --judge-audits "${JUDGE_AUDITS[@]}" \
  --output "$PILOT_ROOT/pilot_go_no_go.json" \
  "${PILOT_RESULTS[@]}"
```

Exit/status contract:

```text
0  = GO_FORMAL_GENERATION
20 = NO_GO_STOP
2  = malformed/prohibited/incomplete data; fail closed
```

The report contains first-pair and second-pair counts/rates, 70-task
projections, strict-judge rejection/retry counts, user-loop/DB=0 diagnostics,
the unchanged 40/48 formal gate, and the claim boundary.

## 10. What happens only after GO

The formal controller must:

1. use `data/processed/v5_3_protocol/generation_manifest.json` (78 tasks);
2. write to a fresh formal raw root, never the pilot raw root;
3. use formal base seed `20260722`, never the pilot seed;
4. generate all 78 tasks, then let the frozen 70/8 partition separate
   arm-train from loss-validation;
5. require the unchanged 40-task/48-pair gate before training;
6. train matched `perfect_success`, `failure_raw`, `repair_50`, and
   `repair_100` arms only after that gate;
7. keep the official test sealed until method selection is frozen.

The pilot report is an operational feasibility artifact. Only later matched
end-to-end task-success comparisons can support the research claim.
`failure_raw` is a descriptive negative-exposure control in V5.3, not an
isolated causal label-masking contrast with `repair_100`: task support and
budgets are matched, but exact pair-occurrence schedules are not coupled.
