# V6.10 run handoff

Status: **compatibility-ready only after every closure gate in this document
passes**. This is not prospective-Pilot or official-evaluation ready: the V6
end-to-end evaluator and task-cluster statistical summarizer are still
required, tested, hashed, and frozen before prospective Pilot authorization.

Canonical protocol:

```text
v6_10_pipeline_closure_v1
```

Scientific contract:

```text
configs/v6_10_closure.yaml
V6_10_CLOSURE_PREREGISTRATION.md
```

This handoff deliberately has a hard release barrier. Do not adapt a V6.9
command by changing only its output directory. V6.10 requires three distinct
runtime pathways:

1. sanitized oracle construction for successful positive data;
2. an unforced teacher first-action measurement; and
3. a fresh-teacher, gold-free continuation for the four causal cells.

## 1. Definition of ready

The release is not runnable until all of the following are true:

- the registry exposes three distinct phases:
  `compatibility`, `pilot`, and `formal`;
- the exact compatibility, prospective-Pilot, and formal task-list hashes
  equal the values in the V6.10 config;
- a static reference-preflight command emits one authoritative receipt with a
  PASS/REJECTED row and an embedded frozen sanitized plan for every
  formal-universe task;
- sanitized plans contain zero tool errors and independently reproduce
  Tau2 ENV reward `1.0` and the successful raw-reference final state;
- static preflight records full official reward as deferred to the dynamic
  frozen judge and never invokes an LLM judge;
- candidate corrective actions are drawn only from successful retained
  sanitized-plan slots;
- matched positive construction uses the sanitized plan;
- causal cells use `fresh_teacher` and cannot call a reference-tail or
  reference-completion helper;
- teacher first-action measurement occurs before any forced action;
- task-local rejection-and-continue is implemented;
- source, container, model-server, task-map, and semantic-contract hashes are
  bound into every task receipt;
- V6.10 is frozen to one shard (`num_shards=1`, `shard_index=0`) and the
  generation closure verifies exact terminal task coverage;
- an end-to-end `retail:35` regression reaches data materialization;
- the complete test suite passes from a clean checkout; and
- the release manifest replaces stale manually maintained "active version"
  fields.

Passing unit tests for the reference-execution helper alone is not sufficient.

## 2. Frozen directory layout

Use an immutable source checkout and new artifact root:

```text
/workspace/repos/recovery-aware-trajectory-selection-v6_10/
/workspace/v6_10/
  releases/<source-commit>/
    runtime-sessions/<runtime-session-id>/
      service_specs.json
      runtime_receipts/
      release_manifest.json
  artifacts/
    static/<attempt-id>/
    compatibility/<attempt-id>/
    pilot/<attempt-id>/
    formal/<attempt-id>/
    measurement/<attempt-id>/
    selection/<attempt-id>/
    training/<attempt-id>/
    evaluation/<attempt-id>/
  evidence/
```

Never run into a V6.9 directory. Never place two attempt IDs in the same
output directory.

A runtime service session names one exact live set of teacher/user/judge
processes. Its ID has the form:

```text
v6_10-runtime-a{NN}-{YYYYMMDDTHHMMSSZ}
```

The ID is an operational namespace, not identity evidence. The actual
identity evidence is the PID/start, executable, argv, snapshot, socket, GPU,
probe, and receipt hashes. A model-service restart always requires a new
runtime session ID, new runtime receipts, a new release manifest, and a fresh
generation attempt directory. It does not require regenerating the immutable
reference preflight or executable registry when their bound source/config
inputs are unchanged.

Attempt IDs have the form:

```text
v6_10-{stage}-a{NN}-{YYYYMMDDTHHMMSSZ}
```

Examples:

```text
v6_10-static-a01-20260730T120000Z
v6_10-compat-a01-20260730T130000Z
v6_10-pilot-a01-20260731T000000Z
```

An attempt number is not a protocol version.

## 3. Canonical model-service topology

The model services must expose the frozen model revisions from the config.
For the observed single Pod with three NVIDIA RTX PRO 4500 Blackwell GPUs
(at least 32,000 MiB each), the frozen
topology is:

- two GPUs in one tensor-parallel service for the 72B AWQ teacher;
- one GPU for the shared 14B AWQ user simulator/judge service.

Two GPUs used for tensor parallelism must be visible to one model-service
process. Unrelated one-GPU Pods cannot silently be treated as one
tensor-parallel 72B server. This V6.10 release accepts only the exact
three-GPU topology above: teacher TP=2 on two GPUs and one shared user/judge
endpoint on the third GPU. A different topology is a source/protocol change
and must be implemented, tested, and frozen before compatibility; it cannot be
substituted at launch. The two-GPU 72B service must pass a real long-context
healthcheck before generation is authorized; available aggregate VRAM alone
is not sufficient evidence. Stop the generation services before using the
same GPUs for later 7B training.

For every endpoint, create a model-server receipt containing:

- model repository and resolved immutable revision;
- local snapshot/config/tokenizer hashes;
- quantization and dtype;
- tensor-parallel size;
- exact safe launch command and live server PID;
- `/proc` boot/start identity, exact observed command, and ownership of the
  loopback endpoint's listening socket;
- exact `CUDA_VISIBLE_DEVICES` mapping to the declared GPU UUIDs;
- byte hashes of config/tokenizer/index files and content-addressed blob IDs
  for every safetensors shard in the frozen local HF snapshot;
- container-image digest;
- vLLM, Torch, CUDA, and driver versions;
- GPU model, UUID, and memory;
- healthcheck result; and
- UTC launch time.

The endpoint URL is transport metadata. It is not proof that the correct model
is being served.

The supplied container-image digest is a pinned operator declaration that is
hash-bound consistently across the release; the current builder does not
claim to observe that digest from inside the container. Python, Torch, CUDA,
driver, vLLM, GPU, process, socket, and snapshot evidence are observed live.

Do not hand-write the strict runtime receipts. Download each revision into the
local Hugging Face cache first and retain the absolute snapshot path returned
by `snapshot_download(..., local_files_only=True)`. Launch vLLM from that
absolute path, in the same container/PID namespace as the receipt builder,
with an absolute Python executable in isolated mode (`python -I -m ...`) and
this exact flag sequence:
`--model`, `--revision`, `--served-model-name`, `--quantization awq`,
`--dtype float16`, explicit `--tensor-parallel-size`, `--load-format
safetensors`, `--max-model-len 8192`, `--gpu-memory-utilization 0.9`,
`--host 127.0.0.1`, and `--port`. Set
`CUDA_VISIBLE_DEVICES` explicitly and capture the actual Python server PID
from `$!`; wrapper-shell or unrelated PIDs are invalid.

The `tokenizer_or_config_sha256` value is the canonical aggregate emitted by
`snapshot_identity`: byte hashes of the small identity files plus HF blob IDs
for all safetensors symlinks. It is not an operator-chosen placeholder. After
initializing the release variables in Section 4, create `service_specs.json`
with the observed PID, GPU UUIDs, exact argv, and aggregate snapshot hash.
The receipt builder derives the process start time from `/proc`; it is not an
operator-authored timestamp:

```bash
# Run Section 4 first so REPO, TAU2, ROOT, SOURCE_COMMIT, and the immutable
# source hashes are defined. Replace the timestamp below before each distinct
# teacher/user/judge process-set launch.
RUNTIME_SESSION_ID=v6_10-runtime-a01-YYYYMMDDTHHMMSSZ
RUNTIME_SESSION_DIR="$ROOT/releases/$SOURCE_COMMIT/runtime-sessions/$RUNTIME_SESSION_ID"
test ! -e "$RUNTIME_SESSION_DIR"
mkdir -p "$RUNTIME_SESSION_DIR/model-logs"

cd "$REPO"
RUNTIME_PYTHON="$(python -c 'import sys; print(sys.executable)')"
HF_CACHE="$ROOT/hf-cache"
mkdir -p "$HF_CACHE"

TEACHER_SNAPSHOT="$("$RUNTIME_PYTHON" -I -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="Qwen/Qwen2.5-72B-Instruct-AWQ", revision="698703eae6604af048a3d2f509995dc302088217", cache_dir="'"$HF_CACHE"'"))')"
USER_JUDGE_SNAPSHOT="$("$RUNTIME_PYTHON" -I -c \
  'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ", revision="539535859b135b0244c91f3e59816150c8056698", cache_dir="'"$HF_CACHE"'"))')"

CUDA_VISIBLE_DEVICES=0,1 nohup "$RUNTIME_PYTHON" -I -m \
  vllm.entrypoints.openai.api_server \
  --model "$TEACHER_SNAPSHOT" \
  --revision 698703eae6604af048a3d2f509995dc302088217 \
  --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
  --quantization awq --dtype float16 --tensor-parallel-size 2 \
  --load-format safetensors --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8101 \
  >"$RUNTIME_SESSION_DIR/model-logs/teacher.log" 2>&1 &
TEACHER_PID=$!

CUDA_VISIBLE_DEVICES=2 nohup "$RUNTIME_PYTHON" -I -m \
  vllm.entrypoints.openai.api_server \
  --model "$USER_JUDGE_SNAPSHOT" \
  --revision 539535859b135b0244c91f3e59816150c8056698 \
  --served-model-name Qwen/Qwen2.5-14B-Instruct-AWQ \
  --quantization awq --dtype float16 --tensor-parallel-size 1 \
  --load-format safetensors --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8201 \
  >"$RUNTIME_SESSION_DIR/model-logs/user-judge.log" 2>&1 &
USER_JUDGE_PID=$!

for endpoint in \
  http://127.0.0.1:8101/v1/models \
  http://127.0.0.1:8201/v1/models
do
  ready=false
  for _ in $(seq 1 180); do
    if curl --fail --silent "$endpoint" >/dev/null; then
      ready=true
      break
    fi
    sleep 5
  done
  test "$ready" = true
done
kill -0 "$TEACHER_PID"
kill -0 "$USER_JUDGE_PID"
```

If either 8192-token service fails to become healthy on the exact 3×RTX PRO
4500 Blackwell topology, stop both PIDs and report a typed preflight failure. Do not silently
reduce context length, change memory utilization, or begin generation.

```bash
python -c 'from pathlib import Path; from scripts.build_v6_10_runtime_receipts import snapshot_identity; print(snapshot_identity(Path("'"$TEACHER_SNAPSHOT"'"), expected_model="Qwen/Qwen2.5-72B-Instruct-AWQ", expected_revision="698703eae6604af048a3d2f509995dc302088217")[1])'
python -c 'from pathlib import Path; from scripts.build_v6_10_runtime_receipts import snapshot_identity; print(snapshot_identity(Path("'"$USER_JUDGE_SNAPSHOT"'"), expected_model="Qwen/Qwen2.5-14B-Instruct-AWQ", expected_revision="539535859b135b0244c91f3e59816150c8056698")[1])'
```

```json
{
  "roles": {
    "teacher": {
      "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
      "resolved_revision": "698703eae6604af048a3d2f509995dc302088217",
      "api_base": "http://127.0.0.1:8101/v1",
      "quantization": "awq",
      "dtype": "float16",
      "tensor_parallel_size": 2,
      "gpu_uuids": ["GPU-TEACHER-0", "GPU-TEACHER-1"],
      "max_model_len": 8192,
      "gpu_memory_utilization": 0.9,
      "launch_command": ["<ABSOLUTE_RUNTIME_PYTHON>", "-I", "-m", "vllm.entrypoints.openai.api_server", "--model", "<ABSOLUTE_72B_SNAPSHOT>", "--revision", "698703eae6604af048a3d2f509995dc302088217", "--served-model-name", "Qwen/Qwen2.5-72B-Instruct-AWQ", "--quantization", "awq", "--dtype", "float16", "--tensor-parallel-size", "2", "--load-format", "safetensors", "--max-model-len", "8192", "--gpu-memory-utilization", "0.9", "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--host", "127.0.0.1", "--port", "8101"],
      "server_pid": 12345,
      "tokenizer_or_config_sha256": "<64-lowercase-hex>"
    },
    "user": {
      "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
      "resolved_revision": "539535859b135b0244c91f3e59816150c8056698",
      "api_base": "http://127.0.0.1:8201/v1",
      "quantization": "awq",
      "dtype": "float16",
      "tensor_parallel_size": 1,
      "gpu_uuids": ["GPU-USER-JUDGE"],
      "max_model_len": 8192,
      "gpu_memory_utilization": 0.9,
      "launch_command": ["<ABSOLUTE_RUNTIME_PYTHON>", "-I", "-m", "vllm.entrypoints.openai.api_server", "--model", "<ABSOLUTE_14B_SNAPSHOT>", "--revision", "539535859b135b0244c91f3e59816150c8056698", "--served-model-name", "Qwen/Qwen2.5-14B-Instruct-AWQ", "--quantization", "awq", "--dtype", "float16", "--tensor-parallel-size", "1", "--load-format", "safetensors", "--max-model-len", "8192", "--gpu-memory-utilization", "0.9", "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--host", "127.0.0.1", "--port", "8201"],
      "server_pid": 23456,
      "tokenizer_or_config_sha256": "<64-lowercase-hex>"
    },
    "judge": {
      "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
      "resolved_revision": "539535859b135b0244c91f3e59816150c8056698",
      "api_base": "http://127.0.0.1:8201/v1",
      "quantization": "awq",
      "dtype": "float16",
      "tensor_parallel_size": 1,
      "gpu_uuids": ["GPU-USER-JUDGE"],
      "max_model_len": 8192,
      "gpu_memory_utilization": 0.9,
      "launch_command": ["<ABSOLUTE_RUNTIME_PYTHON>", "-I", "-m", "vllm.entrypoints.openai.api_server", "--model", "<ABSOLUTE_14B_SNAPSHOT>", "--revision", "539535859b135b0244c91f3e59816150c8056698", "--served-model-name", "Qwen/Qwen2.5-14B-Instruct-AWQ", "--quantization", "awq", "--dtype", "float16", "--tensor-parallel-size", "1", "--load-format", "safetensors", "--max-model-len", "8192", "--gpu-memory-utilization", "0.9", "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--host", "127.0.0.1", "--port", "8201"],
      "server_pid": 23456,
      "tokenizer_or_config_sha256": "<same-14B-hash>"
    }
  }
}
```

The `user` and `judge` entries must describe the same endpoint, PID, GPU,
launch command, and snapshot hash. The builder reads `/proc/<pid>` before and
after each long-context probe and rejects PID reuse, process restart, command
drift, snapshot drift, GPU drift, unsafe/dummy loading flags, or a PID that
does not own the endpoint socket. Then build and self-validate the receipts:

```bash
SERVICE_SPECS="$RUNTIME_SESSION_DIR/service_specs.json"
DEPENDENCY_SPEC="$REPO/requirements-stage0-v5.txt"
RUNTIME_RECEIPTS="$RUNTIME_SESSION_DIR/runtime_receipts"

python scripts/build_v6_10_runtime_receipts.py \
  --source-root "$REPO" \
  --tau2-root "$TAU2" \
  --dependency-lock "$DEPENDENCY_SPEC" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
  --service-specs "$SERVICE_SPECS" \
  --output-dir "$RUNTIME_RECEIPTS" \
  --runtime-python "$(command -v python)" \
  --request-timeout 900
```

This command performs an exact `/models` check and a real generation probe for
each distinct endpoint. The probe must report at least 4096 input tokens,
`max_new_tokens=128`, seed `20260806`, temperature `0`, a nonempty response,
positive latency, and positive peak GPU-memory evidence. It atomically emits:

The CLI flag retains its historical name `--dependency-lock`, but the supplied
file is only a requirements specification. Do not claim a fully reproducible
Python environment until a resolved lock/wheel manifest or archived
installed-package freeze is added for the formal paper run.

```text
source_container_provenance.json
model_server_receipts.json
runtime_receipt_hashes.json
```

## 4. Release preflight

Start from a clean checkout of the intended release commit:

```bash
set -euo pipefail

REPO=/workspace/repos/recovery-aware-trajectory-selection-v6_10
TAU2=/workspace/repos/tau2
ROOT=/workspace/v6_10
CONFIG="$REPO/configs/v6_10_closure.yaml"
SPLIT="$REPO/artifacts/v5_stage0/manifests/split_manifest.json"
PREREG="$REPO/V6_10_CLOSURE_PREREGISTRATION.md"
CONTAINER_IMAGE_DIGEST="sha256:<64-lowercase-hex>"

cd "$REPO"

SOURCE_COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git -C "$TAU2" rev-parse HEAD)" = \
  fc0055dc4e0a316c3f83133267fbd6faaa770992
test -z "$(git -C "$TAU2" status --porcelain=v1 --untracked-files=all)"

GENERATOR_SHA="$(
  sha256sum scripts/run_v6_candidate_generation.py | awk '{print $1}'
)"
CONFIG_SHA="$(sha256sum "$CONFIG" | awk '{print $1}')"
PREREG_SHA="$(
  sha256sum "$PREREG" | awk '{print $1}'
)"
```

Do not assemble the release manifest by hand. After the static preflight,
registry, and runtime receipts below exist, use the release-manifest builder
in Section 5.3. Candidate generation is forbidden until that manifest passes
its own validator and its exact file hash is supplied to the generator.

The release manifest binds these source/config values, every relevant script,
the exact static artifacts, the runtime receipts, and the immutable container
digest.

Run the complete V6/V6.10 code-only suite in the pinned environment. At
minimum it must cover:

- exact 24/26/50 task populations and list hashes;
- expected-error classification and sanitized-plan replay;
- repeated identical reference calls with exact slots;
- `retail:35` through clean and recovery materialization;
- zero failed positive labels;
- causal-cell rejection of every deterministic/oracle continuation mode;
- one unforced teacher probe per accepted error context;
- source/container/model receipt drift;
- atomic `execution_status=PASS` bundle resume for both accepted and typed
  scientific-rejected outcomes, plus partial execution rejection;
- task-local reject-and-continue;
- exact single-shard task partition and terminal closure;
- official-test seal violations; and
- final gate arithmetic.

Do not launch a model job if any test is skipped because a required V6.10
interface is absent.

## 5. Required V6.10 runtime interfaces

The exact script names may be implemented as new entry points or explicit
subcommands, but the following logical interfaces must exist and appear in
`--help` before launch.

### 5.1 Static reference preflight

The implemented entry point is:

```text
scripts/preflight_v6_reference_traces.py
```

Its exact full-universe command is:

```bash
PREFLIGHT="$ROOT/artifacts/static/<static-attempt>/reference_preflight_receipt.json"

python scripts/preflight_v6_reference_traces.py \
  --tau2-root "$TAU2" \
  --split-manifest "$SPLIT" \
  --config "$CONFIG" \
  --output "$PREFLIGHT"
```

The command is atomic and refuses to overwrite its output. It emits one
authoritative PASS/NO_GO receipt. Each per-task object inside that receipt
contains:

- exact-index reference-slot catalog;
- duplicate-semantic-call groups;
- canonical execution and repeat-determinism evidence;
- expected-error indices;
- a sanitized reference plan that deletes only expected-error slots;
- zero-error sanitized replay and final-state-equivalence evidence;
- eligible/ineligible forced-first reference indices.

The receipt must bind:

```text
protocol = v6_reference_execution_preflight_v1
artifact_type = v6_reference_execution_preflight_receipt
design_protocol = v6_10_pipeline_closure_v1
environment_reward_scope = TAU2_EVALUATION_TYPE_ENV_STRICT_REPLAY
full_official_reward_status = DEFERRED_TO_DYNAMIC_JUDGE
tau2 commit
source commit
V6.10 config hash
exact ordered 50-task formal-universe hash
all 50 task terminal statuses
embedded sanitized plans
self hash
official_test_used = false
```

The task universe comes directly from the exact 50 task IDs frozen in
`v6_10_pipeline_closure_v1` and `configs/v6_10_closure.yaml`; it must not
depend on an already generated registry. This removes a circular dependency
between sanitized reference eligibility and registry construction.

This stage is CPU/environment execution. The full 50-task receipt, not a
targeted receipt, is required by registry construction and candidate
generation.

### 5.2 V6.10 executable registry v2

After a full PASS preflight, build the executable registry v2 from the frozen
split/config plus the exact preflight-receipt hash. It must expose:

```text
compatibility: exact old 24 tasks
pilot: exact new 26 tasks
formal: exact 50 tasks
```

Only successful retained sanitized-plan action slots are candidate corrective
actions. Tasks with fewer than two distinct successful calls receive typed
rejections; they do not receive invented candidate pairs.

The registry must bind the preflight-receipt hash and embedded sanitized-plan
hashes. It must not reconstruct eligibility from unexecuted raw reference
calls.

Actual injected-error execution and crossed forced-first cells are later
registry-v2/candidate-generation gates; they are not falsely reported by the
model-free reference preflight.

The contractual interface is:

```bash
REGISTRY="$ROOT/artifacts/static/<static-attempt>/registry.runtime.json"

python scripts/prepare_v6_10_registry.py \
  --tau2-root "$TAU2" \
  --split-manifest "$SPLIT" \
  --config "$CONFIG" \
  --reference-preflight "$PREFLIGHT" \
  --output "$REGISTRY"
```

Expected list hashes:

```text
compatibility
5170b139c6307a3f46e29ab9db2573b864181c7b8ff85e27452e14f591bf27c4

pilot
930e0a0f9b6b7285a14d17d98b0b8f789f70e26b4c4625f8ac5f6cdd9eae0edb

formal
e2433544a160bdbde7c64318309c2f541833a45b3d907c1de3f02503164e0801
```

### 5.3 Frozen release manifest

After Sections 5.1–5.2 and the runtime-receipt command in Section 3 complete,
build the manifest from exact existing files:

```bash
test -n "$RUNTIME_SESSION_ID"
RUNTIME_SESSION_DIR="$ROOT/releases/$SOURCE_COMMIT/runtime-sessions/$RUNTIME_SESSION_ID"
RUNTIME_RECEIPTS="$RUNTIME_SESSION_DIR/runtime_receipts"
RELEASE_MANIFEST="$RUNTIME_SESSION_DIR/release_manifest.json"

python scripts/build_v6_10_release_manifest.py \
  --source-root "$REPO" \
  --tau2-root "$TAU2" \
  --reference-preflight "$PREFLIGHT" \
  --registry "$REGISTRY" \
  --source-container-provenance \
    "$RUNTIME_RECEIPTS/source_container_provenance.json" \
  --model-server-receipts \
    "$RUNTIME_RECEIPTS/model_server_receipts.json" \
  --runtime-receipt-hashes \
    "$RUNTIME_RECEIPTS/runtime_receipt_hashes.json" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
  --output "$RELEASE_MANIFEST"

RELEASE_MANIFEST_SHA="$(
  sha256sum "$RELEASE_MANIFEST" | awk '{print $1}'
)"
```

The manifest must bind the source commit/tree, Tau2 commit, config,
preregistration, split, relevant scripts, reference receipt, registry,
source/container receipt, model receipts, runtime-receipt hash manifest, and
container digest. Relevant scripts include candidate generation, token/
hardness measurement, candidate audit, scoring, selector construction,
materialization, directional training, and checkpoint-registry construction.
Any change to a bound byte requires rebuilding the manifest.

The reference preflight and registry are immutable static artifacts and may
be reused by a later runtime service session only when their bound source,
Tau2, split, config, and preregistration bytes remain exact. Runtime receipts
and the release manifest are never copied across service sessions.

### 5.4 Candidate generator

The generator must expose separate options equivalent to:

```text
--matched-positive-continuation-mode deterministic_reference_completion
--causal-cell-continuation-mode fresh_teacher
--first-action-measurement-mode teacher_unforced
```

For V6.10, the old shared `--recovery-continuation-mode` must not control both
positive construction and causal measurement.

The generator must additionally require:

```text
--expected-source-commit
--expected-generation-script-sha256
--expected-tau2-commit
--reference-preflight-receipt
--reference-preflight-receipt-sha256
--source-container-provenance
--source-container-provenance-sha256
--model-server-receipts
--model-server-receipts-sha256
--release-manifest
--release-manifest-sha256
--attempt-id
```

The semantic contract must bind the sanitized-plan hash and every model-server
receipt hash. Merely binding model names and endpoint URLs is insufficient.

### 5.5 Single-shard closure

This release deliberately does not claim a cross-shard merger. The generator
must reject any V6.10 launch other than `--num-shards 1 --shard-index 0`.
The single-shard closure must reject unless:

- every expected task has exactly one terminal receipt;
- terminal task IDs equal the phase registry exactly;
- all accepted task receipts share source, container, registry, sanitized
  plan, model, and semantic-contract hashes;
- every expected candidate-pair ID occurs exactly once;
- rejected tasks appear in the failure ledger and never in the accepted pool;
- no compatibility or Pilot receipt appears in formal output; and
- the official test is still sealed.

## 6. Contractual candidate-generation command

Use this template only after the interfaces in Section 5 exist and the actual
`--help` output confirms the same semantics:

```bash
set -euo pipefail

PHASE=compatibility
ATTEMPT_ID=v6_10-compat-a01-YYYYMMDDTHHMMSSZ
OUTPUT="$ROOT/artifacts/$PHASE/$ATTEMPT_ID"
REGISTRY="$ROOT/artifacts/static/<static-attempt>/registry.runtime.json"
PREFLIGHT="$ROOT/artifacts/static/<static-attempt>/reference_preflight_receipt.json"
PREFLIGHT_SHA="$(sha256sum "$PREFLIGHT" | awk '{print $1}')"
RUNTIME_SESSION_ID=v6_10-runtime-a01-YYYYMMDDTHHMMSSZ
RUNTIME_SESSION_DIR="$ROOT/releases/$SOURCE_COMMIT/runtime-sessions/$RUNTIME_SESSION_ID"
RUNTIME_RECEIPTS="$RUNTIME_SESSION_DIR/runtime_receipts"
SOURCE_CONTAINER_RECEIPT="$RUNTIME_RECEIPTS/source_container_provenance.json"
SOURCE_CONTAINER_SHA="$(
  sha256sum "$SOURCE_CONTAINER_RECEIPT" | awk '{print $1}'
)"
MODEL_RECEIPTS="$RUNTIME_RECEIPTS/model_server_receipts.json"
MODEL_RECEIPTS_SHA="$(sha256sum "$MODEL_RECEIPTS" | awk '{print $1}')"
RELEASE_MANIFEST="$RUNTIME_SESSION_DIR/release_manifest.json"
RELEASE_MANIFEST_SHA="$(
  sha256sum "$RELEASE_MANIFEST" | awk '{print $1}'
)"

TEACHER_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ
TEACHER_REV=698703eae6604af048a3d2f509995dc302088217
TEACHER_API=http://127.0.0.1:8101/v1

USER_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
USER_REV=539535859b135b0244c91f3e59816150c8056698
USER_API=http://127.0.0.1:8201/v1

python scripts/run_v6_candidate_generation.py \
  --tau2-root "$TAU2" \
  --registry "$REGISTRY" \
  --output-dir "$OUTPUT/generation/shard-00" \
  --phase "$PHASE" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-revision "$TEACHER_REV" \
  --teacher-api-base "$TEACHER_API" \
  --user-model "$USER_MODEL" \
  --user-revision "$USER_REV" \
  --user-api-base "$USER_API" \
  --judge-model "$USER_MODEL" \
  --judge-revision "$USER_REV" \
  --judge-api-base "$USER_API" \
  --clean-agent-mode single_turn_user_reference_replay \
  --matched-positive-continuation-mode deterministic_reference_completion \
  --causal-cell-continuation-mode fresh_teacher \
  --first-action-measurement-mode teacher_unforced \
  --completion-renderer explicit_user_direct_v3 \
  --continuation-seeds 20260806 \
  --max-tokens 512 \
  --max-steps 60 \
  --timeout 900 \
  --num-shards 1 \
  --shard-index 0 \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-generation-script-sha256 "$GENERATOR_SHA" \
  --expected-tau2-commit \
    fc0055dc4e0a316c3f83133267fbd6faaa770992 \
  --reference-preflight-receipt "$PREFLIGHT" \
  --reference-preflight-receipt-sha256 "$PREFLIGHT_SHA" \
  --source-container-provenance "$SOURCE_CONTAINER_RECEIPT" \
  --source-container-provenance-sha256 "$SOURCE_CONTAINER_SHA" \
  --container-image-digest "$CONTAINER_IMAGE_DIGEST" \
  --model-server-receipts "$MODEL_RECEIPTS" \
  --model-server-receipts-sha256 "$MODEL_RECEIPTS_SHA" \
  --release-manifest "$RELEASE_MANIFEST" \
  --release-manifest-sha256 "$RELEASE_MANIFEST_SHA" \
  --attempt-id "$ATTEMPT_ID"
```

This template is not authorized if matched positive construction reads the raw
unsanitized reference plan. The implementation must retrieve the frozen
sanitized plan bound by the preflight receipt.

For prospective Pilot and formal generation, change only:

```text
phase
attempt ID and fresh output directory
phase-specific registry
continuation seeds to 20260806,20260807,20260808
frozen shard count/index
runtime session ID, receipts, and release manifest if model services restarted
```

All other scientific values remain identical. A changed runtime session is
new infrastructure provenance, not permission to change a model revision,
topology, decoding value, task universe, or selector rule.

## 7. Compatibility stage

Compatibility is an engineering gate. Run:

```text
24 development tasks
3 candidate pairs per passing task
one causal continuation seed: 20260806
one unforced teacher query per accepted error branch
```

All 24 tasks must reach atomic `execution_status=PASS` bundles with scientific
outcome `ACCEPTED` or typed `REJECTED`. At least 22 tasks and 66 pairs
must pass. Every historical blocker must either pass or produce the exact
predeclared data-ineligibility reason; an unclassified failure stops the
release. The generator's `COMPATIBILITY_RELEASE_CANDIDATE` status is only a
cardinality precheck; it is not compatibility GO.

After generation reaches complete terminal closure, stop the 72B and 14B
services, measure the frozen 7B token/hardness contract, and run the closure
auditor:

Stopping those processes permanently closes the current runtime service
session. Its receipts and release manifest remain immutable evidence for the
completed compatibility attempt, but they cannot authorize a later live
generation run. Before Pilot or formal generation, launch the services under
a new `RUNTIME_SESSION_ID`, rebuild that session's runtime receipts and
release manifest, and use a fresh generation attempt directory. You may reuse
the immutable reference preflight and registry when their bound static inputs
are unchanged; do not regenerate or silently modify them, and do not
overwrite or copy the prior session's live receipts.

```bash
GEN="$OUTPUT/generation/shard-00"
MEASURED="$OUTPUT/measurement/candidate_pairs.measured.jsonl"
AUDIT_ROOT="$OUTPUT/audit"
STUDENT_MODEL=Qwen/Qwen2.5-7B-Instruct
STUDENT_REV=a09a35458c702b33eeacc393d103063234e8bc28

python scripts/measure_v6_candidate_tokens.py \
  --input "$GEN/candidate_pairs.unscored.jsonl" \
  --output "$MEASURED" \
  --model "$STUDENT_MODEL" \
  --model-revision "$STUDENT_REV" \
  --tokenizer "$STUDENT_MODEL" \
  --tokenizer-revision "$STUDENT_REV" \
  --tau2-root "$TAU2" \
  --load-in-4bit \
  --device cuda \
  --dtype bfloat16

python scripts/audit_v6_candidates.py \
  --registry "$REGISTRY" \
  --candidates "$MEASURED" \
  --phase compatibility \
  --generation-receipt "$GEN/generation_receipt.json" \
  --failure-ledger "$GEN/failure_ledger.jsonl" \
  --task-receipts-dir "$GEN/tasks" \
  --output-root "$AUDIT_ROOT"
```

Compatibility GO exists only when
`$AUDIT_ROOT/audit_report.json` reports
`gate.status=COMPATIBILITY_RELEASE_AUTHORIZED` and
`$AUDIT_ROOT/freeze_manifest.json` reports `status=FROZEN`.

Required compatibility evidence:

```text
run_contract.json
selected_phase_pre_model_preflight.json
tasks/
failure_ledger.jsonl
candidate_pairs.unscored.jsonl
generation_receipt.json
candidate_pairs.measured.jsonl
audit/audit_report.json
audit/accepted_candidate_pairs.jsonl
audit/freeze_manifest.json
files.sha256
```

Do not apply or report the scientific kappa-distribution gate on compatibility
data.

If compatibility reveals a code defect, repair it before freezing the release
and preserve the old attempt. Every tracked code/config change invalidates the
old source-bound reference receipt, registry, runtime receipts, release
manifest, run contract, and task receipts. Commit the repair, then rebuild in
this order:

```text
static reference preflight
→ executable registry
→ runtime receipts and long-context probes
→ release manifest
→ compatibility in a fresh attempt directory
```

This is still V6.10 development; do not create V6.11.

## 8. Release freeze

After audited compatibility GO:

1. require the already committed compatibility source and clean worktree;
2. rebuild or re-identify the immutable container if needed;
3. rerun the complete code-only suite;
4. if a static bound input changed (source, Tau2, split, config, or
   preregistration), rebuild static preflight and registry before creating new
   runtime receipts and a release manifest; if only the runtime model-service
   identity changed, reuse the byte-identical static preflight and registry
   and create a new runtime session with new receipts and manifest;
5. hash the final release manifest and archive it outside the Pod;
6. prohibit all code/config/model/prompt/threshold changes; and
7. start the prospective Pilot only from that exact scientific/static release,
   a live session-specific manifest, and only after the evaluator/summarizer
   blockers in Section 12 are closed.

The compatibility release commit, not an earlier candidate commit, is the
source commit used by Pilot and formal runs.

## 9. Prospective Pilot

Run the exact 26-task Pilot in fresh directories with three continuation
seeds:

```text
20260806, 20260807, 20260808
```

The runner continues after typed task-local rejection and stops only for a
global-abort condition. All 26 tasks must therefore have terminal receipts
before applying the gate.

Repeat the measurement and audit commands from Section 7 with
`--phase pilot`, the Pilot output paths, and the three frozen continuation
seeds. The generator's `GO_TO_MEASUREMENT` status is not Pilot GO.

Pilot GO requires:

- at least 16 passing tasks;
- at least 48 accepted pairs;
- two domains and at least three error families;
- zero leakage and failed positive labels;
- complete independent replay;
- at least 20% high-kappa and 20% low-kappa pairs;
- no kappa bucket above 80%;
- unforced teacher switch accuracy at least `0.80`; and
- at least `0.20` advantage over the `0.50` error-blind baseline.

If the gate fails, preserve V6.10 `NO_GO` and stop. Do not repair a failing
Pilot task and rerun the same prospective population.

Pilot GO exists only when the auditor emits `gate.status=GO_FORMAL_POOL` and a
`FROZEN` freeze manifest.

## 10. Formal generation

After Pilot GO, regenerate all 50 tasks from scratch. Do not copy any
compatibility or Pilot task receipt.

Formal outcomes:

| Qualifying tasks | Decision |
|---:|---|
| 48--50 | Formal selection authorized if all other gates pass |
| 40--47 | Screen only; no formal claim |
| below 40 | Stop |

Formal selection additionally requires at least 144 pairs and the complete
identifiability gate. `FORMAL_SELECTION_CANDIDATE` from the generator is only
authorization to measure and audit; the auditor must emit
`FORMAL_SELECTION_AUTHORIZED` before selectors may run.

Freeze the unscored candidate-pool bytes and SHA-256 before token measurement,
hardness measurement, scoring, or selector execution.

## 11. Measurement, selection, and materialization

After formal generation reaches exact terminal closure, freeze the unscored
pool bytes and hash, then execute the same measurement and closure audit used
above with `--phase formal`:

```bash
GEN="$OUTPUT/generation/shard-00"
MEASURED="$OUTPUT/measurement/candidate_pairs.measured.jsonl"
AUDIT_ROOT="$OUTPUT/audit"
SCORED="$OUTPUT/scoring/scored_candidate_pairs.jsonl"
SCORING_AUDIT="$OUTPUT/scoring/scoring_audit.json"
SELECTOR_DIR="$OUTPUT/selection/selector_manifests"
MATERIALIZED_DIR="$OUTPUT/training/data"
MATERIALIZATION_AUDITS="$OUTPUT/training/materialization_audits"

sha256sum "$GEN/candidate_pairs.unscored.jsonl" \
  > "$OUTPUT/candidate_pool.unscored.sha256"

python scripts/measure_v6_candidate_tokens.py \
  --input "$GEN/candidate_pairs.unscored.jsonl" \
  --output "$MEASURED" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --tau2-root "$TAU2" \
  --load-in-4bit \
  --device cuda \
  --dtype bfloat16

python scripts/audit_v6_candidates.py \
  --registry "$REGISTRY" \
  --candidates "$MEASURED" \
  --phase formal \
  --generation-receipt "$GEN/generation_receipt.json" \
  --failure-ledger "$GEN/failure_ledger.jsonl" \
  --task-receipts-dir "$GEN/tasks" \
  --output-root "$AUDIT_ROOT"
```

Stop unless `audit_report.json.gate.status` is
`FORMAL_SELECTION_AUTHORIZED` (or preserve the preregistered
`SCREEN_ONLY` outcome without making a formal selector claim).

For an authorized formal pool, score only the auditor-accepted rows, then
build only the preregistered directional-screen selectors:

```bash
python scripts/score_v6_candidates.py \
  --input "$AUDIT_ROOT/accepted_candidate_pairs.jsonl" \
  --output "$SCORED" \
  --audit-output "$SCORING_AUDIT"

python scripts/build_v6_selector_manifests.py \
  --scored-pool "$SCORED" \
  --output-dir "$SELECTOR_DIR" \
  --selectors flawless_only,random_stratified,full_proposed
```

Materialize each emitted arm manifest with the exact pinned tokenizer. For
example:

```bash
ARM=full_proposed
python scripts/materialize_v6_sft.py \
  --manifest "$SELECTOR_DIR/$ARM.json" \
  --output "$MATERIALIZED_DIR/$ARM.jsonl" \
  --audit-output "$MATERIALIZATION_AUDITS/$ARM.json" \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28
```

Repeat that command for `flawless_only.json` and each
`random_stratified_seed_{20260806,20260807,20260808}.json`. Do not collapse
the three random selector seeds into one unrecorded manifest.

The executable order is therefore:

1. measure exact target-token cost and frozen-base action log probabilities;
2. audit every branch, positive mask, forced-first cell, teacher probe, and
   provenance link;
3. score every accepted pair;
4. freeze the complete scored-pool hash;
5. build selector manifests;
6. verify random/full recovery arms use matched supervised-token budgets with
   at most `0.005` fractional spread;
7. materialize `flawless_only`, `random_stratified`, and `full_proposed`; and
8. prove that no failed call or tool error receives positive loss.

The flawless view must contain zero tool errors. A task that cannot produce a
successful sanitized zero-error plan is not silently relabeled as flawless.

## 12. Training and official evaluation

The fast directional screen retains:

```text
Qwen2.5-7B-Instruct
training seed 20260722
evaluation seeds 20260722, 20260723, 20260724
```

Training starts only when all three arm manifests and materialization audits
are frozen. If one arm fails training, the comparison stops; do not alter only
that arm.

Prospective Pilot remains unauthorized until the repository contains, tests,
and freezes:

```text
V6 checkpoint registry
V6 end-to-end evaluator
V6 task-cluster statistical summarizer
```

Compatibility pipeline closure alone does not satisfy this requirement. The
unseal receipt later binds the release, pool, measurements, selector
manifests, training runs, checkpoints, evaluator, summarizer, task set, and
seeds. If the evaluator or summarizer is added after compatibility, that
tracked source change invalidates the old release and requires the full
static→registry→runtime→release→compatibility rebuild from Section 7. No
change is allowed after unseal.

## 13. Resume rules

Resume is allowed only when all of the following are unchanged:

```text
attempt ID
runtime service session ID
source commit and clean tree
container digest
Tau2 commit
config and registry hashes
sanitized-plan and preflight hashes
source/container and runtime-receipt hashes
model-server receipt hashes
release-manifest hash
single-shard task mapping
seeds, decoding, and budgets
semantic contract
```

Only an atomic `execution_status=PASS` bundle is resumable. A typed
`scientific_outcome=REJECTED` is a completed execution and is resumable; it is
not rerun as an infrastructure failure. An interrupted task has its temporary
directory quarantined and restarts with the same seed. After two task-worker
process infrastructure restarts within the same live model-service session,
abort the attempt. A model-service restart instead closes the session and
requires a new runtime session and fresh generation attempt.

Any scientific or source change requires a fresh attempt directory.
Compatibility, Pilot, and formal receipts are never cross-reused.

## 14. Required final evidence package

Before a Pod or network volume is deleted, copy and verify:

```text
releases/<source-commit>/runtime-sessions/<runtime-session-id>/
  release_manifest.json
  service_specs.json
  runtime_receipts/source_container_provenance.json
  runtime_receipts/model_server_receipts.json
  runtime_receipts/runtime_receipt_hashes.json
reference_preflight_receipt.json
registry.runtime.json
run_contract.json
selected_phase_pre_model_preflight.json
tasks/
failure_ledger.jsonl
generation_receipt.json
candidate_pairs.unscored.jsonl
candidate_pool.unscored.sha256
candidate_pairs.measured.jsonl
audit/audit_report.json
audit/accepted_candidate_pairs.jsonl
audit/freeze_manifest.json
scoring/scored_candidate_pairs.jsonl
scoring/scoring_audit.json
files.sha256
selector_manifests/
materialization_audits/
training_run_manifests/
checkpoint_registry.json
official_test_unseal_receipt.json
per_task_evaluation.jsonl
statistical_summary.json
console_logs/
```

Large raw trajectories may remain private, but they must be copied to durable
storage with hashes and a retrieval location. Hash-only references to a
temporary RunPod filesystem are not sufficient research evidence.

Generate `files.sha256` last, from the attempt/evidence root, excluding the
manifest itself:

```bash
(
  cd "$OUTPUT"
  find . -type f ! -name files.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > files.sha256
)
```
