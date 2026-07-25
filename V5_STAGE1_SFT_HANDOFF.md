# V5 Stage-1 multi-fault post-fault robustness screen: frozen four-GPU handoff

## 1. Scope, controls, and claim boundary

Stage 1 is a **validation-only directional screen**. It evaluates one
zero-shot control and four trained SFT arms:

| Name | Role |
|---|---|
| `base_model` | evaluation-only zero-shot control; never trained |
| `perfect_success` | flawless-success SFT control |
| `failure_raw` | intentionally harmful failure-supervision control |
| `repair_50` | 50% clean + 50% repair by supervised-token mass |
| `repair_100` | all-repair endpoint |

The zero-shot control is required: without it, the team cannot determine
whether SFT improved the Agent or merely selected the least damaged fine-tuned
model. The four trained arms use one source commit, base-model revision,
training schedule, evaluator, task set, and user/judge configuration.

The primary endpoint is official tau2 end-to-end composite task success under
the controlled error condition. The 21 derived-validation `task_id`s are the
independent units. Stage 1 freezes exactly one observation per task, arm, and
condition. A duplicate task row is rejected even when it carries a different
rollout or seed identifier; it is never counted as another independent task.

The 60 official-test IDs remain sealed. Stage 1 may select a method for later
replication, but it cannot support a paper-level claim.

This is a controlled data-composition and mechanism screen, not an identified
causal effect in the econometric sense. Clean and repair demonstrations are
matched on task multiset, supervised tokens, non-padding tokens, steps, and
model, but their contexts and successful action sequences necessarily differ.
The existing `*_causal_*` filenames and protocol strings are compatibility
identifiers only and must not be presented as proof of causality.

The scientific scope is
`multi_fault_family_post_fault_robustness_screen`. Every controlled fault is a
database-absent argument to a pinned read-only tool and must produce an error
without changing either database. The protocol measures whether an arm
preserves or improves task completion after those faults. It does **not**
establish that the first later successful tool call semantically repairs the
injected error. The `repair_*` arm names are retained only as compatibility
identifiers. `fault_family` summaries are descriptive because each task is
assigned one family; they do not replace or modify the frozen 21-task primary
gate.

This stage deliberately precedes PPO/TRPO/ETO. Online preference or
reinforcement learning would otherwise mix two unanswered variables: whether
the recovery data are beneficial and whether a new objective is beneficial.
Stage 1 first screens the data/label mechanism with SFT controls. A later
objective experiment may reuse only a gate-passing mixture, add explicit
clean-behavior retention, and compare continued-SFT against ETO/PPO under a
matched interaction budget.

## 2. Two isolated environments

Do not install the serving/evaluation and QLoRA stacks into one environment.
They intentionally freeze different Transformers/PyTorch combinations.

### 2.0 Fresh-clone bootstrap on every machine

Start every worker from the same detached repository commit and a separate,
pinned tau2 checkout. Replace only `<PINNED_40_CHAR_SOURCE_COMMIT>` after the
Stage-1 implementation commit is published:

```bash
mkdir -p /workspace/repos
cd /workspace/repos
git clone https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git
cd recovery-aware-trajectory-selection
git checkout --detach <PINNED_40_CHAR_SOURCE_COMMIT>
git submodule update --init --recursive

mkdir -p data/raw
git clone https://github.com/sierra-research/tau2-bench.git data/raw/tau2-bench
git -C data/raw/tau2-bench checkout --detach \
  fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C data/raw/tau2-bench submodule update --init --recursive

test "$(git rev-parse HEAD)" = "<PINNED_40_CHAR_SOURCE_COMMIT>"
test "$(git -C data/raw/tau2-bench rev-parse HEAD)" = \
  "fc0055dc4e0a316c3f83133267fbd6faaa770992"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git -C data/raw/tau2-bench status --porcelain --untracked-files=no)"
test "$(sha256sum artifacts/v5_stage0/manifests/split_manifest.json | awk '{print $1}')" = \
  "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
```

The two `status` checks are fail-closed checks for tracked changes. Untracked
logs and outputs are permitted, but no worker may edit tracked experiment code,
the tracked split manifest, or pinned tau2 source. Run the same commit and
cleanliness checks again immediately before generation, construction,
training, and evaluation.

Cloning tau2 makes its benchmark files locally available; it does not authorize
opening or selecting the official test. Stage 1 selects only the 83 inner-train
and 21 derived-validation IDs named by the tracked split manifest. The 60
official-test IDs must not be manually inspected, exported, executed, or used
for a protocol decision.

### 2.1 Tau2 generation, evaluation, and vLLM serving

```bash
cd /workspace/repos/recovery-aware-trajectory-selection
test "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.12"
python3 -m venv /workspace/venvs/v5-stage1-serve
source /workspace/venvs/v5-stage1-serve/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-stage0-v5.txt
python -m pip install -e data/raw/tau2-bench
python -m pip check
python -c "import tau2, transformers, vllm; print(transformers.__version__, vllm.__version__)"
deactivate
```

This environment runs `run_v5_sft_causal_generate.py` and
`run_v5_sft_causal_eval.py`, and may host the OpenAI-compatible vLLM servers.
Teacher and user-simulator use separate model endpoints during data
generation. During validation, one frozen 7B base server may co-host the base
alias plus all four LoRA aliases. Agent requests select the assigned arm alias;
user-simulator and judge requests always select the unadapted base alias.

### 2.2 QLoRA data preparation and training

```bash
cd /workspace/repos/recovery-aware-trajectory-selection
test "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = "3.12"
python3 -m venv /workspace/venvs/v5-stage1-train
source /workspace/venvs/v5-stage1-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu-v5-sft.txt
python -m pip check
python -c "import torch, transformers, peft, bitsandbytes, datasets, accelerate; \
print(torch.__version__, torch.version.cuda, transformers.__version__)"
deactivate
```

The required training versions are checked again inside
`train_v5_sft_causal.py`. Never install `vllm` or
`requirements-stage0-v5.txt` into this environment.

Every machine records:

```text
git commit
tau2 commit
GPU name and VRAM
NVIDIA driver
Python version
package freeze
model revisions
CUDA availability and BF16 support
```

### 2.3 Four-GPU serving topology and health barrier

GitHub synchronizes code and immutable artifacts; it does not carry live model
requests. Training is independent after barrier A, but generation and
validation require reachable OpenAI-compatible endpoints while they run.

With exactly four 24 GB RTX 4090 GPUs, freeze this generation assignment:

| Machine | Generation GPU service | Generation client shard |
|---|---|---|
| 0 | shared 7B-AWQ user simulator + judge only | none |
| 1 | local 14B-AWQ teacher | shard 0, then shard 3 |
| 2 | local 14B-AWQ teacher | shard 1 |
| 3 | local 14B-AWQ teacher | shard 2 |

All client commands retain `--num-shards 4`. Machine 1 runs shard 3 only after
its shard 0 command completes. Machine 0 must not also start a teacher. This
avoids two default vLLM engines competing for one 24 GB device while retaining
the exact four-way deterministic partition expected by the constructor.

On machines 1–3, start one local teacher:

```bash
source /workspace/venvs/v5-stage1-serve/bin/activate
export HF_HOME=/workspace/cache/huggingface
mkdir -p results/v5_sft_causal/serving

nohup vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --revision 539535859b135b0244c91f3e59816150c8056698 \
  --served-model-name Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key stage1-teacher-local \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm \
  > results/v5_sft_causal/serving/teacher.log 2>&1 &
echo $! > results/v5_sft_causal/serving/teacher.pid

curl -fsS -H "Authorization: Bearer stage1-teacher-local" \
  http://127.0.0.1:8000/health
curl -fsS -H "Authorization: Bearer stage1-teacher-local" \
  http://127.0.0.1:8000/v1/models | \
  python -c 'import json,sys; p=json.load(sys.stdin); ids={x["id"] for x in p["data"]}; assert "Qwen/Qwen2.5-14B-Instruct-AWQ" in ids, ids'
```

On machine 0, start the single shared user/judge server. Expose port 8001 only
through the team's authenticated RunPod TCP mapping or private network:

```bash
source /workspace/venvs/v5-stage1-serve/bin/activate
export HF_HOME=/workspace/cache/huggingface
mkdir -p results/v5_sft_causal/serving

nohup vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key stage1-user-local \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 3 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm \
  > results/v5_sft_causal/serving/user_judge.log 2>&1 &
echo $! > results/v5_sft_causal/serving/user_judge.pid
```

Let `<COMMON_USER_ENDPOINT>` be machine 0's authenticated `/v1` URL. Before
any shard starts, machines 1–3 must all pass:

```bash
curl -fsS -H "Authorization: Bearer stage1-user-local" \
  <COMMON_USER_ENDPOINT>/models | \
  python -c 'import json,sys; p=json.load(sys.stdin); ids={x["id"] for x in p["data"]}; assert "Qwen/Qwen2.5-7B-Instruct-AWQ" in ids, ids'
```

## 3. Bind immutable Stage 0, build Stage-1 fault manifests, and generate

Historical Stage 0 is immutable input, not a directory to regenerate for this
experiment. Its published split is frozen at seed `20260722` and SHA-256
`a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a`.
Do not run `prepare_v5_stage0.py`, rewrite a Stage-0 JSON file, or accept a
different hash. First verify the existing bytes, then build the isolated
Stage-1 protocol manifests into a fresh or empty directory:

```bash
source /workspace/venvs/v5-stage1-serve/bin/activate
test "$(sha256sum artifacts/v5_stage0/manifests/split_manifest.json | awk '{print $1}')" = \
  "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"

python scripts/prepare_v5_stage1_manifests.py \
  --tau2-root data/raw/tau2-bench \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --output-dir data/processed/v5_stage1_protocol \
  --seed 20260722

sha256sum \
  data/processed/v5_stage1_protocol/generation_manifest.json \
  data/processed/v5_stage1_protocol/validation_manifest.json \
  data/processed/v5_stage1_protocol/audit.json \
  data/processed/v5_stage1_protocol/hashes.json
```

The new manifests bind the immutable split hash, pinned tau2 commit, and
per-domain `tasks.json`, `db.json`, and `tools.py` hashes. The generation
manifest contains exactly 83 inner-train tasks; the validation manifest
contains exactly 21 derived-validation tasks. Across both manifests all 104
invalid arguments are unique and database-absent. Each row records its
`fault_family`, read-only tool, invalid argument, and whether it is
`reference_path_or_operation_aligned` or a
`domain_plausible_fallback`. No official-test row is selected, exported,
executed, or used for a protocol decision.

Before any trajectory generation, execute **every** injected call from both
manifests against the pinned environment:

```bash
python scripts/verify_v5_stage0_injections.py \
  --tau2-root data/raw/tau2-bench \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest data/processed/v5_stage1_protocol/generation_manifest.json \
  --output data/processed/v5_stage1_protocol/generation_dynamic_audit.json

python scripts/verify_v5_stage0_injections.py \
  --tau2-root data/raw/tau2-bench \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest data/processed/v5_stage1_protocol/validation_manifest.json \
  --output data/processed/v5_stage1_protocol/validation_dynamic_audit.json
```

The verifier's `stage0` filename is a legacy compatibility name; these
commands read the new Stage-1 manifests and do not write historical Stage 0.
This barrier passes only with 83/83 and 21/21 observed tool errors, unchanged
agent and user database hashes, and no duplicate invalid parameter. Publish
both audit files and their hashes. In addition, every generation and
evaluation worker must locally match the manifest's tau2 commit and
`tasks.json`/`db.json`/`tools.py` hashes; the runners fail closed on drift.
The audit files are not documentary-only: generation, data construction,
evaluation, and final summarization require their exact SHA-bound `COMPLETE`
contracts and refuse to continue if either audit is absent or altered.

Machines 1–3 now run the four fixed task shards from the table above.
`<TEACHER_ENDPOINT>` is their local
`http://127.0.0.1:8000/v1`; `<USER_ENDPOINT>` is machine 0's
`<COMMON_USER_ENDPOINT>`. Replace only those endpoints and
`<MACHINE_INDEX>`:

```bash
source /workspace/venvs/v5-stage1-serve/bin/activate
python scripts/run_v5_sft_causal_generate.py \
  --tau2-root data/raw/tau2-bench \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest data/processed/v5_stage1_protocol/generation_manifest.json \
  --dynamic-audit data/processed/v5_stage1_protocol/generation_dynamic_audit.json \
  --output-dir data/raw/v5_sft_causal_generation \
  --teacher-model Qwen/Qwen2.5-14B-Instruct-AWQ \
  --teacher-revision 539535859b135b0244c91f3e59816150c8056698 \
  --teacher-api-base <TEACHER_ENDPOINT> \
  --teacher-api-key stage1-teacher-local \
  --user-model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --user-revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  --user-api-base <USER_ENDPOINT> \
  --user-api-key stage1-user-local \
  --judge-model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --judge-revision b25037543e9394b818fdfca67ab2a00ecc7dd641 \
  --judge-api-base <USER_ENDPOINT> \
  --judge-api-key stage1-user-local \
  --teacher-mode ground_truth \
  --condition both \
  --shard-index <MACHINE_INDEX> \
  --num-shards 4 \
  --num-trials 3 \
  --seed 20260722 \
  --max-steps 60 \
  --timeout 900 \
  --max-tokens 512 \
  --expected-source-commit <PINNED_40_CHAR_SOURCE_COMMIT>
```

Shard indices are exactly `0`, `1`, `2`, and `3`; they are not the same as
the physical machine IDs in this phase. The four outputs are
merged by copying the immutable files into one directory; do not concatenate
JSON manually. Expected filenames follow:

```text
retail_clean.shard-000-of-004.json
retail_error.shard-000-of-004.json
...
run_contract.shard-003-of-004.json
```

The runner may omit a domain file when a global shard contains no task from
that domain. The preparation and aggregation layers use exact task coverage,
not the mere presence of four files per domain.

### Synchronization barrier A: generation to preparation

The four rented machines are not assumed to share a live filesystem. Each
machine stops after uploading its immutable generation shards, contract, log,
and hashes. The coordinator then:

1. verifies the published Stage-0 split hash and both Stage-1 full dynamic
   audits before accepting any shard;
2. requires all four shard contracts to be `COMPLETE`, verifies every
   declared result filename/SHA-256, and verifies exact task coverage;
3. assembles the shard directory;
4. runs the constructor exactly once;
5. publishes `audit.json`, `hashes.json`, the four training JSONL files,
   `validation_loss.jsonl`, and `validation_manifest.json`;
6. releases one common hash manifest before any training starts.

A failed generation leaves its contract `INCOMPLETE` and cannot enter the
constructor. Never edit that contract to `COMPLETE`; archive the partial
directory and rerun the same frozen shard into a fresh directory.

After all generation artifacts are uploaded, stop the serving process on
every machine before starting QLoRA (`teacher.pid` on machines 1–3 and
`user_judge.pid` on machine 0). Confirm `nvidia-smi` shows no remaining vLLM
process. Training must never share GPU memory with a generation server.

Code, manifests, hashes, logs, and small JSON/JSONL results may move through
GitHub. Do not push large model adapters into ordinary Git history. Use an
explicit RunPod network/shared volume, Git LFS, or a versioned GitHub Release
asset, and verify the published adapter hash after download.

## 4. Build and audit the four frozen SFT arms

After all generation shards are present, run the data constructor once on the
coordinator in the training environment:

```bash
source /workspace/venvs/v5-stage1-train/bin/activate
python scripts/prepare_v5_sft_causal.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --generation-manifest data/processed/v5_stage1_protocol/generation_manifest.json \
  --validation-manifest data/processed/v5_stage1_protocol/validation_manifest.json \
  --generation-dynamic-audit data/processed/v5_stage1_protocol/generation_dynamic_audit.json \
  --validation-dynamic-audit data/processed/v5_stage1_protocol/validation_dynamic_audit.json \
  --raw-dir data/raw/v5_sft_causal_generation \
  --tau2-root data/raw/tau2-bench \
  --output-dir data/processed/v5_sft_causal \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-source-commit <PINNED_40_CHAR_SOURCE_COMMIT>
```

This command must fail closed if the exact tokenizer revision is absent. Its
required outputs are:

```text
paired_master_pool.jsonl
arms/perfect_success/train.jsonl
arms/failure_raw/train.jsonl
arms/repair_50/train.jsonl
arms/repair_100/train.jsonl
validation_loss.jsonl
validation_manifest.json
audit.json
hashes.json
```

`validation_loss.jsonl` is an inner-train-held-out loss set, not the 21-task
end-to-end validation set. `validation_manifest.json` contains only the 21
paired evaluation identities and injection contract; it contains no SFT
labels. It must be a validated copy of
`data/processed/v5_stage1_protocol/validation_manifest.json`, including the
multi-fault protocol, split/source hashes, and relevance fields. Do not train
unless `audit.json` reports PASS and the four scheduled training files have
the frozen 512 rows, matched task/slot budget, label contract, expected arm
ratios, and the exact fault metadata declared by the isolated generation
manifest.

## 5. Smoke and formal training

Assign one trained arm to each machine:

| Machine | Training job |
|---|---|
| 0 | `perfect_success` |
| 1 | `failure_raw` |
| 2 | `repair_50` |
| 3 | `repair_100` |

`base_model` has no training job. Before launching, the coordinator publishes
the exact source commit, model revision, and SHA-256 values from `hashes.json`.
For each machine set its assigned `<ARM>` and frozen values:

```bash
source /workspace/venvs/v5-stage1-train/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python scripts/train_v5_sft_causal.py \
  --train-file data/processed/v5_sft_causal/arms/<ARM>/train.jsonl \
  --validation-file data/processed/v5_sft_causal/validation_loss.jsonl \
  --data-audit data/processed/v5_sft_causal/audit.json \
  --data-hashes data/processed/v5_sft_causal/hashes.json \
  --output-dir results/v5_sft_causal/<ARM>/smoke \
  --arm <ARM> \
  --mode smoke \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-source-commit <PINNED_40_CHAR_SOURCE_COMMIT> \
  --expected-train-sha256 <ARM_TRAIN_SHA256> \
  --expected-validation-sha256 <VALIDATION_LOSS_SHA256>
```

Only after the smoke run, longest-sequence check, finite-loss audit, and CUDA
memory audit pass may that machine start formal training in a fresh output
directory:

```bash
python scripts/train_v5_sft_causal.py \
  --train-file data/processed/v5_sft_causal/arms/<ARM>/train.jsonl \
  --validation-file data/processed/v5_sft_causal/validation_loss.jsonl \
  --data-audit data/processed/v5_sft_causal/audit.json \
  --data-hashes data/processed/v5_sft_causal/hashes.json \
  --output-dir results/v5_sft_causal/<ARM>/formal \
  --arm <ARM> \
  --mode formal \
  --model-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-source-commit <PINNED_40_CHAR_SOURCE_COMMIT> \
  --expected-train-sha256 <ARM_TRAIN_SHA256> \
  --expected-validation-sha256 <VALIDATION_LOSS_SHA256>
```

Do not add `--local-files-only` until the full model revision is already in
that machine's cache. An OOM or environment fallback must be frozen centrally;
one machine may not independently truncate, change rank, alter steps, or use a
different dtype.

### Synchronization barrier B: training to evaluation

Each machine stops after its formal adapter, manifest, metrics, and hashes are
uploaded. Evaluation begins only after the coordinator has all four complete
adapters, verifies their common source/model/data contracts, and publishes one
checkpoint registry. Every evaluation machine downloads the same registry and
verifies adapter hashes locally. A machine must not evaluate its own arm early
while the other arms are still changing.

After all four uploaded formal directories are present on the coordinator,
build the registry with the fail-closed constructor below. Do not hand-edit
the JSON:

```bash
python scripts/build_v5_checkpoint_registry.py \
  --source-commit <PINNED_40_CHAR_SOURCE_COMMIT> \
  --base-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --arm perfect_success=results/v5_sft_causal/perfect_success/formal \
  --arm failure_raw=results/v5_sft_causal/failure_raw/formal \
  --arm repair_50=results/v5_sft_causal/repair_50/formal \
  --arm repair_100=results/v5_sft_causal/repair_100/formal \
  --output results/v5_sft_causal/checkpoint_registry.json

sha256sum results/v5_sft_causal/checkpoint_registry.json
```

The constructor rejects smoke runs, missing or empty adapters, duplicate arm
bindings or adapter bytes, wrong arms, source commits, base revisions, and
training protocols. It recomputes and binds both each
`checkpoint_final/adapter_model.safetensors` SHA-256 and each formal
`checkpoint_final/adapter_config.json` SHA-256, plus each formal
`run_manifest.json` SHA-256. It also requires all four runs to share the same
constructor audit, hash manifest, and 83/21 dynamic-audit identities before it
atomically publishes the immutable registry.

The frozen registry is
`results/v5_sft_causal/checkpoint_registry.json`:

```json
{
  "protocol": "v5_stage1_checkpoint_registry",
  "source_commit": "<40-char experiment source commit>",
  "base_model_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
  "training_data_provenance": {
    "data_audit_sha256": "<audit.json sha256>",
    "data_hashes_sha256": "<hashes.json sha256>",
    "dynamic_audits": {
      "generation": {
        "protocol": "v5_stage1_dynamic_injection_audit",
        "sha256": "<sha256>",
        "manifest_sha256": "<sha256>",
        "split_manifest_sha256": "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a",
        "source_split": "derived_inner_train",
        "verified_injections": 83,
        "official_test_used": false,
        "official_test_sealed": true
      },
      "validation": {
        "protocol": "v5_stage1_dynamic_injection_audit",
        "sha256": "<sha256>",
        "manifest_sha256": "<sha256>",
        "split_manifest_sha256": "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a",
        "source_split": "derived_validation",
        "verified_injections": 21,
        "official_test_used": false,
        "official_test_sealed": true
      }
    },
    "official_test_used": false,
    "official_test_sealed": true
  },
  "entries": {
    "base_model": {
      "model_id": "openai/v5-base",
      "adapter_sha256": null,
      "adapter_config_sha256": null,
      "training_run_manifest_sha256": null
    },
    "perfect_success": {
      "model_id": "openai/v5-perfect-success",
      "adapter_sha256": "<adapter_model.safetensors sha256>",
      "adapter_config_sha256": "<adapter_config.json sha256>",
      "training_run_manifest_sha256": "<run_manifest.json sha256>"
    },
    "failure_raw": {
      "model_id": "openai/v5-failure-raw",
      "adapter_sha256": "<sha256>",
      "adapter_config_sha256": "<sha256>",
      "training_run_manifest_sha256": "<sha256>"
    },
    "repair_50": {
      "model_id": "openai/v5-repair-50",
      "adapter_sha256": "<sha256>",
      "adapter_config_sha256": "<sha256>",
      "training_run_manifest_sha256": "<sha256>"
    },
    "repair_100": {
      "model_id": "openai/v5-repair-100",
      "adapter_sha256": "<sha256>",
      "adapter_config_sha256": "<sha256>",
      "training_run_manifest_sha256": "<sha256>"
    }
  }
}
```

All five `model_id` values must be unique. The `openai/` prefix is LiteLLM's
provider route; the corresponding vLLM `--served-model-name` values omit that
prefix (`v5-base`, `v5-perfect-success`, and so on). The registry itself is
hashed after creation and becomes immutable. Both the user-simulator and judge
model IDs must equal `entries.base_model.model_id` exactly. A single vLLM API
base may host the unadapted base plus all LoRA adapters, provided the served
aliases remain distinct.

Before publishing the registry, the coordinator hashes every local adapter,
checks those hashes against the four training manifests, records the exact
vLLM launch command and startup log, and saves the `/v1/models` response as a
serving receipt. An alias appearing in `/v1/models` is necessary but does not
by itself prove which adapter bytes were loaded; the local pre-launch hash
check supplies that link.

Each evaluation machine now starts exactly one unquantized 7B server and
co-hosts the unadapted base plus all four LoRA adapters. It does not start a
second user model:

```bash
source /workspace/venvs/v5-stage1-serve/bin/activate
export HF_HOME=/workspace/cache/huggingface
mkdir -p results/v5_sft_causal/serving

for ARM in perfect_success failure_raw repair_50 repair_100; do
  sha256sum "results/v5_sft_causal/${ARM}/formal/checkpoint_final/adapter_model.safetensors"
  sha256sum "results/v5_sft_causal/${ARM}/formal/checkpoint_final/adapter_config.json"
done

nohup vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --served-model-name v5-base \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key stage1-eval-local \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 1 \
  --max-cpu-loras 4 \
  --lora-modules \
    v5-perfect-success=results/v5_sft_causal/perfect_success/formal/checkpoint_final \
    v5-failure-raw=results/v5_sft_causal/failure_raw/formal/checkpoint_final \
    v5-repair-50=results/v5_sft_causal/repair_50/formal/checkpoint_final \
    v5-repair-100=results/v5_sft_causal/repair_100/formal/checkpoint_final \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --generation-config vllm \
  > results/v5_sft_causal/serving/evaluation.log 2>&1 &
echo $! > results/v5_sft_causal/serving/evaluation.pid

curl -fsS -H "Authorization: Bearer stage1-eval-local" \
  http://127.0.0.1:8000/health
curl -fsS -H "Authorization: Bearer stage1-eval-local" \
  http://127.0.0.1:8000/v1/models | \
  tee results/v5_sft_causal/serving/models.json | \
  python -c 'import json,sys; p=json.load(sys.stdin); got={x["id"] for x in p["data"]}; want={"v5-base","v5-perfect-success","v5-failure-raw","v5-repair-50","v5-repair-100"}; assert got >= want, (got,want)'
```

The evaluation runner independently repeats this identity check: all four
`--adapter-dir ARM=DIR` bindings are mandatory, and both
`adapter_model.safetensors` and `adapter_config.json` must hash to their frozen
registry entries. It then queries `/v1/models` and requires all five registry
aliases before creating a run contract. Agent, user, and judge are required to
use the same API base. The contract records only the stable verified hashes and
aliases—not machine-local adapter paths—so contracts remain comparable across
workers. Any mismatch is a hard stop.

## 6. Five-way end-to-end validation

Evaluation is sharded by task ID, not by checkpoint. Every machine evaluates
the zero-shot base and all four trained models on its task shard under both
conditions. This balances machine effects across models.

The trained endpoints must expose the same non-AWQ 7B base revision with the
corresponding final adapter. The `base_model` alias exposes the identical 7B
revision without an adapter. User and judge requests select that exact
unadapted registry alias, even when all five aliases share one API base.

After the final checkpoint registry exists, run `base_model` first on all 21
paired validation tasks (21 clean + 21 error runs) as a capacity check. If
clean and error success are both zero—or otherwise so close to zero that
almost no task can distinguish the conditions—record a floor effect and pause
positive method interpretation. `base_model` is never trained, and its first
frozen-seed result is retained rather than rerun to seek a preferred result.

For each machine and each
`<EVAL_NAME> ∈ {base_model, perfect_success, failure_raw, repair_50,
repair_100}`:

```bash
source /workspace/venvs/v5-stage1-serve/bin/activate
python scripts/run_v5_sft_causal_eval.py \
  --tau2-root data/raw/tau2-bench \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --manifest data/processed/v5_sft_causal/validation_manifest.json \
  --dynamic-audit data/processed/v5_stage1_protocol/validation_dynamic_audit.json \
  --checkpoint-registry results/v5_sft_causal/checkpoint_registry.json \
  --adapter-dir perfect_success=results/v5_sft_causal/perfect_success/formal/checkpoint_final \
  --adapter-dir failure_raw=results/v5_sft_causal/failure_raw/formal/checkpoint_final \
  --adapter-dir repair_50=results/v5_sft_causal/repair_50/formal/checkpoint_final \
  --adapter-dir repair_100=results/v5_sft_causal/repair_100/formal/checkpoint_final \
  --output-dir results/v5_sft_causal/evaluation/<EVAL_NAME> \
  --arm <EVAL_NAME> \
  --agent-model <EVAL_NAME_REGISTRY_MODEL_ID> \
  --agent-api-base http://127.0.0.1:8000/v1 \
  --agent-api-key stage1-eval-local \
  --user-model openai/v5-base \
  --user-api-base http://127.0.0.1:8000/v1 \
  --user-api-key stage1-eval-local \
  --judge-model openai/v5-base \
  --judge-api-base http://127.0.0.1:8000/v1 \
  --judge-api-key stage1-eval-local \
  --condition both \
  --shard-index <MACHINE_INDEX> \
  --num-shards 4 \
  --max-steps 60 \
  --timeout 900 \
  --max-tokens 512 \
  --seed 20260722 \
  --num-trials 1
```

The validation manifest contract is:

```json
{
  "protocol": "v5_stage1_sft_causal_validation",
  "seed": 20260722,
  "source_split": "derived_validation",
  "official_test_used": false,
  "official_test_sealed": true,
  "scientific_claim_allowed": false,
  "split_manifest_sha256": "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a",
  "tau2_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
  "source_files": {
    "retail": {
      "tasks_json_sha256": "<sha256>",
      "db_json_sha256": "<sha256>",
      "tools_py_sha256": "<sha256>"
    },
    "airline": {
      "tasks_json_sha256": "<sha256>",
      "db_json_sha256": "<sha256>",
      "tools_py_sha256": "<sha256>"
    }
  },
  "fault_protocol": {
    "protocol": "v5_multifamily_readonly_faults_v1",
    "claim_scope": "multi_fault_family_post_fault_robustness_screen",
    "family_level_inference": "descriptive_only"
  },
  "paired_task_count": 21,
  "domain_counts": {"retail": 15, "airline": 6},
  "rows": ["<21 paired task rows>"]
}
```

Each error-condition row additionally binds `fault_family`, `tool_name`,
`tool_type: "READ"`, a unique invalid argument, `expected_tool_error: true`,
`expected_state_mutation: false`, `on_reference_path`, and
`fault_relevance`. The evaluation runner rechecks the local tau2 commit and all
six source-file hashes before it opens an endpoint.

Each evaluation directory may contain merged results:

```text
retail_clean.json
retail_error.json
airline_clean.json
airline_error.json
```

or runner-native shards such as:

```text
retail_clean.shard-000-of-004.json
retail_clean.shard-001-of-004.json
```

A merged file and shards for the same domain/condition may not coexist.
Across all existing shards, every one of the 21 validation task IDs must occur
exactly once per model and condition. Missing non-empty shards, overlapping
shards, duplicated task IDs, or official-test IDs fail closed.

## 7. Metrics and statistics

For each model, report:

- **Primary:** tau2 official composite end-to-end success in the error
  condition;
- tau2 official composite success in the clean condition;
- clean retention relative to `perfect_success`;
- exact repetition of the injected failed tool name and arguments after the
  observed error;
- valid post-error tool-result rate, requiring a non-error result matched to
  an assistant tool call emitted after the injected error;
- within-model error-minus-clean success;
- task-paired deltas for every model pair;
- explicit trained-arm deltas relative to the zero-shot `base_model`.

The implemented post-fault diagnostics used by the gate are only exact replay
of the injected failed call and the valid post-error tool-result rate.
`invalid_tool_call_rate`, `recovery_steps`, token counts, and wall-clock time
may be retained as `archived_raw_telemetry_not_gate`, but they are not frozen
derived metrics and must not be reported as gate evidence.

The summarizer also emits results grouped by `fault_family` (task count,
domains, clean/error success, repeated-call rate, and valid post-error result
rate). Every such group is labeled `descriptive_only`: task and family are not
independently randomized, small family cells do not support confidence claims,
and no family result changes the 21-task primary gate.

The deterministic bootstrap resamples the 21 task IDs. It never resamples
individual rollouts as though they were independent tasks. Its interval is
descriptive at this stage.

The pre-registered directional gate applies only to
`repair_50 - perfect_success`. It passes only when all three conditions hold:

\[
\sum_{i=1}^{21}
(\mathrm{InjectedSuccess}_{i,\mathrm{repair50}}
-
\mathrm{InjectedSuccess}_{i,\mathrm{perfect}})
\ge 2,
\]

\[
\sum_{i=1}^{21}
(\mathrm{CleanSuccess}_{i,\mathrm{repair50}}
-
\mathrm{CleanSuccess}_{i,\mathrm{perfect}})
\ge -1,
\]

and the repeated-identical-failed-call rate does not increase.

Passing authorizes replication; it is not a final scientific claim.

## 8. Aggregation command

```bash
source /workspace/venvs/v5-stage1-train/bin/activate
python scripts/summarize_v5_sft_causal.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --evaluation-manifest data/processed/v5_sft_causal/validation_manifest.json \
  --dynamic-audit data/processed/v5_stage1_protocol/validation_dynamic_audit.json \
  --checkpoint-registry results/v5_sft_causal/checkpoint_registry.json \
  --arm base_model=results/v5_sft_causal/evaluation/base_model \
  --arm perfect_success=results/v5_sft_causal/evaluation/perfect_success \
  --arm failure_raw=results/v5_sft_causal/evaluation/failure_raw \
  --arm repair_50=results/v5_sft_causal/evaluation/repair_50 \
  --arm repair_100=results/v5_sft_causal/evaluation/repair_100 \
  --output results/v5_sft_causal/mechanism_screen_summary.json
```

The summarizer requires the complete run-contract collection in every model
directory. Each contract must bind the exact checkpoint-registry,
evaluation-manifest, and split-manifest SHA-256; local source commit; registry
entry; multi-fault protocol; pinned tau2 commit and source-file hashes; arm and
served model ID; task shard; roles; decoding; and conditions.
The runner first writes `status: "INCOMPLETE"` and no result hashes. Only after
all files for that shard finish successfully does it atomically replace the
contract with `status: "COMPLETE"` and a filename-to-SHA-256 map. Aggregation
rejects incomplete contracts, missing or undeclared results, swapped shards,
and any post-run byte change. The frozen decoding tuple is enforced exactly:
temperature 0, 512 output tokens, 60 steps, 900 seconds, seed 20260722, one
trial, and both clean and error conditions.
The five contract collections must cover exactly the same 21 tasks and shard
schedule. Registry drift, shared aliases, missing contracts, role leakage,
manifest drift, split/test leakage, missing or duplicate tasks, cross-model
seed mismatch, missing injection evidence, or infrastructure termination all
fail closed. A scientifically valid aggregation writes `status: "PASS"` even
when the directional gate fails; protocol validity and a positive result are
separate questions.

## 9. Four-machine responsibility and artifacts

| Machine | Data generation | Training | Validation |
|---|---|---|---|
| 0 / coordinator | shared 7B-AWQ user/judge service; no teacher/client shard | `perfect_success` | task shard 0, all 5 models |
| 1 | 14B-AWQ teacher; client shards 0 then 3 | `failure_raw` | task shard 1, all 5 models |
| 2 | 14B-AWQ teacher; client shard 1 | `repair_50` | task shard 2, all 5 models |
| 3 | 14B-AWQ teacher; client shard 2 | `repair_100` | task shard 3, all 5 models |

Every machine uploads:

```text
environment_manifest.json
run_contract.shard-XXX-of-004.json
checkpoint_registry.json and its SHA-256
command.jsonl
console/evaluator logs
raw tau2 JSON shards
training run_manifest.json
training_metrics.json
checkpoint fingerprint and adapter hashes
result-file SHA-256 values
```

The coordinator retains immutable raw shards, prepared JSONL hashes, the
checkpoint registry, all checkpoint fingerprints, all five evaluation
directories and complete run contracts, the immutable Stage-0 split and
published hash, both isolated Stage-1 manifests, both full dynamic-audit
reports, the exact prepared validation-manifest copy, and
`mechanism_screen_summary.json`. No machine may edit the evaluator, rewrite
Stage 0, or open official test while another machine is running.
