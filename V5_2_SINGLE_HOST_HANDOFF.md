# V5.2 single-host 4×RTX 4090 frozen handoff

## 1. What V5.2 is—and is not

V5.2 is a one-seed, derived-validation **screen** of four SFT data
compositions. It asks whether a 50/50 mixture of flawless-success and
successful post-fault demonstrations improves end-to-end task completion
after a controlled read-only tool failure, relative to flawless-success SFT,
without materially reducing clean task success.

The four trained arms are:

| Arm | Clean share | Post-fault share | Is the injected failed action supervised? | Role |
|---|---:|---:|---:|---|
| `perfect_success` | 100% | 0% | no | flawless-success control |
| `failure_raw` | 0% | 100% | yes | intentionally harmful diagnostic control |
| `repair_50` | 50% | 50% | no | preregistered primary candidate |
| `repair_100` | 0% | 100% | no | all-post-fault endpoint |

Shares are matched by scheduled supervised-token mass, not merely by JSONL
row count. The zero-shot `base_model` is evaluation-only.

The primary contrast is:

```text
repair_50 - perfect_success
```

The primary metric is tau2 official composite **end-to-end task success in the
controlled-error condition**. Exact replay of the failed call and the
post-error valid-result rate are diagnostics, not substitutes for task
success.

V5.2 does not guarantee a positive result. Passing the directional validation
gate authorizes replication; it is not a paper-level claim. The 60 official
test task IDs remain sealed.

## 2. Why this is V5.2 rather than a continuation of the failed V5 run

The registered V5 preflight produced no eligible same-seed clean/error pairs
and correctly stopped before training. V5.2 is a new protocol revision with
three explicit changes:

1. exactly 12 fixed attempts per task and condition;
2. clean and error trajectories are paired at `task_id` level and may come
   from different attempt seeds;
3. each task contributes at most two pairs, with a formal training-pool gate
   of at least 40 distinct train task IDs and at least 48 pairs.

These rules are frozen before generation. Do not keep generating beyond 12
attempts for difficult tasks, relax the gate after observing yield, or select
pairs by validation performance, loss, length, or convenience.

V5.2 is still a controlled data-composition screen, not a trajectory-level
counterfactual causal estimate. The clean and post-fault examples may be from
different seeds and necessarily contain different contexts and action
sequences.

## 3. Immutable sources and the frozen 75/8 split

Run only from a published 40-character V5.2 implementation commit descended
from the audited V5 parent:

```text
V5 audited parent:
48dc4e89e712c63f7ebbe04caf6f8f36ac2f8cf0

tau2:
fc0055dc4e0a316c3f83133267fbd6faaa770992

Stage-0 split SHA-256:
a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a
```

Before launch replace `<V5_2_SOURCE_COMMIT>` everywhere with the exact
published V5.2 implementation commit. Never run from a moving branch tip.

The 83 derived-inner-train tasks are partitioned, within domain, into 75
arm-training tasks and 8 loss-validation tasks:

| Domain | Arm train | Loss validation |
|---|---:|---:|
| retail | 53 | 6 |
| airline | 22 | 2 |
| total | 75 | 8 |

The frozen loss-validation IDs are:

```text
retail: 21, 28, 43, 99, 110, 112
airline: 14, 27
```

The deterministic rule and complete 75/8 ID lists are in
`configs/v5_2_sft_causal.yaml`. Its canonical compact, sorted-key partition
JSON must hash to:

```text
b476ec66996485445dd0b65f9fc982347526ff70fbec5769bbe9805aa70ce50a
```

The eight loss-validation tasks do not receive gradient updates and cannot be
used to change the protocol or choose an arm. They are not the 21-task
end-to-end derived-validation set.

## 4. One RunPod host, four isolated GPUs

Required formal-screen hardware:

```text
one RunPod Pod
4 × NVIDIA RTX 4090
at least 24,000 MiB visible VRAM per GPU
one shared /workspace filesystem
Python 3.12
```

Every command that starts a GPU process must set `CUDA_VISIBLE_DEVICES`
explicitly. The generation topology is:

| GPU | Service | Port | Client shard |
|---:|---|---:|---:|
| 0 | 7B-AWQ user simulator and judge | 8001 | none |
| 1 | 14B-AWQ teacher | 8011 | 0 of 3 |
| 2 | 14B-AWQ teacher | 8012 | 1 of 3 |
| 3 | 14B-AWQ teacher | 8013 | 2 of 3 |

Generation uses exactly `--num-shards 3`. It does not reuse the former
four-machine assignment in which one teacher ran two shards.

Training runs one arm per GPU:

| GPU | Arm |
|---:|---|
| 0 | `perfect_success` |
| 1 | `failure_raw` |
| 2 | `repair_50` |
| 3 | `repair_100` |

Evaluation starts one unquantized 7B base-plus-four-LoRA vLLM server per GPU on
ports 8100–8103. Each GPU evaluates all five models on one of four
task-ID shards. This balances GPU effects across models.

V5.2 intentionally retains the evaluator's existing compatibility aliases:

```text
openai/v5-base
openai/v5-perfect-success
openai/v5-failure-raw
openai/v5-repair-50
openai/v5-repair-100
```

The `openai/` prefix is the client/provider route; vLLM serves the corresponding
names without that prefix. Renaming them to `v5-2-*` would break the frozen
registry/evaluator identity contract and is forbidden.

Generation servers must stop before training. Training processes must stop and
release CUDA memory before evaluation servers start.

## 5. Non-negotiable serving and interaction contract

Every V5.2 vLLM server—teacher, user/judge, and evaluation—must use:

```text
--max-model-len 32768
```

No 16,384 fallback is allowed. Before a formal stage, save every launch command
and query `/v1/models`. The orchestrator must reject a launch receipt that does
not declare 32,768.

All Agent interactions use a sequential tool-action interface:

- requests send `parallel_tool_calls=false`;
- the assistant executes one tool action and then replans from its result;
- if a provider returns ordered parallel calls, execute only the first call
  and defer the rest until replanning;
- if a response mixes text and a tool call, remove the text before executing
  the call;
- SHA-256 metadata for removed text and deferred calls is retained;
- empty or malformed simulations are excluded, never converted into labels.

The interface normalization must not silently erase evidence. Counts and
hashes for every normalization category belong in the audit.

## 6. Workspace and environments

Use one repository and two separate virtual environments:

```text
/workspace/repos/recovery-aware-trajectory-selection
/workspace/repos/recovery-aware-trajectory-selection/data/raw/tau2-bench
/workspace/venvs/v5-2-serve
/workspace/venvs/v5-2-train
/workspace/cache/huggingface
/workspace/v5_2_runtime
```

Serving/evaluation dependencies and QLoRA dependencies must not be installed
into the same environment. The train environment freezes PyTorch 2.7.1
CUDA 12.8 and the repository's V5 SFT requirements. `vllm` must not be
installed into the train environment.

Bootstrap:

```bash
mkdir -p /workspace/repos /workspace/venvs /workspace/cache/huggingface
cd /workspace/repos
git clone https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git
cd recovery-aware-trajectory-selection
git checkout --detach <V5_2_SOURCE_COMMIT>
git submodule update --init --recursive

mkdir -p data/raw
git clone https://github.com/sierra-research/tau2-bench.git data/raw/tau2-bench
git -C data/raw/tau2-bench checkout --detach \
  fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C data/raw/tau2-bench submodule update --init --recursive
```

Create the two frozen environments before invoking the controller:

```bash
python3 -m venv /workspace/venvs/v5-2-serve
source /workspace/venvs/v5-2-serve/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-stage0-v5.txt
python -m pip install -e data/raw/tau2-bench
python -m pip check
deactivate

python3 -m venv /workspace/venvs/v5-2-train
source /workspace/venvs/v5-2-train/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu-v5-sft.txt
python -m pip install -e data/raw/tau2-bench
python -m pip check
deactivate
```

The editable pinned tau2 checkout is installed in both environments. The
training environment needs it during audited data preparation, while vLLM
remains exclusive to the serving environment.

The frozen orchestration CLI is:

```bash
python scripts/run_v5_2_single_host.py --stage <STAGE>
```

`<STAGE>` is one of:

```text
preflight
self-test
prefetch
protocol
generate
prepare
train
registry
evaluate
summarize
all
status
stop-services
```

The implementation must make repeated calls idempotent by verifying immutable
complete contracts. It may resume only documented, hash-verified units. It
must never reinterpret a partial output as complete.

## 7. Stage-by-stage execution

### 7.1 Preflight

Run:

```bash
python scripts/run_v5_2_single_host.py --stage preflight
```

The `preflight` stage must:

1. verify four RTX 4090 GPUs, VRAM, driver, CUDA, BF16, Python, free disk, and
   writable `/workspace`;
2. verify the detached repository commit, clean tracked worktree, pinned tau2
   commit, and Stage-0 split hash;
3. verify the two already-created isolated environments;
4. require all reserved ports to be free and at least 140 GiB free disk;
5. verify the tracked handoff/config/prompt serving invariants;
6. write the preflight receipt.

The split validator may read the already-published official-test IDs only as
a leakage-exclusion set. Official-test task contents, prompts, environments,
rollouts, rewards, and outcomes must not be inspected, exported, or executed.

Then prefetch the three exact model revisions and build/execute the full
dynamic fault protocol:

```bash
python scripts/run_v5_2_single_host.py --stage self-test
python scripts/run_v5_2_single_host.py --stage prefetch
python scripts/run_v5_2_single_host.py --stage protocol
```

`self-test` runs the V5.2 tests and all V5 regression tests on which V5.2
depends. `protocol` must build the isolated V5.2 generation/evaluation manifests,
dynamically execute all 83 generation and all 21 derived-validation injected
read-only calls, require each injection to return an error and preserve both
database hashes, and publish `COMPLETE` audits. Together,
`preflight + self-test + prefetch + protocol` form the generation-readiness barrier.
The 75/8 partition and its frozen digest are recomputed and enforced by the
V5.2 constructor before arm construction.
The controller verifies but does not install the two environments.

### 7.2 Generate

Run:

```bash
python scripts/run_v5_2_single_host.py --stage generate
```

The stage starts the GPU0 user/judge service and the three teacher services,
checks all four health/model receipts, then runs shards 0–2 concurrently.

Frozen workload:

```text
83 tasks × 2 conditions × 12 attempts = 1,992 rollouts
```

Every task-condition pair has exactly attempt indices 0–11. Failed attempts
remain in the denominator and audit. The runner must not request attempt 12 or
higher. Formal V5.2 does not use an interim feasibility stop: all 12 frozen
attempts are completed for every task and condition even if an intermediate
yield estimate looks poor. The feasibility auditor is applied to the complete
generation pool during preparation. This preserves a common denominator and
the preregistered selection rule.

Trial seeds follow the pinned tau2 `TextRunConfig` behavior, and every
simulation records the actual seed. Do not invent or claim a
domain/task/condition-specific seed formula; the same raw trial seed may recur
across tasks.

Only a shard with a `COMPLETE` contract and verified result hashes may enter
preparation. Any incomplete shard remains `INCOMPLETE`; archive and rerun the
same frozen shard rather than editing its contract.

After all three shards verify, stop all four generation servers and confirm
zero associated GPU processes remain.

### 7.3 Prepare and enforce the formal data gate

Run:

```bash
python scripts/run_v5_2_single_host.py --stage prepare
```

The expected underlying constructor is:

```bash
python scripts/prepare_v5_2_sft_causal.py \
  --split-manifest artifacts/v5_stage0/manifests/split_manifest.json \
  --generation-manifest data/processed/v5_2_protocol/generation_manifest.json \
  --validation-manifest data/processed/v5_2_protocol/evaluation_manifest.json \
  --generation-dynamic-audit data/processed/v5_2_protocol/generation_dynamic_audit.json \
  --validation-dynamic-audit data/processed/v5_2_protocol/evaluation_dynamic_audit.json \
  --raw-dir data/raw/v5_2_sft_causal_generation \
  --tau2-root data/raw/tau2-bench \
  --output-dir data/processed/v5_2_sft_causal \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --expected-source-commit <V5_2_SOURCE_COMMIT> \
  --expected-generation-source-commit <V5_2_SOURCE_COMMIT> \
  --local-files-only
```

For each of the 75 arm-train tasks, the constructor:

1. filters clean and error attempts independently by frozen eligibility;
2. deduplicates and excludes any attempt whose required training variant
   exceeds the frozen 8,192-token no-truncation limit before pairing;
3. orders eligible attempts by the SHA-256 of
   `v5.2-eligible-attempt-order|20260722|domain|task_id|condition|attempt_index|attempt_seed`,
   with attempt index and attempt seed as tie breakers; trajectory-content
   SHA-256 is used for deduplication, not ranking;
4. zips clean and error lists by rank;
5. retains no more than the first two pairs.

Same-seed pairing is not required. Pair selection cannot depend on loss,
trajectory length, validation/test outcome, or manual judgment.

Before any training, both formal minima must pass on the 75 arm-train tasks:

```text
at least 40 distinct task IDs
at least 48 task-level clean/error pairs
```

The constructor must also report, without inventing an after-the-fact hidden
threshold:

- selected and available clean/error trajectories by task;
- pair count and cross-seed-pair fraction;
- all exclusion reasons over all 1,992 attempts;
- per-variant sequence lengths and the count excluded by the 8,192-token
  training limit;
- distinct supervised-token mass by source and arm;
- scheduled supervised-token mass by source and arm;
- total non-padding token mass;
- failure-action supervised-token mass in `failure_raw`;
- pair yield per GPU-hour.

The four arms must draw from the same eligible task-level pair pool, have the
same scheduled task multiset, and match scheduled supervised-token mass within
1%, total non-padding tokens within 2%, and optimizer steps exactly. Token
matching may select a different eligible pair occurrence within the same task,
so identical pair-occurrence multisets are not required.

If either 40 tasks or 48 pairs is missing, the run is a valid fail-closed
preflight. Stop before training and package the negative feasibility result.
Do not lower the gate.

### 7.4 Smoke and formal training

Run:

```bash
python scripts/run_v5_2_single_host.py --stage train
```

The orchestrator launches one smoke run per arm on its frozen GPU. Each smoke
must complete two optimizer steps, produce finite loss, pass the
longest-sequence check, preserve input hashes, and remain within VRAM.

Only then may it launch four formal QLoRA jobs concurrently into fresh output
directories. An individual job may not change LoRA rank, sequence length,
quantization, gradient accumulation, dtype, tokenizer, batch order, or
optimizer steps in response to OOM. A centrally frozen protocol revision is
required for any fallback.

### 7.5 Register checkpoints

Run:

```bash
python scripts/run_v5_2_single_host.py --stage registry
```

The registry constructor must reject missing/empty adapters, smoke
checkpoints, duplicate adapter bytes, wrong arms, mismatched source/model/data
hashes, or changed run manifests. It binds:

- the unadapted base revision;
- all four adapter and adapter-config SHA-256 values;
- all four formal run-manifest SHA-256 values;
- the data audit and hash manifest;
- V5.2 partition, fault-manifest, and dynamic-audit identities;
- `official_test_used: false` and `official_test_sealed: true`.

Do not hand-edit the registry.

### 7.6 Five-way end-to-end evaluation

Run:

```bash
python scripts/run_v5_2_single_host.py --stage evaluate
```

The evaluator:

1. starts four base-plus-LoRA servers, one per GPU, on ports 8100–8103, every
   one with `--max-model-len 32768`;
2. verifies all five aliases and local adapter hashes;
3. uses the unadapted base alias for user simulator and judge;
4. evaluates all five models under clean and controlled-error conditions;
5. assigns one task-ID shard to each GPU, while each GPU runs every model;
6. uses exactly one observation per task, model, and condition.

The 21 derived-validation task IDs are independent units. Rollouts, training
rows, pairs, and token counts do not enlarge the statistical sample size.

The directional gate passes only if all three conditions hold:

```text
repair_50 has at least 2 additional error-condition successes out of 21
repair_50 loses at most 1 clean-condition success out of 21
repair_50 does not increase exact replay of the injected failed call
```

A protocol-valid aggregation is `PASS` even if this directional gate fails.
Scientific validity and a positive direction are separate.

### 7.7 Package and upload

First complete aggregation and print the frozen status:

```bash
python scripts/run_v5_2_single_host.py --stage summarize
python scripts/run_v5_2_single_host.py --stage status
```

The scientific controller deliberately does not push to GitHub or include
large weights. After `summarize` and `status` pass, the Agent assembles and
audits the small result package. The package must contain:

```text
README/result status and claim boundary
exact source and tau2 commits
configuration and partition hashes
environment manifest and package freeze
commands.jsonl and stage logs
dynamic fault audits
all generation run contracts and result hashes
pool-yield/exclusion audit
prepared-data hashes and arm token-mass table
four training manifests and metrics
checkpoint registry and adapter fingerprints
all evaluation contracts and result hashes
mechanism-screen summary
GPU-hours and wall-clock accounting
official-test sealed receipt
```

Do not put `.safetensors`, model-cache files, secrets, API keys, or unbounded
raw logs into ordinary Git. Keep adapters on persistent RunPod storage or a
versioned release asset and publish their SHA-256 values.

For the small audited package, create a result branch from the exact source
commit:

```bash
git switch -c results/v5.2-4x4090-<UTC_DATE>
git add <AUDITED_SMALL_RESULT_PACKAGE>
test -z "$(git diff --cached --name-only | grep -E '\\.safetensors$|(^|/)secrets?(/|$)')"
git commit -m "Add audited V5.2 single-host screen results"
git push -u origin results/v5.2-4x4090-<UTC_DATE>
```

Before pushing, inspect the staged file list and package size, scan for
credentials, and confirm tracked experiment code/config still matches
`<V5_2_SOURCE_COMMIT>`. If Git authentication is absent, stop after packaging
and report the exact branch/push command; do not paste a token into logs.

## 8. One-command formal run

After the implementation commit, tests, model downloads, and Git
authentication have all passed preflight, the intended unattended command is:

```bash
cd /workspace/repos/recovery-aware-trajectory-selection
source /workspace/venvs/v5-2-serve/bin/activate

tmux new-session -d -s v5-2-formal \
  "python scripts/run_v5_2_single_host.py --stage all \
    2>&1 | tee /workspace/v5_2_runtime/formal_console.log"
```

The Agent must monitor the run rather than assume success. `--stage all` must
stop automatically on:

- source/config/manifest/hash drift;
- failed dynamic injection audit;
- non-32,768 vLLM launch;
- malformed or incomplete generation;
- fewer than 40 train task IDs or 48 pairs;
- nonfinite loss, OOM, or modified training input;
- checkpoint-registry mismatch;
- incomplete or overlapping evaluation coverage;
- official-test access;
- package hash or credential-scan failure.

## 9. Required final report

The final Agent response must state:

1. exact commit/config/partition/tau2 hashes;
2. whether preflight, generation, preparation, training, registration,
   evaluation, and package stages each passed;
3. all-attempt eligibility and exclusion counts;
4. distinct train tasks, pairs, cross-seed fraction, and token mass;
5. per-arm training status and adapter hashes;
6. complete five-way clean/error end-to-end results;
7. the preregistered `repair_50 - perfect_success` paired delta and whether
   the directional gate passed;
8. GPU hours and estimated cost;
9. official-test sealed status;
10. result branch URL or the exact reason upload did not occur.

Never report predicted values as measured values. If the gate fails, report the
negative result. If infrastructure fails, label it infrastructure failure
rather than a scientific result.
