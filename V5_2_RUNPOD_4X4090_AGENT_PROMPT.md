# Copy-paste prompt: V5.2 on one RunPod host with 4×RTX 4090

Copy everything below into the Codex task that can operate the rented RunPod
host. The implementation commit and launch-date result branch are already
frozen.

---

You are the sole execution coordinator for the V5.2 one-seed SFT screen. Work
on one RunPod Pod with four local RTX 4090 GPUs and one shared `/workspace`.
Execute the audited pipeline through preflight, generation, preparation,
training, checkpoint registration, end-to-end evaluation, packaging, and
GitHub result-branch upload. Do not merely write a plan.

Repository:

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

Frozen implementation commit:

```text
3e5e9bb6e42435cf09a5c4b73b26ea35e9927bb5
```

Expected result branch:

```text
results/v5.2-4x4090-20260726
```

The commit above is the required 40-character published V5.2 implementation
commit. Never substitute a moving branch tip.

## Authority and limits

You are authorized to:

- clone the two public repositories;
- install pinned dependencies into new V5.2 virtual environments;
- download the frozen public model revisions;
- run all repository tests and formal V5.2 jobs;
- create files only under the experiment's data/results/runtime paths;
- create and push the small audited result branch after inspecting it.

You are not authorized to:

- alter any Python source, evaluator, config, split, task IDs, model revision,
  prompt, label mask, decoding value, attempt count, threshold, or metric;
- inspect, export, execute, or use official-test task content;
- generate more than 12 attempts for any task-condition pair;
- silently recover from OOM by truncating or changing the protocol;
- put model weights, adapters, secrets, or model caches in ordinary Git;
- report estimates or predictions as measured results;
- promise or manufacture a positive finding;
- terminate/delete the Pod, Network Volume, or repository.

At the end, stop all GPU processes and tell the user that RunPod billing must
be stopped in the console. Do not delete persistent data.

## Frozen scientific protocol

Read these files completely before doing anything else:

```text
configs/v5_2_sft_causal.yaml
V5_2_SINGLE_HOST_HANDOFF.md
V5_STAGE1_PREFLIGHT_RESULT.md
```

Also retain the machine-readable
`artifacts/v5_stage1_preflight/result.json`. Treat the config and handoff as
the execution authority.

Non-negotiable facts:

```text
tau2 commit:
fc0055dc4e0a316c3f83133267fbd6faaa770992

Stage-0 split SHA-256:
a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a

V5.2 75/8 partition SHA-256:
b476ec66996485445dd0b65f9fc982347526ff70fbec5769bbe9805aa70ce50a

attempts per task per condition:
12

task-level pairs:
clean/error may use different attempt seeds

maximum pairs per task:
2

formal train-pool gate:
at least 40 distinct train task IDs AND at least 48 pairs

generation shards:
3

evaluation shards:
4

primary contrast:
repair_50 - perfect_success

primary metric:
tau2 official composite end-to-end success under controlled error

official test:
sealed and unused
```

Four trained arms:

```text
GPU0 perfect_success
GPU1 failure_raw
GPU2 repair_50
GPU3 repair_100
```

Evaluation aliases remain the V5 compatibility aliases:

```text
openai/v5-base
openai/v5-perfect-success
openai/v5-failure-raw
openai/v5-repair-50
openai/v5-repair-100
```

Do not rename them to `v5-2-*`.

Every teacher, user/judge, and evaluation vLLM process must be launched with:

```text
--max-model-len 32768
```

Requests must use `parallel_tool_calls=false`. Tool actions are sequential:
execute one ordered call, observe its result, then replan. Preserve audit
hashes for removed mixed text or deferred calls.

## Phase 0: inspect before mutation

First report:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader
python3 --version
df -h /workspace
git --version
```

Require exactly four visible RTX 4090 GPUs with at least 24,000 MiB each and
Python 3.12. If hardware differs, stop before formal work and report the exact
difference. Do not reinterpret the frozen hardware contract.

Check that `/workspace` is writable and has enough space for repositories,
three frozen model revisions, raw 1,992-rollout data, four adapters, logs, and
packaging. Never remove an unrelated directory to make space.

## Phase 1: clone and pin

Use:

```bash
mkdir -p /workspace/repos /workspace/venvs /workspace/cache/huggingface
cd /workspace/repos

git clone \
  https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git
cd recovery-aware-trajectory-selection
git checkout --detach 3e5e9bb6e42435cf09a5c4b73b26ea35e9927bb5
git submodule update --init --recursive

mkdir -p data/raw
git clone https://github.com/sierra-research/tau2-bench.git data/raw/tau2-bench
git -C data/raw/tau2-bench checkout --detach \
  fc0055dc4e0a316c3f83133267fbd6faaa770992
git -C data/raw/tau2-bench submodule update --init --recursive
```

If the repository directories already exist, do not overwrite them. Verify
their remotes, commits, and tracked cleanliness, and either reuse the exact
clean checkout or create a new V5.2-specific checkout.

Hard checks:

```bash
test "$(git rev-parse HEAD)" = \
  "3e5e9bb6e42435cf09a5c4b73b26ea35e9927bb5"
test "$(git -C data/raw/tau2-bench rev-parse HEAD)" = \
  "fc0055dc4e0a316c3f83133267fbd6faaa770992"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git -C data/raw/tau2-bench status --porcelain --untracked-files=no)"
test "$(sha256sum artifacts/v5_stage0/manifests/split_manifest.json | \
  awk '{print $1}')" = \
  "a9fa1d0bec1f9eca500b63745ee7d405b4fc168a6f56b54806f6dea5fa67524a"
```

Save the outputs in the runtime audit directory.

Create and verify the two isolated environments:

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

If an exact existing environment is reused, verify it instead of reinstalling.
The pinned tau2 checkout is required in both environments because audited data
preparation runs with the training tokenizer stack. Do not install vLLM in the
training environment or mix the two Transformers versions.

## Phase 2: run the complete generation-readiness barrier

Use the repository's exact implemented CLI. The expected interface is:

```bash
cd /workspace/repos/recovery-aware-trajectory-selection
python scripts/run_v5_2_single_host.py --stage preflight
python scripts/run_v5_2_single_host.py --stage self-test
python scripts/run_v5_2_single_host.py --stage prefetch
python scripts/run_v5_2_single_host.py --stage protocol
```

The three stages together must verify:

- hardware, CUDA, BF16, disk, Python, packages, and two isolated environments;
- source, tau2, and immutable Stage-0 split identities;
- tracked V5.2 serving invariants;
- the V5.2 suite and all V5 regression tests used by V5.2;
- exact frozen model revisions;
- all 83 generation and all 21 derived-validation controlled injections;
- read-only error plus unchanged agent/user database hashes;
- official-test sealed status;
- 32,768 context on every vLLM role.

The V5.2 constructor later recomputes and enforces the frozen 75/8 partition
digest before it can create arm data.

If any readiness check fails, stop formal execution, preserve the evidence,
package the infrastructure/preflight failure if possible, and report it.

## Phase 3: launch the unattended formal pipeline

After preflight passes, run the complete pipeline in `tmux` so SSH loss does
not kill it:

```bash
cd /workspace/repos/recovery-aware-trajectory-selection
mkdir -p /workspace/v5_2_runtime

tmux new-session -d -s v5-2-formal \
  "python scripts/run_v5_2_single_host.py --stage all \
    2>&1 | tee /workspace/v5_2_runtime/formal_console.log"
```

Monitor rather than assuming success:

```bash
tmux capture-pane -pt v5-2-formal -S -120
nvidia-smi
```

Send concise progress updates at phase changes and at least once per hour.
Do not interrupt a healthy long generation/evaluation merely because the
program is quiet. Check process state, GPU utilization, contracts, and log
growth without modifying the run.

## Required internal phase behavior

### Generation

Use one host-local user/judge endpoint on GPU0 and one teacher on each of GPUs
1–3:

```text
GPU0 port 8001: Qwen2.5-7B-Instruct-AWQ user/judge
GPU1 port 8011: Qwen2.5-14B-Instruct-AWQ teacher, shard 0/3
GPU2 port 8012: Qwen2.5-14B-Instruct-AWQ teacher, shard 1/3
GPU3 port 8013: Qwen2.5-14B-Instruct-AWQ teacher, shard 2/3
```

All four servers use max model length 32,768. Generation must account for:

```text
83 tasks × clean/error × 12 attempts = 1,992 rollouts
```

All three shard contracts must be `COMPLETE` with matching hashes. Failed
attempts remain in the denominator. Never request a thirteenth attempt.
Do not stop formal generation early because an interim feasibility estimate
looks poor: V5.2 deliberately completes all 12 attempts for every
task-condition pair, then applies the feasibility audit to the complete pool.
Use the pinned tau2 `TextRunConfig` trial-seed behavior and record each actual
simulation seed. Do not claim an unverified domain/task/condition-specific seed
formula.

Stop all generation vLLM processes and verify CUDA memory is released before
training.

### Data preparation

The expected constructor is:

```text
scripts/prepare_v5_2_sft_causal.py
```

It must use the frozen 75 arm-train / 8 loss-validation partition. Within each
train task it independently orders eligible clean and error attempts by the
frozen salted hash of protocol seed, domain, task, condition, attempt index,
and attempt seed; it uses trajectory-content SHA-256 only for deduplication.
Before pairing, it excludes and reports any otherwise eligible attempt whose
required training variant exceeds the frozen 8,192-token no-truncation limit.
It then zips by rank and keeps at most two pairs. Same-seed matching is not
required.

Continue to training only if both are true:

```text
distinct eligible train tasks >= 40
eligible pairs >= 48
```

Also report supervised-token mass and total non-padding-token mass by arm and
source. Token mass has no hidden after-the-fact feasibility threshold, but it
is required for arm matching and interpretation.

If the data gate fails, this is a valid V5.2 fail-closed preflight:

- do not train any arm;
- do not invent partial arm metrics;
- package the complete generation/yield/exclusion evidence;
- upload a clearly labeled negative feasibility result branch.

### Training

If and only if the data gate passes, run smoke on all four GPUs. Require two
finite optimizer steps, longest-sequence success, input-hash preservation, and
no OOM. Then run formal QLoRA concurrently:

```text
GPU0 perfect_success
GPU1 failure_raw
GPU2 repair_50
GPU3 repair_100
```

Do not independently change batch size, gradient accumulation, rank, sequence
length, dtype, token budget, or optimizer steps. Stop and report any protocol
incompatibility.

### Registration

Build one immutable checkpoint registry using the existing V5 aliases. Verify
adapter model, adapter config, formal run manifest, data audit, config,
partition, and dynamic-audit hashes. Never hand-edit the registry.

### Evaluation

Start one unquantized 7B base-plus-four-LoRA server per GPU on ports 8100–8103,
all at max model length 32,768. Each GPU evaluates **all five models** on one
task-ID shard. User simulator and judge always use the unadapted
`openai/v5-base` alias.

Evaluate the same 21 derived-validation task IDs once for every model under
both clean and controlled-error conditions. The official test remains sealed.

Primary analysis:

```text
repair_50 - perfect_success
```

The directional screen passes only if:

```text
error condition: repair_50 gains at least 2 successes out of 21
clean condition: repair_50 loses no more than 1 success out of 21
exact failed-call replay rate does not increase
```

If it fails, record the negative result. A valid negative result is not a
pipeline failure.

## Phase 4: audit, package, and push

The `all` stage ends with aggregation and status. Run them explicitly if a
resumed workflow stopped immediately before aggregation:

```bash
python scripts/run_v5_2_single_host.py --stage summarize
python scripts/run_v5_2_single_host.py --stage status
```

The scientific controller intentionally does not publish to GitHub. Assemble
and inspect the small result package after status. The result package must
include provenance, environment, commands, audit and
contract hashes, attempt denominators/exclusions, pair yield, token-mass
tables, training/registry/evaluation evidence when those stages occurred,
GPU-hours/cost, and an official-test sealed receipt.

Inspect the package before Git:

```bash
git status --short
du -sh <AUDITED_SMALL_RESULT_PACKAGE>
find <AUDITED_SMALL_RESULT_PACKAGE> -type f -size +90M -print
rg -n -i 'api[_-]?key|access[_-]?token|secret|authorization: bearer' \
  <AUDITED_SMALL_RESULT_PACKAGE>
```

Investigate every credential-scan match; do not blindly commit it. Confirm no
`.safetensors`, model cache, private key, API key, or huge raw output is
staged.

Then create and upload the result branch:

```bash
git switch -c results/v5.2-4x4090-20260726
git add <AUDITED_SMALL_RESULT_PACKAGE>
test -z "$(git diff --cached --name-only | \
  grep -E '\\.safetensors$|(^|/)secrets?(/|$)|\\.pem$|id_ed25519')"
git diff --cached --stat
git commit -m "Add audited V5.2 single-host screen results"
git push -u origin results/v5.2-4x4090-20260726
```

Do not use `--force`. If authentication or repository authorization is
missing, retain the local result branch and report the exact one-line push
command for the user; do not paste credentials into the terminal transcript.

After push, verify the remote branch commit and compare the uploaded package
hash manifest with local bytes.

## Failure policy

Stop and preserve evidence if any of the following occurs:

- source/config/split/partition/tau2 hash drift;
- access to official-test content;
- fewer/more than four required GPUs;
- any vLLM server not using 32,768 context;
- failed dynamic injection or state-mutation audit;
- malformed, missing, overlapping, or incomplete generation contracts;
- an attempt index outside 0–11;
- data gate below 40 tasks or 48 pairs;
- changed input after preparation;
- nonfinite loss or OOM requiring a protocol change;
- registry/adapter/alias mismatch;
- incomplete or duplicate evaluation task coverage;
- secret or oversized model artifact in the Git package.

Do not delete failed evidence, rewrite `INCOMPLETE` to `COMPLETE`, substitute a
partial metric, or modify source. An infrastructure failure is not a
scientific negative result; a valid completed comparison whose directional
gate fails is a scientific negative screen.

## Final response format

Return one concise but complete report containing:

1. source commit, tau2 commit, config SHA, Stage-0 split SHA, and V5.2
   partition SHA;
2. hardware and environment versions;
3. PASS/FAIL status for preflight, generation, preparation, training,
   registration, evaluation, packaging, and upload;
4. all 1,992-attempt eligibility/exclusion counts;
5. distinct train tasks, pair count, cross-seed fraction, per-arm token mass,
   GPU hours, wall time, and estimated cost;
6. training status and adapter hashes for all four arms, if trained;
7. five-model clean/error end-to-end result table, if evaluated;
8. paired `repair_50 - perfect_success` result and directional-gate decision;
9. explicit statement that official test was sealed and unused;
10. GitHub result branch URL and remote commit, or the exact upload blocker;
11. exact paths to the full logs and persistent adapters;
12. confirmation that all GPU processes were stopped, plus a reminder for the
    user to stop the Pod's billing in the RunPod console.

Never say “successful experiment” merely because code ran. Distinguish:

```text
pipeline validity
data-pool feasibility
training completion
directional scientific result
paper-level confirmation
```

Only the first four are in scope, and none is guaranteed.

---
