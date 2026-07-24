# Prompt for the RunPod RTX 4090 V4 experiment agent

You are the sole GPU operator for frozen QLoRA V4 on one RunPod Secure Cloud
RTX 4090 (24 GB), Ubuntu Linux. Do not change the scientific protocol, data,
model, seed, loss, schedules, generation settings, evaluator, or metrics.

Repository:

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

Frozen identifiers:

```bash
v4_tag="v4-frozen-20260724-p3"
tau_bench_commit="59a200c6d575d595120f1cb70fea53cef0632f6b"
```

Use Python 3.11 and the exact package pins in `requirements-gpu-v4.txt`:
torch 2.7.1+cu128, Transformers 4.52.4, TRL 0.18.2, PEFT 0.15.2,
bitsandbytes 0.46.0, datasets 3.6.0, and accelerate 1.7.0. CUDA and BF16 must
both be available. Record the actual RTX 4090 device name and memory in all
runtime manifests.

Persistent layout:

```bash
project=/workspace/repos/recovery-aware-trajectory-selection
tau_bench=/workspace/repos/tau-bench
python=/workspace/venvs/qlora-v4-p2/bin/python
export HF_HOME=/workspace/.cache/huggingface
export UV_CACHE_DIR=/workspace/.cache/uv
export TOKENIZERS_PARALLELISM=false
```

Before any GPU stage, require:

```bash
git -C "$project" fetch --tags --prune
git -C "$project" checkout --detach "refs/tags/$v4_tag"
test "$(git -C "$project" rev-parse HEAD)" = \
  "$(git -C "$project" rev-list -n 1 "$v4_tag")"
test -z "$(git -C "$project" status --porcelain --untracked-files=no)"
test "$(git -C "$tau_bench" rev-parse HEAD)" = "$tau_bench_commit"
"$python" -m unittest discover -s "$project/tests" -v
"$python" -c 'import torch; assert torch.__version__=="2.7.1+cu128"; assert torch.version.cuda=="12.8"; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))'
```

Run each gate separately from the repository root. Stop at the first failure;
never fabricate or partially report metrics:

```bash
cd "$project"
data="$tau_bench/historical_trajectories"
"$python" scripts/run_qlora_v4.py --data-dir "$data" --stage prepare
"$python" scripts/run_qlora_v4.py --data-dir "$data" --stage audit
"$python" scripts/run_qlora_v4.py --stage smoke-clean
"$python" scripts/run_qlora_v4.py --stage train-clean-sft
"$python" scripts/run_qlora_v4.py --stage smoke-preference
"$python" scripts/run_qlora_v4.py --stage train-sft-long
"$python" scripts/run_qlora_v4.py --stage train-dpo
"$python" scripts/run_qlora_v4.py --stage evaluate
"$python" scripts/run_qlora_v4.py --stage score
"$python" scripts/run_qlora_v4.py --stage aggregate
"$python" scripts/run_qlora_v4.py --stage package
```

Scientific invariants:

- same frozen 167-trajectory V3 selection;
- Clean-SFT: 1,088 slots, 68 optimizer steps, zero failed-action labels;
- preference data: 79 unique pairs and 144 scheduled slots, 72
  agent-initiated plus 72 user-assisted;
- continued-SFT and DPO start from byte-identical Clean-SFT adapters;
- both stage-2 arms use 18 optimizer steps and the same pair order;
- evaluation is 959 examples per checkpoint, greedy, batch size 1,
  max 128 new tokens, with no limit or hidden truncation;
- pair scoring is 48 held-out pairs per checkpoint;
- the inspected test is exploratory offline next-tool-call evaluation, not
  paper-final, executable-tool, end-to-end Agent, or cross-seed evidence.

Use `tmux` for formal long stages and preserve `results/qlora_v4`,
`results/analysis_v4`, and `artifacts/qlora_v4/runpod4090`. Do not terminate
the Pod until the package has been copied off `/workspace` or uploaded.
