# QLoRA v2 handoff

V2 is a new, isolated experiment. It does not change `scripts/run_qlora_v1_1.py`,
`configs/qlora_v1.yaml`, `data/processed/qlora_v1_1/`, or
`results/qlora_v1_1/`. Let an in-progress v1.1 run finish; do not run v1.1 and
v2 concurrently on the same GPU.

## What v2 repairs

1. The complete system policy is retained in every prompt. The initial task and
   recent history are truncated by tokenizer-aware message rules, never by
   slicing the serialized prompt or target JSON.
2. The three selected datasets have exactly the same tokenizer-measured SFT
   budget after trajectory-to-example expansion.
3. Each source model receives exactly the same token quota in every arm.
4. Every selected example is exposed at least once, and all arms use the same
   padded microbatches and optimizer steps.
5. Evaluation uses the exact prebuilt prompt, saves one prediction at a time,
   resumes after interruption, includes a zero-shot base-model control, and
   reports task-macro metrics plus task-cluster bootstrap intervals.

The result is still **offline next-tool-call imitation**, not executable or
end-to-end Agent success.

## While the RTX 5060 is running v1.1

Do nothing on that machine. The Mac can prepare and audit v2 independently.
Do not `git pull`, install packages, start v2, or reuse v1.1 output folders
until the v1.1 process and result copy are complete.

## Frozen sources

- τ-bench: `59a200c6d575d595120f1cb70fea53cef0632f6b`
- Qwen2.5-0.5B-Instruct: `7ae557604adf67be50417f59c2c2f167def9a775`
- experiment config: `configs/qlora_v2.yaml`

## Mac: prepare and audit only

Use Python 3.11. The Mac does not need PyTorch or CUDA for this step.

```bash
git clone https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git
cd recovery-aware-trajectory-selection
git clone https://github.com/sierra-research/tau-bench.git data/raw/tau-bench
git -C data/raw/tau-bench checkout 59a200c6d575d595120f1cb70fea53cef0632f6b

python3.11 -m venv .venv
.venv/bin/python -m pip install transformers==4.52.4 sentencepiece==0.2.0
.venv/bin/python scripts/run_qlora_v2.py \
  --arm all \
  --data-dir data/raw/tau-bench/historical_trajectories \
  --stage prepare
```

Preparation must stop on any contract violation. Its current audited output is:

- task split: 78 train / 11 validation / 22 test task keys;
- shared examples: 392 validation / 959 test;
- exactly 1,690,929 selected SFT tokens per arm;
- source-token quotas: 523,182 GPT-4o + 1,167,747 Sonnet per arm;
- 1,088 padded microbatches and 68 optimizer steps per arm;
- 2,228,224 padded training tokens per arm.

These counts are evidence checks, not accuracy results.

## RTX 5060 Laptop 8 GB: install after v1.1 finishes

Open a fresh PowerShell in a fresh clone or clean v2 branch. Python 3.11 is
required. CUDA 12.8 PyTorch wheels support RTX 50-series GPUs; the runner also
checks CUDA and bfloat16 before doing any model work.

```powershell
git clone https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git
cd recovery-aware-trajectory-selection
git clone https://github.com/sierra-research/tau-bench.git data/raw/tau-bench
git -C data/raw/tau-bench checkout 59a200c6d575d595120f1cb70fea53cef0632f6b

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-v2.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
```

## Smoke first, then one-machine sequential formal run

Prepare data once:

```powershell
.\.venv\Scripts\python.exe scripts\run_qlora_v2.py --arm all --data-dir data\raw\tau-bench\historical_trajectories --stage prepare
```

Run a smoke test for each arm. A smoke passes only if it saves
`results/qlora_v2/smoke/<arm>/checkpoint_final/`, produces finite train/eval
loss, and has no CUDA OOM or tokenizer-contract error.

```powershell
.\.venv\Scripts\python.exe scripts\run_qlora_v2.py --arm all --data-dir data\raw\tau-bench\historical_trajectories --stage smoke
```

If all smoke tests pass, run all trained arms sequentially. `--resume` is built
into evaluation, so a stopped evaluation resumes from its prediction JSONL.

```powershell
.\.venv\Scripts\python.exe scripts\run_qlora_v2.py --arm all --data-dir data\raw\tau-bench\historical_trajectories --stage train
.\.venv\Scripts\python.exe scripts\run_qlora_v2.py --arm all --data-dir data\raw\tau-bench\historical_trajectories --stage evaluate
```

Do not run multiple arms concurrently on the same 8 GB GPU. Do not change the
context cap, quantization, generation length, schedule, seed, or evaluation
subset for only one arm.

## Result validation

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_qlora_v2.py
```

Aggregation deliberately fails if an arm is missing, was evaluated with a
limit, used a different model revision/environment, or violated the frozen
training/evaluation contract. Formal outputs live only under
`results/qlora_v2/`; v1.1 outputs remain unchanged.
