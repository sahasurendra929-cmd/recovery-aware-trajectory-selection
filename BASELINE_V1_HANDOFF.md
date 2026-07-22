# QLoRA baseline v1 handoff

## Frozen contract

Use `configs/qlora_v1.yaml`. Do not alter the seed, group split, tokenizer
protocol, model, token budget, loss, or evaluator across arms.

This is an **offline held-out next-tool-call SFT baseline**. It is not an
end-to-end Agent evaluation.

## Coordinator commands (Mac)

```bash
python3 scripts/run_data_baseline.py --data-dir /PATH/tau-bench/historical_trajectories --output-dir results/selection_v1
python3 scripts/build_sft_data.py --data-dir /PATH/tau-bench/historical_trajectories --manifest-dir results/selection_v1 --output-dir data/processed/qlora_v1
python3 scripts/evaluate_tool_actions.py --test-file data/processed/qlora_v1/shared/test.jsonl --adapter /PLACEHOLDER --output results/qlora_v1/dry_run.json --dry-run

# Produce the descriptive Table-1-style audit while GPUs train.
python3 scripts/audit_selection_data.py \
  --data-dir /PATH/tau-bench/historical_trajectories \
  --manifest-dir results/selection_v1 \
  --output-dir results/analysis_v1
```

The Mac must send the three manifests plus `data/processed/qlora_v1/` to every
GPU trainer after checking `build_summary.json`.

## GPU trainer commands

### One-time bootstrap (on every GPU machine)

Either copy `data/processed/qlora_v1/` and `results/selection_v1/` from the
Mac after it has checked `build_summary.json`, **or reproduce those files from
the pinned source revision below.** Do not mix the two methods.

```bash
git clone https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git
cd recovery-aware-trajectory-selection
git clone https://github.com/sierra-research/tau-bench.git data/raw/tau-bench
git -C data/raw/tau-bench checkout 59a200c6d575d595120f1cb70fea53cef0632f6b

python3 scripts/run_data_baseline.py \
  --data-dir data/raw/tau-bench/historical_trajectories \
  --output-dir results/selection_v1
python3 scripts/build_sft_data.py \
  --data-dir data/raw/tau-bench/historical_trajectories \
  --manifest-dir results/selection_v1 \
  --output-dir data/processed/qlora_v1
```

Confirm the last command reports exactly `validation_examples=361` and
`test_examples=900` before training.

Install compatible CUDA packages in an isolated environment:

```bash
# Install the CUDA-compatible PyTorch wheel for this machine first.
pip install -r requirements-gpu.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Run the assigned arm; replace `ARM` with exactly one of
`random_success`, `shortest_success`, or `recovery_balanced`:

```bash
python scripts/train_qlora.py \
  --train-file data/processed/qlora_v1/ARM/train.jsonl \
  --validation-file data/processed/qlora_v1/shared/validation.jsonl \
  --output-dir results/qlora_v1/ARM \
  --smoke-test

python scripts/train_qlora.py \
  --train-file data/processed/qlora_v1/ARM/train.jsonl \
  --validation-file data/processed/qlora_v1/shared/validation.jsonl \
  --output-dir results/qlora_v1/ARM

python scripts/evaluate_tool_actions.py \
  --test-file data/processed/qlora_v1/shared/test.jsonl \
  --adapter results/qlora_v1/ARM/checkpoint_final \
  --output results/qlora_v1/ARM/metrics.json
```

## Acceptance checks

1. `build_summary.json` reports no selected trace outside train tasks.
2. Each GPU trainer completes the 10-example smoke test before the formal run.
3. Each arm uploads/sends `checkpoint_final`, `run_manifest.json`, training
   log, and `metrics.json`.
4. If OOM, change only `--max-seq-len 512` to `384`, record it, and flag that
   the comparison is no longer perfectly controlled.

## Result collection (Mac)

Copy each GPU's `metrics.json` and `run_manifest.json` into the matching
`results/qlora_v1/<arm>/` directory on the Mac. Then run:

```bash
python3 scripts/aggregate_qlora_results.py \
  --results-root results/qlora_v1 \
  --output-dir results/analysis_v1/qlora_comparison
```

Before all three arms finish, the command deliberately exits non-zero and
lists missing arms. When all outputs are present, it writes `comparison.md`,
`comparison.csv`, and `comparison.json`, and checks the frozen contract.
