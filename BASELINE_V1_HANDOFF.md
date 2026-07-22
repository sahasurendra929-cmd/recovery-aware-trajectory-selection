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
```

The Mac must send the three manifests plus `data/processed/qlora_v1/` to every
GPU trainer after checking `build_summary.json`.

## GPU trainer commands

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
