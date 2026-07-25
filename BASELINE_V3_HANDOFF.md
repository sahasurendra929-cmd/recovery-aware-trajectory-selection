# QLoRA V3 RTX 5060 overnight handoff

V3 is a one-variable diagnostic experiment. It freezes the V2 model, prompts,
SFT labels, source/token budgets, training compute, validation set, and
959-example evaluator, and changes only trajectory selection.

The operational goal is to finish with a complete, resumable, and auditable
result. A positive scientific result cannot be guaranteed. The V2 test set has
already been inspected, so any V3 improvement is exploratory screening
evidence, not paper-final confirmation or end-to-end Agent success.

## Why this selector is safer than V2 Recovery Coverage

V2 Recovery Coverage raised recovery labels from 78 to 188 but reduced unique
tasks from 63 to 44 and shifted the tool distribution. V3 stops maximizing raw
recovery volume. It saturates distinct recovery signatures subject to hard
retention constraints.

The pinned data and tokenizer must reproduce this exact V3 selection before
the GPU is used:

| Check | V2 Random | V3 constrained recovery |
| --- | ---: | ---: |
| Selected SFT tokens | 1,690,929 | 1,690,929 |
| Scheduled microbatches / optimizer steps | 1,088 / 68 | 1,088 / 68 |
| Selected examples | 1,066 | 1,069 |
| Unique tasks | 63 | 76 |
| V2 Random task overlap | 63 | 63 |
| V2 Random token overlap | 100% | 67.22% |
| Recovery targets | 78 | 102 |
| Agent-initiated recovery targets | 13 | 20 |
| Non-recovery targets | 988 | 967 |
| Failed-action SFT labels | 81 | 104 |
| Target-tool total-variation distance | 0 | 0.0699 |
| Scheduled target-tool total-variation distance | 0 | 0.07445 |
| Scheduled completion/loss tokens | 36,706 | 36,599 |

V3 deliberately retains the V2 label rule, including failed tool calls. Their
count is audited and constrained; they are not silently deleted. Removing
failed labels or adding DPO changes a second factor and requires a corresponding
clean Random control in the next experiment.

## Frozen versions

- Repository tag: `v3-frozen-20260723` (the release message records its commit)
- τ-bench: `59a200c6d575d595120f1cb70fea53cef0632f6b`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Seed: `20260722`
- Python: 3.11
- PyTorch: 2.7.1 + CUDA 12.8
- Remaining packages: `requirements-gpu-v2.txt`

Do not run from a moving branch after the frozen tag is published.

## 1. Fresh deployment on Windows

Use a new directory so partial V1/V2/V3 files cannot mix.

```powershell
git clone --no-checkout https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git D:\recovery-aware-trajectory-selection-v3
git -C D:\recovery-aware-trajectory-selection-v3 config core.autocrlf false
git -C D:\recovery-aware-trajectory-selection-v3 config core.eol lf
Set-Location D:\recovery-aware-trajectory-selection-v3
git fetch origin --tags
git checkout --detach refs/tags/v3-frozen-20260723
git rev-parse HEAD
git status --short

New-Item -ItemType Directory -Force data\raw | Out-Null
git clone --no-checkout https://github.com/sierra-research/tau-bench.git data\raw\tau-bench
git -C data\raw\tau-bench config core.autocrlf false
git -C data\raw\tau-bench config core.eol lf
git -C data\raw\tau-bench checkout --detach 59a200c6d575d595120f1cb70fea53cef0632f6b
git -C data\raw\tau-bench rev-parse HEAD
git -C data\raw\tau-bench status --short
```

Both worktrees must be clean, and both printed commits must exactly match the
frozen values.

## 2. Install and preflight

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-v2.txt

.\.venv\Scripts\python.exe -c "import torch,transformers,peft,bitsandbytes; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
nvidia-smi
```

PASS requires an RTX 5060 Laptop GPU with approximately 8,151 MiB, CUDA
available, BF16 supported, the pinned packages, at least 6 GiB free VRAM, and
no other Python/CUDA model process. Keep the laptop plugged in, disable sleep,
and provide cooling. Do not set `expandable_segments:True`; the prior Windows
PyTorch build reported that allocator option as unsupported.

## 3. Run each gate separately

Do not start with `--stage overnight`. Separate stages prevent a data-contract
failure from wasting GPU time.

```powershell
$v3Python = ".\.venv\Scripts\python.exe"
$v3Data = "data\raw\tau-bench\historical_trajectories"

& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage prepare
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage audit
```

`audit` must print `status: PASS`, `selected_tier: strict`, 1,069 training
examples, 1,088 scheduled microbatches, and the frozen hashes. If it fails,
stop. Do not edit a threshold or hand-build a manifest.

Then run the CUDA smoke and formal training:

```powershell
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage smoke
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage train
```

Smoke must save a nonempty adapter, finite train/eval loss, and then load that
adapter with the frozen NF4 evaluator to force 128 new tokens from a 1,664-token
longest-prompt test case. The smoke runner alone uses its internal one-example
limit; the formal evaluation remains all 959 examples. Formal training must
complete all 68 steps and produce:

```text
results/qlora_v3/constrained_recovery/checkpoint_final/
results/qlora_v3/constrained_recovery/run_manifest.json
results/qlora_v3/constrained_recovery/training_metrics.json
results/qlora_v3/constrained_recovery/training_log.json
results/qlora_v3/constrained_recovery/training_artifact_audit.json
```

Finally, run the full evaluation:

```powershell
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage evaluate
```

It must evaluate all 959 examples with NF4, batch 1, prebuilt prompts at most
1,664 tokens, greedy decoding, and 128 generated tokens. Never pass `--limit`,
truncate prompts, shorten generation, change precision, or compute a metric
from partial predictions.

## 4. Interruption and OOM policy

Evaluation appends one prediction at a time and is safely resumable. If the
agent disconnects, first inspect whether the original process is still alive:

```powershell
nvidia-smi
Get-CimInstance Win32_Process |
  Where-Object {$_.CommandLine -match "run_qlora_v3|evaluate_tool_actions_v3"}
```

- If it is alive, do not start a second evaluator.
- If it exited and no final `metrics.json` exists, run the identical
  `--stage evaluate` command again.
- The evaluator may remove only one unterminated malformed final JSON line; it
  records that repair in `metrics.resume_recovery.jsonl`.
- Duplicate IDs, foreign IDs, middle-file corruption, changed contracts, or
  altered prior predictions are fatal.

For CUDA OOM, close unrelated GPU programs and retry the exact evaluation once.
After a second OOM, stop and preserve the checkpoint, partial predictions,
contract, command audit, and logs. Do not change the experiment.

Training is not treated as resumable. If formal training fails after writing
artifacts, keep the failed directory as evidence and restart from zero with a
new `--output-root`; never overwrite or treat a partial checkpoint as formal.

## 5. Pair with the completed V2 Random control

If the original V2 machine directory remains available, pass its
`results\qlora_v2` directory directly:

```powershell
& $v3Python scripts\run_qlora_v3.py `
  --data-dir $v3Data `
  --stage aggregate `
  --v2-results-root D:\recovery-aware-trajectory-selection-v2\results\qlora_v2
```

Otherwise clone the audited result branch and point to the directory whose
child is `random_success`:

```powershell
git clone --branch results/v2-rtx5060-20260723 --single-branch `
  https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git `
  D:\recovery-v2-reference

& $v3Python scripts\run_qlora_v3.py `
  --data-dir $v3Data `
  --stage aggregate `
  --v2-results-root D:\recovery-v2-reference\artifacts\qlora_v2\rtx5060
```

Aggregation rereads and rescores all 959 V2/V3 generated strings, rejects
contract drift, and pairs examples by ID. A missing V2 reference permits only a
standalone V3 audit and forbids a direction judgement.

## 6. Predeclared interpretation

Let \(M_{\mathrm{nonrec}}\) be full-call exact match on the 906
non-recovery examples and \(M_{\mathrm{rec}}\) the same metric on the 53
recovery examples.

\[
M_{\mathrm{nonrec}}^{V3} \ge \frac{286}{906} - 0.02 = 0.29567
\]

is the non-recovery retention gate.

\[
M_{\mathrm{rec}}^{V3} \ge \frac{31+2}{53}=0.62264
\]

is the recovery-signal gate: at least two more recovery examples correct than
V2 Random.

| Outcome | Diagnostic interpretation |
| --- | --- |
| Both gates pass | Directional support for constrained recovery selection |
| Only non-recovery passes | Ordinary behavior is retained, but no recovery signal |
| Only recovery passes | Recovery gain with unacceptable non-recovery tradeoff |
| Neither passes | This constrained selector is not supported |

These are screening gates, not significance claims. A positive pilot must
still be confirmed on fresh unseen tasks and multiple training seeds.

## 7. Expected duration

The same RTX 5060 completed each V2 training arm in roughly 15–20 minutes.
Budget:

| Stage | Expected time |
| --- | ---: |
| Clone, environment, cold downloads | 30–90 minutes |
| Prepare + audit | 5–20 minutes |
| Tests + smoke | 10–25 minutes |
| Formal training | 15–25 minutes |
| 959-example evaluation | 2–4 hours |
| Aggregation, packaging, upload | 15–40 minutes |
| Expected total | about 4–7 hours |

Reserve 8–10 hours for download, Windows, and interruption margin. V3 runs one
new arm only; do not rerun the three V2 arms.

## 8. Package only after PASS

Only after `results/analysis_v3/comparison.json` reports `valid: true`,
`v2_reference.status: compatible`, and `direction_judgement.allowed: true`,
run:

```powershell
& $v3Python scripts\package_qlora_v3_results.py
```

The package contains the adapter, metrics, full predictions, training audits,
selection/build contracts, analysis, command/environment evidence, and a
SHA256 upload manifest. It intentionally excludes raw benchmark data,
processed train/validation/test JSONL, caches, the virtual environment, and
smoke artifacts.
