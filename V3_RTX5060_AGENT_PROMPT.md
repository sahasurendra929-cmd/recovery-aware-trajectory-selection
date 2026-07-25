# Prompt for the RTX 5060 experiment agent

Copy everything below into the Codex agent on the RTX 5060 machine after the
V3 frozen commit is published.

---

你是 QLoRA V3 的唯一 GPU 操作员。请在 NVIDIA GeForce RTX 5060 Laptop
GPU（8 GB）上完整执行 `constrained_recovery` 单臂诊断实验，并在通过全部
审计后上传结果。

仓库：

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

冻结源码 tag：

```text
v3-frozen-20260723
```

权威操作说明：

```text
BASELINE_V3_HANDOFF.md
configs/qlora_v3.yaml
```

必须遵守：

1. 使用全新目录 `D:\recovery-aware-trajectory-selection-v3`，checkout 上述
   tag 的 detached HEAD；不要在 V1/V2 目录内运行。
2. τ-bench 必须 checkout
   `59a200c6d575d595120f1cb70fea53cef0632f6b`。
3. 使用 Python 3.11、torch 2.7.1 + cu128 和
   `requirements-gpu-v2.txt` 的精确版本。
4. V3 只改变 selection。不得修改模型、seed、SFT label、prompt、source
   quota、token budget、1088/68 schedule、NF4、BF16、959 条 test、
   1664 prompt cap、128 generation 或 evaluator。
5. 正式评估禁止 `--limit`、截断 prompt、缩短 generation、跳过样本、
   手算/补写 metrics、伪造结果、把 offline next-tool-call 称为
   end-to-end Agent 成功。`--stage smoke` 会由冻结 runner 内部自动对
   最长 prompt 做一次单样本显存检查，不得手工更改。
6. 不要因为结果不好而改阈值或重选数据。科学结果可以为负；流程必须完整。

按以下顺序执行，每一步成功后再进入下一步：

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

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-v2.txt
.\.venv\Scripts\python.exe -c "import torch,transformers,peft,bitsandbytes; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0)); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
nvidia-smi

$v3Python = ".\.venv\Scripts\python.exe"
$v3Data = "data\raw\tau-bench\historical_trajectories"

& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage prepare
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage audit
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage smoke
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage train
& $v3Python scripts\run_qlora_v3.py --data-dir $v3Data --stage evaluate
```

Prepare/Audit 只有在以下值全部一致时才可继续：

```text
selected_tier=strict
train examples=1069
unique tasks=76
recovery targets=102
agent-initiated targets=20
non-recovery targets=967
selected SFT tokens=1690929
microbatches=1088
optimizer steps=68
test examples=959
trace fingerprint=a65bba64baf7c9a6e816e721b382511211aa9df6f5204e7c4cce74f78b992cc5
train hash=6b991fe03c7b79132438f8681dccee9e4fab2003a5859bd1abce26ba32ed046d
schedule hash=46acdd204d3dc213389af9b44ed6884031899a82615f8a9be47e024c30e2ea38
test hash=0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96
```

Smoke 还必须报告 `longest_prompt_evaluation=PASS`、
`longest_prompt_tokens=1664`、`evaluation_max_new_tokens=128` 和
`evaluation_loading=nf4_4bit`；该检查会强制实际生成 128 tokens，否则
不得进入正式训练。

评估断线处理：

- 先用 `nvidia-smi` 和 `Get-CimInstance Win32_Process` 检查原进程；
- 原进程仍在时禁止启动第二个 evaluator；
- 原进程已退出且没有最终 metrics 时，原样重跑同一条
  `--stage evaluate` 命令，保留 partial predictions；
- 若 OOM，关闭其他 GPU 程序后原参数重试一次；第二次仍 OOM 就停止并
  保存证据，禁止改协议。

评估完成后，优先与本机原 V2 结果配对。如果路径存在：

```powershell
& $v3Python scripts\run_qlora_v3.py `
  --data-dir $v3Data `
  --stage aggregate `
  --v2-results-root D:\recovery-aware-trajectory-selection-v2\results\qlora_v2
```

如果该路径不存在，则 clone 结果分支：

```powershell
git clone --branch results/v2-rtx5060-20260723 --single-branch `
  https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git `
  D:\recovery-v2-reference

& $v3Python scripts\run_qlora_v3.py `
  --data-dir $v3Data `
  --stage aggregate `
  --v2-results-root D:\recovery-v2-reference\artifacts\qlora_v2\rtx5060
```

要求 `results\analysis_v3\comparison.json` 中：

```text
valid=true
v2_reference.status=compatible
direction_judgement.allowed=true
```

`direction_judgement.label` 可以是正向、trade-off 或负向，必须照实保留。

然后打包并验证：

```powershell
& $v3Python scripts\package_qlora_v3_results.py
Get-Content artifacts\qlora_v3\rtx5060\UPLOAD_MANIFEST.json
```

最后只上传生成的审计包，不上传 raw data、processed JSONL、`.venv`、cache
或 smoke：

```powershell
git switch -c results/v3-rtx5060-20260724
git add artifacts\qlora_v3\rtx5060
git ls-files --error-unmatch artifacts/qlora_v3/rtx5060/results/qlora_v3/constrained_recovery/checkpoint_final/adapter_model.safetensors
git status --short
git commit -m "Add complete RTX 5060 QLoRA v3 result"
$v3Adapter = "artifacts/qlora_v3/rtx5060/results/qlora_v3/constrained_recovery/checkpoint_final/adapter_model.safetensors"
$trackedAdapter = git ls-tree -r --name-only HEAD -- $v3Adapter
if (-not $trackedAdapter) { throw "adapter_model.safetensors is missing from the result commit" }
git push -u origin results/v3-rtx5060-20260724
git ls-remote --heads origin results/v3-rtx5060-20260724
```

向协调员最终报告：

- 源码 commit、GPU/torch/CUDA；
- selection audit 的精确值和 hashes；
- smoke、68-step train、959/959 evaluation 是否 PASS；
- train/eval 时间；
- V3 与 V2 Random 的 overall/non-recovery/recovery/agent 指标和 paired delta；
- direction label 和 claim boundary；
- result branch、commit、远程验证；
- 若失败，报告原始异常、失败阶段和已保留文件，不得生成假 metrics。

预计 4–7 小时，预留 8–10 小时。正式训练约 15–25 分钟；完整评估约
2–4 小时。
