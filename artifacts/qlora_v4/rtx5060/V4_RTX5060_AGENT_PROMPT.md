# Prompt for the RTX 5060 V4 experiment agent

Copy everything below into the Codex agent on the RTX 5060 only after the V4
implementation has been reviewed, all release and preparation values have
been frozen, and the tagged commit has been published. The Clean-SFT adapter
hash is intentionally runtime-generated rather than stored in source config.

---

你是 QLoRA V4 的唯一 GPU 操作员。你的任务是在 NVIDIA GeForce RTX 5060
Laptop GPU（8 GB）上执行 Clean-SFT、chosen-only continued-SFT 和
Clean-SFT→DPO 三个新结果点，并用完全相同的评估器进行比较。

仓库：

```text
https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection
```

必须先完整阅读：

```text
BASELINE_V4_HANDOFF.md
configs/qlora_v4.yaml
```

如果配置中的 release tag 或数据 hash 为空，立即停止并报告静态协议尚未
冻结。不要自己填写 commit 或 hash。`initialization_adapter_sha256: null` 是唯一允许的
运行前空值：正式 Clean-SFT 完成后由 runner 写入 manifest，再由两个二阶段
分支和 package 验证；不得手工改 source config。

## 不可更改的科学设计

1. 所有实验使用同一份 V3 `constrained_recovery` selection，即固定的
   167 条 trajectories；不得重选数据。
2. Standard V3 是已有对照，不重跑。
3. Clean-SFT 把已知失败动作的监督 slots 重分配给 source +
   target-tool matched clean targets；失败动作和 error response 仍保留在
   后续 repair prompt 中。这检验的是固定预算下的 supervision
   reallocation，不是“只删 label”的纯因果效应。
4. Clean-SFT 必须保持 1,088 microbatches、gradient accumulation 16、
   68 optimizer steps，并使用 source + target-tool matched clean
   replacement。
5. 偏好数据固定为 79 对：18 对 agent-initiated、61 对 user-assisted。
6. 二阶段 schedule 固定为 144 slots：72 agent、72 user；batch size 1、
   gradient accumulation 8、18 optimizer steps。
7. continued-SFT 和 DPO 必须从 SHA256 完全相同的 Clean-SFT checkpoint
   分叉，重置 optimizer/scheduler，使用相同 pair IDs/order、
   prompt/chosen exposure、seed、LR 和 optimizer steps。二者都必须使用
   constant LR scheduler、0 warmup 和 SequentialSampler，不得 shuffle。
8. DPO 的 rejected 是已经失败动作的 canonical repeat；不能使用任意生成的
   “坏答案”，不能把 timeout 或其他可重试错误标为 rejected。
9. 现有 959 条 test 已经被查看过，只能作为 exploratory screen。不得声称
   paper-final、end-to-end 或 executable Agent success，也不能承诺正结果。
10. 所有 arms 固定步数；validation 只作 audit，不做 early stopping、
    best-checkpoint 或超参数选择。

## 部署与预检

使用全新目录，不要复用 V1/V2/V3 的工作目录或 output。先从远端默认分支
读取已经发布的冻结 tag，再 detached checkout；不要手工猜 tag：

```powershell
git -c core.autocrlf=false -c core.eol=lf clone https://github.com/sahasurendra929-cmd/recovery-aware-trajectory-selection.git D:\recovery-aware-trajectory-selection-v4-p2
git -C D:\recovery-aware-trajectory-selection-v4-p2 config core.autocrlf false
git -C D:\recovery-aware-trajectory-selection-v4-p2 config core.eol lf
Set-Location D:\recovery-aware-trajectory-selection-v4-p2
$v4Tag = "v4-frozen-20260724-p2"
git fetch origin --tags
git checkout --detach "refs/tags/$v4Tag"
$v4TagMatch = Select-String -Path configs\qlora_v4.yaml -Pattern '^\s*repository_tag:\s*(\S+)\s*$'
if (-not $v4TagMatch) { throw "repository_tag missing from tagged config" }
$v4ConfigTag = $v4TagMatch.Matches[0].Groups[1].Value
if ($v4ConfigTag -ne $v4Tag) { throw "tagged config names a different release" }
$v4Head = git rev-parse HEAD
$v4TaggedHead = git rev-list -n 1 "refs/tags/$v4Tag"
if ($v4Head -ne $v4TaggedHead) { throw "HEAD does not resolve to frozen V4 tag" }
$v4Head
git status --short

New-Item -ItemType Directory -Force data\raw | Out-Null
git clone --no-checkout https://github.com/sierra-research/tau-bench.git data\raw\tau-bench
git -C data\raw\tau-bench config core.autocrlf false
git -C data\raw\tau-bench config core.eol lf
git -C data\raw\tau-bench checkout --detach 59a200c6d575d595120f1cb70fea53cef0632f6b
git -C data\raw\tau-bench rev-parse HEAD
git -C data\raw\tau-bench status --short

$v4ReusablePython = "D:\recovery-aware-trajectory-selection-v4\.venv\Scripts\python.exe"
$v4Python = $null
if (Test-Path $v4ReusablePython) {
  & $v4ReusablePython -c "import sys,torch,transformers,trl,peft,bitsandbytes,datasets,accelerate; assert sys.version_info[:2]==(3,11); assert torch.__version__=='2.7.1+cu128'; assert torch.version.cuda=='12.8'; assert transformers.__version__=='4.52.4'; assert trl.__version__=='0.18.2'; assert peft.__version__=='0.15.2'; assert bitsandbytes.__version__=='0.46.0'; assert datasets.__version__=='3.6.0'; assert accelerate.__version__=='1.7.0'; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
  if ($LASTEXITCODE -eq 0) {
    $v4Python = $v4ReusablePython
    Write-Output "Reusing the exact verified V4 GPU environment; no CUDA/PyTorch download."
  }
}
if (-not $v4Python) {
  py -3.11 -m venv .venv
  $v4Python = ".\.venv\Scripts\python.exe"
  & $v4Python -m pip install --upgrade pip
  & $v4Python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  & $v4Python -m pip install -r requirements-gpu-v4.txt
}
& $v4Python -m pip check
& $v4Python -c "import torch,transformers,trl,peft,bitsandbytes,datasets,accelerate; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),transformers.__version__,trl.__version__,peft.__version__,bitsandbytes.__version__,datasets.__version__,accelerate.__version__); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
& $v4Python -m unittest discover -s tests -v
nvidia-smi
```

PASS 条件：

```text
Python 3.11
torch 2.7.1+cu128
transformers 4.52.4
trl 0.18.2
peft 0.15.2
bitsandbytes 0.46.0
datasets 3.6.0
accelerate 1.7.0
RTX 5060 Laptop GPU / about 8151 MiB
CUDA available / BF16 supported
at least 6 GiB free VRAM
no other Python/CUDA model process
repository and tau-bench worktrees clean
```

电脑接通电源、禁止休眠并保持散热。不要设置
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；之前的 Windows
PyTorch 已明确提示不支持。

## 严格按阶段执行

```powershell
$v4Data = "data\raw\tau-bench\historical_trajectories"

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage prepare
if ($LASTEXITCODE -ne 0) { throw "V4 prepare failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage audit
if ($LASTEXITCODE -ne 0) { throw "V4 audit failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage smoke-clean
if ($LASTEXITCODE -ne 0) { throw "Clean-SFT smoke failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage train-clean-sft
if ($LASTEXITCODE -ne 0) { throw "Clean-SFT training failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage smoke-preference
if ($LASTEXITCODE -ne 0) { throw "Preference smoke failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage train-sft-long
if ($LASTEXITCODE -ne 0) { throw "continued-SFT training failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage train-dpo
if ($LASTEXITCODE -ne 0) { throw "DPO training failed" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage evaluate
if ($LASTEXITCODE -ne 0) { throw "V4 formal evaluation failed or incomplete" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage score
if ($LASTEXITCODE -ne 0) { throw "V4 pair scoring failed or incomplete" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage aggregate
if ($LASTEXITCODE -ne 0) { throw "V4 aggregation rejected the result" }

& $v4Python scripts\run_qlora_v4.py --data-dir $v4Data --stage package
if ($LASTEXITCODE -ne 0) { throw "V4 packaging failed" }
```

不要第一次就运行综合 `overnight/all` 命令。必须先完成正式 Clean-SFT，
再运行以该 checkpoint 为初始化的 preference smoke。每个 gate 至少单独
通过一次后，才可以在以后重复实验中使用自动串行 runner。

## Prepare/Audit 必须确认

```text
same V3 trace fingerprint
Clean-SFT failed_action_labels=0
future_leakage=0
split_overlap=0
Clean-SFT schedule=1088 slots / 68 steps
scheduled completion tokens within 1% of 36599
unique preference pairs=79
pair modes=18 agent / 61 user
preference schedule=144 slots
scheduled modes=72 agent / 72 user
preference smoke schedule=16 distinct pairs / 2 steps
smoke schedule contains the frozen global-longest pair ID
stage-2 gradient accumulation=8
stage-2 optimizer steps=18
all prepared hashes equal configs/qlora_v4.yaml
no validation/test outcomes used for selection or scheduling
held-out test preference-pairs=48 and hash frozen
held-out pair train-task overlap=0
test outcome annotations=959 rows and frozen hash
```

任何一项失败都停止。禁止手工编辑 JSONL、复制一个相近 hash、放松 pair
规则或补足数量。

## Smoke 必须确认

Clean-SFT smoke 和两个二阶段 arm 都要实际运行至少两个 optimizer steps。
二阶段 smoke 必须在正式 Clean-SFT 完成后运行，并加载它的
`checkpoint_final`，不得用临时 smoke adapter 代替。

DPO smoke 必须：

- 在同一个 NF4 base 上加载 internal name=`default` 的 trainable policy
  role 和 internal name=`reference` 的 frozen reference role；两者初始
  内容来自同一个 Clean-SFT adapter；
- 使用 `ref_model=None`、`precompute_ref_log_probs=True`、
  `precompute_ref_batch_size=1` 和 `use_logits_to_keep=True`；
- 包含最长 preference pair；
- policy/reference 初始 log-prob 最大差值 ≤ `1e-4`；
- 初始 DPO loss 在 `0.693147 ± 0.02`；
- loss、reward margin、log-prob 和 grad norm 全部有限；
- reference adapter 前后 hash 不变；
- policy adapter 已改变；
- checkpoint 能重新加载；
- 两个二阶段 Trainer 都使用传入的实际 dataset 构造
  `SequentialSampler`，并记录 `model_accepts_loss_kwargs=false` 和
  `custom_loss_gradient_accumulation_scaled_by_trainer=true`；
- 最长 1,664-token prompt 可以用 NF4 实际生成 128 tokens；
- peak reserved VRAM ≤ 7.5 GiB。

若 smoke OOM：

1. 关闭浏览器、IDE 模型和其他 CUDA 进程；
2. 用 `nvidia-smi` 和 Windows process command line 确认原进程已经退出；
3. 在原 stage 命令末尾加 `--archive-partial-output`，让 runner 保存失败
   attempt 后用全新 Python 进程原参数重跑一次；
4. 确认 batch=1、dynamic padding、gradient checkpointing、NF4、
   reference precompute、`use_logits_to_keep` 均生效；
5. 第二次仍失败则停止 5060 DPO，保留日志并请求迁移到 4090。

禁止通过删最长 pair、缩短 prompt、缩短 completion、改变 pair schedule、
开启 CPU fallback 或只修改 DPO arm 来“解决” OOM。

## 正式结果完整性

Clean-SFT 正式训练必须完成 68 steps。continued-SFT 和 DPO 必须各完成 18
steps，并证明：

```text
initial Clean-SFT adapter SHA256 identical
pair schedule SHA256 identical
prompt/chosen exposure identical
optimizer and scheduler both reset
seed/LR/steps identical
dropout disabled in both stage-2 arms
DPO reference adapter unchanged
```

不要把 continued-SFT 称为 exact compute-matched。DPO 额外读取 rejected 和
reference，必须分别报告 policy/reference tokens、GPU time、wall time 和
peak VRAM。

三个新 checkpoint 都必须在相同 959 条 test 上完整评估：

```text
NF4 4-bit
prompt <=1664
no runtime truncation
greedy decoding
max_new_tokens=128
batch size=1
959/959
no --limit
```

每个 checkpoint 必须由全新的 Python 进程单独评估；一个进程退出并释放
CUDA 后才启动下一个。禁止同时驻留多个 base model 或三个 adapters。
runner 的 held-out pair score 必须调用独立的 `train_preference_v4.py
--mode score`，对每个 adapter 使用 48-pair frozen test file 和全新空输出
目录（该路径启动前必须不存在），并生成 `metrics.json`、
`pair_scores.jsonl`、`score_manifest.json`。
不得跨 adapters 复用 policy log-prob cache。
train、smoke 和 score 的底层调用都必须传
`--expected-pair-file-sha256`；smoke 还必须传
`--expected-longest-pair-id`。这些值只能由 runner 从 frozen config 读取，
操作员不得手输或猜测。

评估断线后先检查：

```powershell
nvidia-smi
Get-CimInstance Win32_Process |
  Where-Object {$_.CommandLine -match "run_qlora_v4|evaluate_tool_actions"}
```

原进程仍在就继续等待，禁止启动第二个 evaluator。进程已退出且没有完整
metrics 时，只能用同一条命令 resume。不能删除 partial predictions、手算
metrics、只评估 recovery subset 或把部分结果冒充完整结果。

## 汇总和解释

必须报告：

- overall tool accuracy / full-call EM；
- outcome-success；
- non-recovery-success；
- recovery-success；
- failed-action imitation rate（越低越好）；
- 48 个 strict pairs 上的 exact failed-call replay rate；
- held-out pair-ranking accuracy；
- chosen-minus-rejected log-prob margin；
- per-token normalized margin（只作长度偏差诊断）；
- canonical tool-call parse validity；
- paired task-cluster uncertainty；
- train/eval tokens、wall time、GPU time、peak VRAM。

三个互斥 evaluation subsets 必须严格等于：

```text
ordinary clean / no prior error = 852
recovery after prior error      = 50
known failed-gold action        = 57
total                           = 959
outcome-success denominator     = 852 + 50 = 902
```

overall/full-call EM 只为和 V2/V3 连续汇报；它会把复制 57 个 failed-gold
labels 算作“正确”，不能拿它作为 V4 的主要成功指标。

重点比较：

```text
Standard V3 vs Clean-SFT
Clean-SFT vs continued-SFT
continued-SFT vs DPO
```

只有当预冻结 directional gate 的四项点估计全部通过，才进入新测试、三种子
确认：recovery-success delta ≥ `+0.04`、48-pair failed-call replay delta
≤ `-2/48`、summed-logp pair-ranking delta ≥ `+3/48`，以及
non-recovery-success delta ≥ `-0.02`。这只是 advance gate，不是论文结论。
若要称为 statistically supported exploratory signal，还要求 recovery 的
paired task-cluster 95% CI 下界大于 0，且 non-recovery CI 下界不低于
`-0.02`。
held-out pair-ranking 必须来自 test split 的独立 high-confidence pairs，
不能拿 79 个训练 pairs 计算后冒充 held-out。只有 pair-ranking 上升不能声称
生成恢复能力提高。
这里 held-out 只表示与训练集分离，不表示 uninspected paper test。

Standard V3 的现有 959 条 generation predictions 可以在 contract 完全一致时
重算 clean/recovery/repeat 指标，但其中没有 token log-probabilities。若没有
额外执行同一 frozen scorer，Standard V3 的 pair-ranking 必须明确写 `N/A`，
不能从 generation exact-match 伪造。

无论结果正向、负向或无差异，都必须原样汇报。当前 959 条结果不能作为最终
论文证据；最终方法需要 fresh uninspected test 和 3 个 training seeds。

## 打包和上传

只有 aggregate 显示全部 arm `valid=true`、959/959 且所有 contract/hash
一致后，才使用 runner 的 package 输出。上传包至少包括：

```text
configs/qlora_v4.yaml
clean/preference manifests and hashes
environment and command audits
three adapters and adapter hashes
training logs/metrics/manifests
full prediction JSONL and contracts
pair-ranking outputs
comparison JSON/CSV/Markdown
UPLOAD_MANIFEST.json
```

禁止上传 raw τ-bench、processed train/validation/test JSONL、`.venv`、
Hugging Face cache、smoke artifacts 或不完整 checkpoint。

结果 branch 建议：

```text
results/v4-p2-rtx5060-20260724
```

确认 `UPLOAD_MANIFEST.json` 的每个文件 hash 后，只 stage package 路径，
不要 `git add .`：

```powershell
$v4ResultsBranch = "results/v4-p2-rtx5060-20260724"
git ls-remote --exit-code --heads origin $v4ResultsBranch | Out-Null
if ($LASTEXITCODE -eq 0) { throw "remote result branch already exists; do not overwrite" }

git switch -c $v4ResultsBranch
git add -- artifacts/qlora_v4/rtx5060
$v4Staged = git diff --cached --name-only
$v4Staged
if (-not $v4Staged) { throw "no packaged V4 artifacts were staged" }
if ($v4Staged -match '(^|/)data/(raw|processed)/|(^|/)\.venv/|(^|/)smoke') {
  throw "forbidden data/cache/smoke artifact was staged"
}

git commit -m "Add audited RTX 5060 QLoRA V4 results"
git push -u origin $v4ResultsBranch
$v4LocalCommit = git rev-parse HEAD
$v4RemoteLine = git ls-remote --heads origin $v4ResultsBranch
if (-not $v4RemoteLine) { throw "remote branch verification failed" }
$v4RemoteCommit = ($v4RemoteLine -split '\s+')[0]
if ($v4LocalCommit -ne $v4RemoteCommit) { throw "remote commit does not match local commit" }
```

如果 adapter 因 `.gitignore` 没有出现在 staged list，停止并报告 release
packaging 配置错误；禁止用 `git add -f` 绕过审计。没有远程 branch 和 commit
一致性验证就不能报告“上传成功”。

## 时间预期

暖缓存预计约 8–18 小时，冷部署约 9–20 小时。训练部分通常小于两小时，
主要时间来自三个 checkpoint 各自完整生成 959 条评估和三次 48-pair
scoring。不要承诺八小时内一定完成；需要预留最多 20 小时，必要时把评估
分两晚完成。
