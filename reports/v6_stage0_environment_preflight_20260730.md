# V6 Stage 0 — Environment and Provenance Preflight

Date: 2026-07-30 (Asia/Shanghai)

Status: **PASS**

## Frozen source identities

- V6 source branch: `codex/v6-causal-recovery-selection`
- V6 source commit: `9f620d438884270f3231101924ff9ef2f6cc5d09`
- tau2 commit: `fc0055dc4e0a316c3f83133267fbd6faaa770992`
- Run root: `/workspace/v6_directional_20260730`
- Existing V5 workspaces and artifacts were left untouched.

## Compute environment

- Host: RunPod pod `x1bpypeogewj0m`
- GPU: 3 × NVIDIA RTX PRO 4500 Blackwell
- VRAM: 32,623 MiB per GPU
- NVIDIA driver: `580.173.02`
- Python: `3.12.3`
- PyTorch: `2.8.0+cu128`
- CUDA runtime reported by PyTorch: `12.8`
- CUDA devices visible: 3
- bfloat16 support: yes
- GPU compute capability: `(12, 0)` on all three devices

## Frozen model revision reachability

The exact revision URLs returned HTTP 206 and were therefore reachable:

- Teacher: `Qwen/Qwen2.5-32B-Instruct-AWQ`
  revision `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c`
- User simulator / strict judge: `Qwen/Qwen2.5-14B-Instruct-AWQ`
  revision `539535859b135b0244c91f3e59816150c8056698`
- Student: `Qwen/Qwen2.5-7B-Instruct`
  revision `a09a35458c702b33eeacc393d103063234e8bc28`

## Storage and access observations

- `/workspace` is mounted and writable.
- The RunPod SSH endpoint changed after migration from port `49941` to
  `40205`; the new endpoint was verified.
- The persistent volume contains historical V5 artifacts but no completed V6
  artifacts. A new isolated V6 run directory was therefore created.
- GitHub read access works from RunPod.
- RunPod has no GitHub write credential. GitHub write access was verified on
  the controlling workstation with a dry-run push. Stage commits will be
  transferred from RunPod when needed and pushed from the workstation.

## Decision

Environment, benchmark identity, model revisions, CUDA availability, GPU
count, and bfloat16 support satisfy the V6 directional-screen preconditions.
Proceed to the code-only V6 unit-test and syntax preflight.

