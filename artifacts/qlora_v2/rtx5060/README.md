# RTX 5060 QLoRA v2 results

This directory is a compact, auditable handoff of the completed **v2** experiment. It is not a v1.1 result bundle.

## Scope and claim boundary

- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB (8151 MiB reported), driver 591.86.
- Source repository commit: `86a5f9fbb97d0b154bd53f2efa4d65f2aaccac89`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Seed: `20260722`.
- Held-out test set: 959 examples, SHA256 `0da63463a65d3b377b3ef3a7e0032a8ffabdc8ab3e439c33850a1eea1ee8fd96`.
- `random_success`, `shortest_success`, and `recovery_coverage` all completed training and the full 959-example evaluation.
- The unadapted `base_model` control also completed the same 959-example evaluation.
- Every evaluation used NF4 4-bit model loading, the most recent 512 prompt tokens with left truncation, greedy decoding, and `max_new_tokens=128`. No CPU or fp16 evaluation fallback was used.
- These v2 results must not be directly mixed with results produced by a different evaluator protocol.
- The experiment measures offline held-out next-tool-call imitation. It does **not** establish end-to-end or executable Agent success.

## Main results

| Control | Examples | JSON valid | Tool accuracy | Full exact match | Task-macro exact match | Recovery exact match | Agent-initiated exact match |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_model | 959 | 17.52% | 6.15% | 0.63% | 1.26% | 7.55% | 7.69% |
| random_success | 959 | 99.79% | 61.73% | 33.06% | 35.42% | 58.49% | 0.00% |
| shortest_success | 959 | 99.27% | 59.96% | 28.15% | 31.87% | 58.49% | 15.38% |
| recovery_coverage | 959 | 99.79% | 57.14% | 25.23% | 28.70% | 56.60% | 15.38% |

The authoritative values and bootstrap confidence intervals are in each `metrics.json` and in `comparison/comparison.json`. No metric was edited or recomputed for this handoff.

## Training and runtime notes

- All three arms used the frozen v2 selection and training contract: one epoch, batch size 1, gradient accumulation 16, and learning rate `1e-4`.
- All three arms completed 68 optimizer steps from 1,690,929 selected SFT tokens.
- No CUDA out-of-memory event occurred.
- A first smoke attempt stopped before training because a Hugging Face model download was incomplete. The model file was checksum-verified and the smoke test was rerun from zero successfully; no failed-smoke output is included here.
- Benign warnings observed: Hugging Face cache symlink support on Windows, and Transformers reporting an empty `label_names` list for the PEFT wrapper.

## Excluded local artifacts

Checkpoints, adapter weights, tokenizer files, model caches, raw datasets, and processed JSONL data are intentionally not uploaded. The retained local adapter weights are:

| Arm | Local path | Bytes | SHA256 |
|---|---|---:|---|
| random_success | `D:\recovery-aware-trajectory-selection-v2\results\qlora_v2\random_success\checkpoint_final\adapter_model.safetensors` | 35,237,104 | `46250a057d85c5ecaaaa1a01a4ff5a90c45df913bbe28dc595a994a90c02729d` |
| shortest_success | `D:\recovery-aware-trajectory-selection-v2\results\qlora_v2\shortest_success\checkpoint_final\adapter_model.safetensors` | 35,237,104 | `fb1c1de781798fd2a3da8581361c5e45c03abbdd5da6a18f67519a4ee4758408` |
| recovery_coverage | `D:\recovery-aware-trajectory-selection-v2\results\qlora_v2\recovery_coverage\checkpoint_final\adapter_model.safetensors` | 35,237,104 | `2148e460f961d7fef07bc8736ded3da635b4e322dc4174824b95bfb566400cd1` |

`UPLOAD_MANIFEST.json` lists the size and SHA256 of every uploaded payload file. It excludes itself because a file cannot contain its own stable cryptographic digest.
