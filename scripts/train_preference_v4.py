#!/usr/bin/env python3
"""Train either frozen V4 continuation arm on one CUDA GPU.

The two formal arms start from the exact same Clean-SFT adapter and consume
the exact same 144-row preference schedule in sequential order:

* ``continued_sft`` optimizes completion-only causal cross entropy on
  ``prompt + chosen``.
* ``dpo`` optimizes sigmoid DPO on ``prompt, chosen, rejected`` with beta 0.1.

For DPO, one NF4 base holds two byte-identical copies of the Clean-SFT LoRA
adapter.  The ``default`` copy is the policy; the ``reference`` copy is
strictly frozen and is used once to precompute reference log probabilities.
This avoids loading a second 4-bit base model.

The script intentionally fails closed on comparison drift, hidden truncation,
non-finite loss, mutable reference weights, or an incomplete initialization
checkpoint.  It never reads a held-out test file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import statistics
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "qlora_v4_preference_continuation"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEED = 20260722
MAX_PROMPT_TOKENS = 1664
MAX_COMPLETION_TOKENS = 384
MAX_SEQUENCE_TOKENS = 2048
FORMAL_UNIQUE_PAIRS = 79
FORMAL_SCHEDULE_ROWS = 144
FORMAL_MAX_STEPS = 18
FORMAL_BATCH_SIZE = 1
FORMAL_GRAD_ACCUM = 8
FORMAL_LEARNING_RATE = 1.0e-5
SMOKE_MAX_STEPS = 2
SMOKE_SCHEDULE_ROWS = 16
SMOKE_MAX_RESERVED_BYTES = 8_053_063_680  # 7.5 GiB
LR_SCHEDULER = "constant"
WARMUP_STEPS = 0
DPO_BETA = 0.1
DPO_LOSS_TYPE = "sigmoid"
DPO_LABEL_SMOOTHING = 0.0
POLICY_ADAPTER = "default"
REFERENCE_ADAPTER = "reference"
REQUIRED_CHECKPOINT_FILES = ("adapter_config.json", "adapter_model.safetensors")
FROZEN_TAG = "v4-frozen-20260724-p3"
ROOT = Path(__file__).resolve().parents[1]


def resolve_train_sampler_dataset(
    trainer: Any,
    train_dataset: Any | None = None,
) -> Any:
    """Return the dataset supplied by the current Transformers dataloader.

    Transformers 4.52.4 calls ``_get_train_sampler(train_dataset)`` after its
    dataloader path has resolved the effective dataset.  Falling back to the
    trainer attribute keeps compatibility with direct no-argument calls while
    ensuring that the sampler follows the actual dataset passed by the pinned
    library.
    """
    return trainer.train_dataset if train_dataset is None else train_dataset


def initialize_cuda_peak_tracking(torch: Any) -> int:
    """Initialize the CUDA allocator before resetting peak counters."""
    device_index = 0
    torch.cuda.init()
    torch.cuda.set_device(device_index)
    allocator_probe = torch.empty(1, device=f"cuda:{device_index}")
    torch.cuda.synchronize(device_index)
    del allocator_probe
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    return device_index


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frozen_source_commit() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tagged = subprocess.run(
        ["git", "rev-parse", f"refs/tags/{FROZEN_TAG}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != tagged:
        raise RuntimeError(f"HEAD {head} differs from {FROZEN_TAG}={tagged}")
    return head


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"refusing to serialize non-finite float: {value!r}")
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(json_safe(row)) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"expected a JSON object in {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise RuntimeError(f"empty JSONL file: {path}")
    return rows


def require_empty_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"output path exists but is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"output directory is non-empty: {path}. Refusing to mix formal and "
            "partial artifacts; use a new directory."
        )


def checkpoint_fingerprint(checkpoint: Path) -> tuple[str, dict[str, str]]:
    file_hashes: dict[str, str] = {}
    digest = hashlib.sha256()
    for filename in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"incomplete Clean-SFT adapter checkpoint: {path}")
        file_hash = sha256_file(path)
        file_hashes[filename] = file_hash
        digest.update(filename.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest(), file_hashes


def schedule_content_key(row: dict[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "prompt": row.get("prompt"),
                "chosen": row.get("chosen"),
                "rejected": row.get("rejected"),
            }
        )
    )


def schedule_display_id(row: dict[str, Any], content_key: str) -> str:
    for field in ("pair_id", "preference_id", "schedule_id", "example_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return f"derived:{content_key[:20]}"


def validate_schedule_shape(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_unique_pairs: int | None,
) -> dict[str, Any]:
    content_keys: list[str] = []
    display_ids: list[str] = []
    for index, row in enumerate(rows):
        for field in ("prompt", "chosen", "rejected"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"schedule row {index} has invalid {field!r}")
        if row["chosen"] == row["rejected"]:
            raise RuntimeError(f"schedule row {index} has identical chosen and rejected completions")
        key = schedule_content_key(row)
        content_keys.append(key)
        display_ids.append(schedule_display_id(row, key))

    if len(rows) != expected_rows:
        raise RuntimeError(f"schedule must contain exactly {expected_rows} rows; found {len(rows)}")
    unique_count = len(set(content_keys))
    if expected_unique_pairs is not None and unique_count != expected_unique_pairs:
        raise RuntimeError(
            f"schedule must expose exactly {expected_unique_pairs} unique preference pairs; "
            f"found {unique_count}"
        )
    return {
        "row_count": len(rows),
        "unique_pair_count": unique_count,
        "ordered_content_sha256": sha256_text("\n".join(content_keys) + "\n"),
        "ordered_display_ids_sha256": sha256_text("\n".join(display_ids) + "\n"),
        "first_display_ids": display_ids[:5],
        "last_display_ids": display_ids[-5:],
    }


def validate_split_provenance(
    rows: list[dict[str, Any]],
    *,
    expected_split: str,
) -> dict[str, Any]:
    task_keys: set[str] = set()
    for index, row in enumerate(rows):
        split = row.get("split")
        if split != expected_split:
            raise RuntimeError(
                f"pair provenance mismatch at row {index}: split={split!r}, "
                f"expected {expected_split!r}"
            )
        task_key = row.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            raise RuntimeError(f"pair provenance row {index} lacks a nonempty task_key")
        task_keys.add(task_key)
    return {
        "expected_split": expected_split,
        "all_rows_match_expected_split": True,
        "row_count": len(rows),
        "unique_task_keys": len(task_keys),
        "ordered_task_keys_sha256": sha256_text(
            "\n".join(str(row["task_key"]) for row in rows) + "\n"
        ),
    }


def validate_preference_strata(
    rows: list[dict[str, Any]],
    *,
    formal_train: bool = False,
    smoke_train: bool = False,
) -> dict[str, Any]:
    allowed_modes = {"agent_initiated", "user_assisted"}
    allowed_errors = {"generic_error", "not_found"}
    mode_counts = {value: 0 for value in sorted(allowed_modes)}
    error_counts = {value: 0 for value in sorted(allowed_errors)}
    unique_pair_modes: dict[str, str] = {}
    for index, row in enumerate(rows):
        mode = row.get("recovery_mode")
        error_type = row.get("error_type")
        if mode not in allowed_modes:
            raise RuntimeError(f"invalid recovery_mode in pair row {index}: {mode!r}")
        if error_type not in allowed_errors:
            raise RuntimeError(f"invalid error_type in pair row {index}: {error_type!r}")
        mode_counts[mode] += 1
        error_counts[error_type] += 1
        content_key = schedule_content_key(row)
        previous_mode = unique_pair_modes.setdefault(content_key, mode)
        if previous_mode != mode:
            raise RuntimeError(f"same preference pair has conflicting recovery modes: {content_key}")

    unique_mode_counts = {value: 0 for value in sorted(allowed_modes)}
    for mode in unique_pair_modes.values():
        unique_mode_counts[mode] += 1
    if formal_train:
        if mode_counts != {"agent_initiated": 72, "user_assisted": 72}:
            raise RuntimeError(f"formal preference schedule mode drift: {mode_counts}")
        if unique_mode_counts != {"agent_initiated": 18, "user_assisted": 61}:
            raise RuntimeError(f"formal unique preference-pair mode drift: {unique_mode_counts}")
    if smoke_train:
        if any(mode_counts[value] <= 0 for value in allowed_modes):
            raise RuntimeError(f"smoke schedule does not cover both recovery modes: {mode_counts}")
        if any(error_counts[value] <= 0 for value in allowed_errors):
            raise RuntimeError(f"smoke schedule does not cover both error types: {error_counts}")
    return {
        "mode_counts": mode_counts,
        "error_type_counts": error_counts,
        "unique_pair_mode_counts": unique_mode_counts,
        "allowed_modes_only": True,
        "allowed_error_types_only": True,
        "formal_72_72_mode_balance": mode_counts
        == {"agent_initiated": 72, "user_assisted": 72}
        if formal_train
        else None,
        "smoke_covers_both_modes_and_error_types": (
            all(mode_counts[value] > 0 for value in allowed_modes)
            and all(error_counts[value] > 0 for value in allowed_errors)
        )
        if smoke_train
        else None,
    }


def require_expected_file_hash(path: Path, expected: str) -> str:
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise RuntimeError(f"expected SHA256 must be 64 lowercase hex characters; got {expected!r}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"frozen input hash mismatch for {path}: {observed} != {expected}")
    return observed


def validate_frozen_formal_args(args: argparse.Namespace) -> None:
    expected = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_seq_len": MAX_SEQUENCE_TOKENS,
        "max_steps": FORMAL_MAX_STEPS,
        "batch_size": FORMAL_BATCH_SIZE,
        "grad_accum": FORMAL_GRAD_ACCUM,
        "learning_rate": FORMAL_LEARNING_RATE,
        "beta": DPO_BETA,
    }
    for field, frozen in expected.items():
        observed = getattr(args, field)
        if observed != frozen:
            raise RuntimeError(f"formal V4 drift for {field}: {observed!r} != {frozen!r}")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one fixed V4 continuation arm from the same Clean-SFT adapter."
    )
    parser.add_argument(
        "--mode",
        choices=("train", "score"),
        default="train",
        help="Train one continuation arm, or score a strict validation/test pair file in a separate process.",
    )
    parser.add_argument("--arm", choices=("continued_sft", "dpo"))
    parser.add_argument("--pair-file", type=Path)
    parser.add_argument(
        "--expected-pair-file-sha256",
        help="Frozen SHA256 for pair-file; required so a train/test file cannot be silently substituted.",
    )
    parser.add_argument("--clean-sft-adapter", type=Path)
    parser.add_argument(
        "--score-adapter",
        type=Path,
        help="Adapter checkpoint to load in --mode score (Clean-SFT, continued-SFT, or DPO).",
    )
    parser.add_argument(
        "--score-split",
        choices=("validation", "test"),
        help="Required provenance label for the strict pair file used in --mode score.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    parser.add_argument("--max-completion-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument("--max-steps", type=int, default=FORMAL_MAX_STEPS)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=FORMAL_GRAD_ACCUM)
    parser.add_argument("--learning-rate", type=float, default=FORMAL_LEARNING_RATE)
    parser.add_argument("--beta", type=float, default=DPO_BETA)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the frozen base/tokenizer revision to already be in the HF cache.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Require a dedicated 16-row smoke schedule and run two optimizer "
            "steps; output is never a formal result."
        ),
    )
    parser.add_argument(
        "--expected-longest-pair-id",
        help="Required with --smoke-test; must identify the global longest pair included in its 16-row schedule.",
    )
    parser.add_argument(
        "--static-self-test",
        action="store_true",
        help="Run dependency-free contract helper checks and exit.",
    )
    args = parser.parse_args()
    if not args.static_self_test:
        if args.mode == "train":
            required = (
                ("--arm", args.arm),
                ("--pair-file", args.pair_file),
                ("--expected-pair-file-sha256", args.expected_pair_file_sha256),
                ("--clean-sft-adapter", args.clean_sft_adapter),
                ("--output-dir", args.output_dir),
            )
        else:
            required = (
                ("--pair-file", args.pair_file),
                ("--expected-pair-file-sha256", args.expected_pair_file_sha256),
                ("--score-adapter", args.score_adapter),
                ("--score-split", args.score_split),
                ("--output-dir", args.output_dir),
            )
            if args.arm is not None or args.clean_sft_adapter is not None:
                parser.error("--mode score forbids --arm and --clean-sft-adapter; use --score-adapter")
            if args.smoke_test:
                parser.error("--mode score forbids --smoke-test; pass a deliberately small pair file if needed")
        if args.mode == "train" and args.smoke_test and not args.expected_longest_pair_id:
            parser.error("--smoke-test requires --expected-longest-pair-id")
        if args.mode == "train" and not args.smoke_test and args.expected_longest_pair_id:
            parser.error("--expected-longest-pair-id is valid only with --smoke-test")
        missing = [flag for flag, value in required if value is None]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
    return args


def run_static_self_test() -> None:
    row_a = {
        "pair_id": "a",
        "prompt": "p",
        "chosen": "c",
        "rejected": "r",
        "split": "train",
        "task_key": "task-a",
        "recovery_mode": "agent_initiated",
        "error_type": "not_found",
    }
    row_b = {
        "pair_id": "b",
        "prompt": "p2",
        "chosen": "c2",
        "rejected": "r2",
        "split": "train",
        "task_key": "task-b",
        "recovery_mode": "user_assisted",
        "error_type": "generic_error",
    }
    audit = validate_schedule_shape([row_a, row_b, row_a], expected_rows=3, expected_unique_pairs=2)
    if audit["row_count"] != 3 or audit["unique_pair_count"] != 2:
        raise RuntimeError("schedule helper self-test failed")
    if schedule_content_key(row_a) != schedule_content_key(dict(row_a)):
        raise RuntimeError("canonical schedule hashing self-test failed")
    split_audit = validate_split_provenance([row_a, row_b], expected_split="train")
    if not split_audit["all_rows_match_expected_split"]:
        raise RuntimeError("split provenance self-test failed")
    strata_audit = validate_preference_strata([row_a, row_b], smoke_train=True)
    if not strata_audit["smoke_covers_both_modes_and_error_types"]:
        raise RuntimeError("preference strata self-test failed")
    comparison = comparison_contract_id(
        init_fingerprint="a" * 64,
        schedule_file_sha256="b" * 64,
        schedule_order_sha256="c" * 64,
        max_steps=FORMAL_MAX_STEPS,
        batch_size=FORMAL_BATCH_SIZE,
        grad_accum=FORMAL_GRAD_ACCUM,
        learning_rate=FORMAL_LEARNING_RATE,
    )
    if len(comparison) != 64:
        raise RuntimeError("comparison-contract hashing self-test failed")
    print(json.dumps({"static_self_test": "passed", "protocol": PROTOCOL}, indent=2))


def comparison_contract_id(
    *,
    init_fingerprint: str,
    schedule_file_sha256: str,
    schedule_order_sha256: str,
    max_steps: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
) -> str:
    # Deliberately excludes the arm/objective: equal IDs prove that the two
    # arms shared all frozen comparison inputs and exposure controls.
    contract = {
        "protocol": PROTOCOL,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "init_checkpoint_fingerprint": init_fingerprint,
        "train_schedule_sha256": schedule_file_sha256,
        "ordered_schedule_sha256": schedule_order_sha256,
        "seed": SEED,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "lr_scheduler_type": LR_SCHEDULER,
        "warmup_steps": WARMUP_STEPS,
        "sampler": "SequentialSampler",
        "dropout": 0.0,
    }
    return sha256_text(canonical_json(contract))


def set_deterministic_runtime(torch: Any, seed: int, set_seed: Any) -> None:
    set_seed(seed)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must equal ':4096:8' before the first CUDA call"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_and_audit_dropout(model: Any, torch: Any) -> dict[str, Any]:
    changed = 0
    dropout_modules = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_modules += 1
            if module.p != 0.0:
                changed += 1
            module.p = 0.0
    for config in getattr(model, "peft_config", {}).values():
        if hasattr(config, "lora_dropout"):
            config.lora_dropout = 0.0
    config = getattr(model, "config", None)
    if config is not None:
        for field in (
            "attention_dropout",
            "hidden_dropout",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "classifier_dropout",
        ):
            if hasattr(config, field):
                setattr(config, field, 0.0)
    nonzero = [
        (name, module.p)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and module.p != 0.0
    ]
    if nonzero:
        raise RuntimeError(f"dropout-disable audit failed: {nonzero[:5]}")
    return {
        "dropout_modules": dropout_modules,
        "modules_changed_to_zero": changed,
        "nonzero_dropout_modules": 0,
        "lora_dropout_config": 0.0,
    }


def adapter_parameter_items(model: Any, adapter_name: str) -> list[tuple[str, Any]]:
    marker = f".{adapter_name}."
    items = [(name, parameter) for name, parameter in model.named_parameters() if marker in name]
    if not items:
        raise RuntimeError(f"no parameters found for adapter {adapter_name!r}")
    return sorted(items, key=lambda item: item[0])


def adapter_state_sha256(model: Any, adapter_name: str, torch: Any) -> str:
    marker = f".{adapter_name}."
    digest = hashlib.sha256()
    for name, parameter in adapter_parameter_items(model, adapter_name):
        normalized_name = name.replace(marker, ".<adapter>.")
        tensor = parameter.detach().cpu().contiguous()
        digest.update(normalized_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def set_reference_frozen(model: Any) -> int:
    count = 0
    for _, parameter in adapter_parameter_items(model, REFERENCE_ADAPTER):
        parameter.requires_grad_(False)
        count += parameter.numel()
    return count


def trainable_parameter_audit(model: Any) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable policy parameters")
    wrong = [name for name, _ in trainable if f".{POLICY_ADAPTER}." not in name]
    if wrong:
        raise RuntimeError(f"non-policy parameters are trainable: {wrong[:10]}")
    return {
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "all_trainable_parameters_belong_to_policy_adapter": True,
    }


def optimizer_reference_overlap(trainer: Any, model: Any) -> int:
    if trainer.optimizer is None:
        raise RuntimeError("trainer did not construct an optimizer")
    optimizer_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group.get("params", ())
    }
    reference_ids = {id(parameter) for _, parameter in adapter_parameter_items(model, REFERENCE_ADAPTER)}
    return len(optimizer_ids & reference_ids)


def tokenize_and_audit_schedule(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    max_seq_len: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    eos_id = tokenizer.eos_token_id
    eos_text = tokenizer.eos_token
    if eos_id is None or not isinstance(eos_text, str) or not eos_text:
        raise RuntimeError("the frozen tokenizer has no usable EOS token")

    for index, row in enumerate(rows):
        # These settings exactly match TRL 0.18.2 DPOTrainer.tokenize_row for
        # decoder-only models.  Appending EOS unconditionally is intentional.
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        if eos_text in row["chosen"] or eos_text in row["rejected"]:
            raise RuntimeError(
                f"schedule row {index} completion already contains EOS text; "
                "the runner appends exactly one EOS"
            )
        chosen_ids = tokenizer(row["chosen"], add_special_tokens=False)["input_ids"] + [eos_id]
        rejected_ids = tokenizer(row["rejected"], add_special_tokens=False)["input_ids"] + [eos_id]
        chosen_concat_ids = tokenizer(
            row["chosen"] + eos_text,
            add_special_tokens=False,
        )["input_ids"]
        rejected_concat_ids = tokenizer(
            row["rejected"] + eos_text,
            add_special_tokens=False,
        )["input_ids"]
        if chosen_ids != chosen_concat_ids or rejected_ids != rejected_concat_ids:
            raise RuntimeError(
                f"EOS token-boundary drift in schedule row {index}; "
                "TRL append semantics differ from prepared text+EOS counts"
            )
        expected_counts = {
            "prompt_tokens": len(prompt_ids),
            "chosen_tokens": len(chosen_ids),
            "rejected_tokens": len(rejected_ids),
        }
        for field, observed_count in expected_counts.items():
            prepared_count = row.get(field)
            if prepared_count != observed_count:
                raise RuntimeError(
                    f"tokenizer/count drift in row {index} for {field}: "
                    f"prepared={prepared_count!r}, observed={observed_count}"
                )
        if not prompt_ids or len(chosen_ids) <= 1 or len(rejected_ids) <= 1:
            raise RuntimeError(f"empty tokenized field in schedule row {index}")
        if len(prompt_ids) > max_prompt_tokens:
            raise RuntimeError(
                f"prompt would be truncated in row {index}: {len(prompt_ids)} > {max_prompt_tokens}"
            )
        if len(chosen_ids) > max_completion_tokens:
            raise RuntimeError(
                f"chosen completion would be truncated in row {index}: "
                f"{len(chosen_ids)} > {max_completion_tokens}"
            )
        if len(rejected_ids) > max_completion_tokens:
            raise RuntimeError(
                f"rejected completion would be truncated in row {index}: "
                f"{len(rejected_ids)} > {max_completion_tokens}"
            )
        if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_seq_len:
            raise RuntimeError(f"full preference sequence would be truncated in row {index}")
        encoded.append(
            {
                "prompt_input_ids": prompt_ids,
                "chosen_input_ids": chosen_ids,
                "rejected_input_ids": rejected_ids,
            }
        )
        prompt_lengths.append(len(prompt_ids))
        chosen_lengths.append(len(chosen_ids))
        rejected_lengths.append(len(rejected_ids))

    stats = {
        "scheduled_rows": len(rows),
        "chosen_exposures": len(rows),
        "prompt_tokens": sum(prompt_lengths),
        "chosen_completion_tokens": sum(chosen_lengths),
        "rejected_completion_tokens": sum(rejected_lengths),
        "chosen_sequence_tokens": sum(p + c for p, c in zip(prompt_lengths, chosen_lengths)),
        "rejected_sequence_tokens": sum(p + r for p, r in zip(prompt_lengths, rejected_lengths)),
        "max_prompt_tokens_observed": max(prompt_lengths),
        "max_chosen_completion_tokens_observed": max(chosen_lengths),
        "max_rejected_completion_tokens_observed": max(rejected_lengths),
        "runtime_truncation_count": 0,
    }
    return encoded, stats


def finite_loss_audit(log_history: Iterable[dict[str, Any]], result_metrics: dict[str, Any]) -> dict[str, Any]:
    losses: list[float] = []
    grad_norms: list[float] = []
    numeric_values_checked = 0
    for record in log_history:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(f"non-finite logged metric {key}: {value!r}")
                numeric_values_checked += 1
            if (
                "loss" in key.lower()
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                losses.append(numeric)
            if (
                key == "grad_norm"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                grad_norms.append(numeric)
    for key, value in result_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"non-finite result metric {key}: {value!r}")
            numeric_values_checked += 1
        if (
            "loss" in key.lower()
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            losses.append(numeric)
    if not losses:
        raise RuntimeError("trainer produced no auditable finite loss values")
    if not grad_norms:
        raise RuntimeError("trainer produced no auditable grad_norm values")
    return {
        "finite": True,
        "all_numeric_logged_metrics_finite": True,
        "numeric_values_checked": numeric_values_checked,
        "value_count": len(losses),
        "minimum": min(losses),
        "maximum": max(losses),
        "final_train_loss": float(result_metrics["train_loss"]),
        "grad_norm_finite": True,
        "grad_norm_count": len(grad_norms),
        "grad_norm_minimum": min(grad_norms),
        "grad_norm_maximum": max(grad_norms),
    }


def reference_logprob_audit(dataset: Any) -> dict[str, Any]:
    columns = set(dataset.column_names)
    required = {"ref_chosen_logps", "ref_rejected_logps"}
    if not required.issubset(columns):
        raise RuntimeError(f"DPO reference precompute columns missing: {sorted(required - columns)}")
    chosen = [float(value) for value in dataset["ref_chosen_logps"]]
    rejected = [float(value) for value in dataset["ref_rejected_logps"]]
    if not chosen or len(chosen) != len(rejected):
        raise RuntimeError("invalid precomputed reference log-probability lengths")
    if not all(math.isfinite(value) for value in chosen + rejected):
        raise RuntimeError("non-finite precomputed reference log probability")
    digest = hashlib.sha256()
    for chosen_value, rejected_value in zip(chosen, rejected):
        digest.update(struct.pack("<ff", chosen_value, rejected_value))
    return {
        "precomputed": True,
        "row_count": len(chosen),
        "sha256_float32_pairs": digest.hexdigest(),
        "chosen_min": min(chosen),
        "chosen_max": max(chosen),
        "rejected_min": min(rejected),
        "rejected_max": max(rejected),
    }


def environment_audit(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "datasets": package_version("datasets"),
        "accelerate": package_version("accelerate"),
        "trl": package_version("trl"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def run_score_mode(
    args: argparse.Namespace,
    *,
    torch: Any,
    AutoModelForCausalLM: Any,
    AutoTokenizer: Any,
    BitsAndBytesConfig: Any,
    PeftModel: Any,
    set_seed: Any,
) -> None:
    """Score strict preference pairs without training or modifying an adapter."""
    frozen = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_seq_len": MAX_SEQUENCE_TOKENS,
    }
    for field, expected in frozen.items():
        observed = getattr(args, field)
        if observed != expected:
            raise RuntimeError(f"V4 score drift for {field}: {observed!r} != {expected!r}")

    require_empty_directory(args.output_dir)
    adapter_fingerprint, adapter_file_hashes = checkpoint_fingerprint(args.score_adapter)
    rows = read_jsonl(args.pair_file)
    pair_audit = validate_schedule_shape(
        rows,
        expected_rows=len(rows),
        # A strict validation/test file must not repeat a pair.
        expected_unique_pairs=len(rows),
    )
    split_audit = validate_split_provenance(rows, expected_split=args.score_split)
    strata_audit = validate_preference_strata(rows)
    pair_file_hash = require_expected_file_hash(
        args.pair_file,
        args.expected_pair_file_sha256,
    )

    set_deterministic_runtime(torch, args.seed, set_seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded_rows, token_audit = tokenize_and_audit_schedule(
        rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base,
        args.score_adapter,
        adapter_name=POLICY_ADAPTER,
        is_trainable=False,
    )
    dropout_audit = disable_and_audit_dropout(model, torch)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("score mode unexpectedly left trainable parameters")
    model.eval()

    def completion_logprob(prompt_ids: list[int], completion_ids: list[int]) -> tuple[float, float]:
        input_ids = prompt_ids + completion_ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones_like(input_tensor)
        # Include the final prompt position (which predicts completion token 0)
        # and exclude the final unused position after the forward pass.
        logits_to_keep = len(completion_ids) + 1
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
            logits = outputs.logits[:, :-1, :].float()
            targets = torch.tensor([completion_ids], dtype=torch.long, device=logits.device)
            if logits.shape[:2] != targets.shape:
                raise RuntimeError(
                    f"score tail-logit alignment failure: {tuple(logits.shape[:2])} "
                    f"!= {tuple(targets.shape)}"
                )
            target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            token_logps = target_logits - torch.logsumexp(logits, dim=-1)
            total = float(token_logps.sum(dtype=torch.float64).item())
        if not math.isfinite(total):
            raise RuntimeError(f"non-finite completion log probability: {total!r}")
        return total, total / len(completion_ids)

    # The orchestrator may pre-create the run directory for logs.  The
    # fail-closed check above already rejected every nonempty directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started_at = utc_now()
    start_time = time.monotonic()
    score_rows: list[dict[str, Any]] = []
    summed_margins: list[float] = []
    normalized_margins: list[float] = []
    strict_correct = 0
    normalized_correct = 0
    ties = 0

    for index, (row, encoded) in enumerate(zip(rows, encoded_rows)):
        chosen_sum, chosen_mean = completion_logprob(
            encoded["prompt_input_ids"],
            encoded["chosen_input_ids"],
        )
        rejected_sum, rejected_mean = completion_logprob(
            encoded["prompt_input_ids"],
            encoded["rejected_input_ids"],
        )
        summed_margin = chosen_sum - rejected_sum
        normalized_margin = chosen_mean - rejected_mean
        if not math.isfinite(summed_margin) or not math.isfinite(normalized_margin):
            raise RuntimeError(f"non-finite pair margin at row {index}")
        is_correct = summed_margin > 0.0
        is_normalized_correct = normalized_margin > 0.0
        strict_correct += int(is_correct)
        normalized_correct += int(is_normalized_correct)
        ties += int(summed_margin == 0.0)
        summed_margins.append(summed_margin)
        normalized_margins.append(normalized_margin)
        content_key = schedule_content_key(row)
        score_rows.append(
            {
                "pair_index": index,
                "pair_id": schedule_display_id(row, content_key),
                "pair_content_sha256": content_key,
                "split": args.score_split,
                "task_key": row.get("task_key"),
                # Compatibility alias for older result tooling.
                "task_id": row.get("task_key"),
                "chosen_tokens_including_eos": len(encoded["chosen_input_ids"]),
                "rejected_tokens_including_eos": len(encoded["rejected_input_ids"]),
                "chosen_summed_logp_including_eos": chosen_sum,
                "rejected_summed_logp_including_eos": rejected_sum,
                "summed_logp_margin_chosen_minus_rejected": summed_margin,
                "chosen_per_token_logp_including_eos": chosen_mean,
                "rejected_per_token_logp_including_eos": rejected_mean,
                "per_token_normalized_margin_chosen_minus_rejected": normalized_margin,
                "summed_logp_correct": is_correct,
                "per_token_normalized_correct": is_normalized_correct,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f"scored {index + 1}/{len(rows)} strict {args.score_split} pairs", flush=True)

    duration_seconds = time.monotonic() - start_time
    completed_at = utc_now()
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    pair_count = len(score_rows)
    metrics = {
        "protocol": f"{PROTOCOL}_pair_scoring",
        "split": args.score_split,
        "pair_count": pair_count,
        "pair_accuracy_summed_logp": strict_correct / pair_count,
        "summed_logp_correct_count": strict_correct,
        "summed_logp_tie_count": ties,
        "mean_summed_logp_margin": statistics.fmean(summed_margins),
        "median_summed_logp_margin": statistics.median(summed_margins),
        "per_token_normalized_pair_accuracy": normalized_correct / pair_count,
        "per_token_normalized_correct_count": normalized_correct,
        "mean_per_token_normalized_margin": statistics.fmean(normalized_margins),
        "median_per_token_normalized_margin": statistics.median(normalized_margins),
        "completion_eos_included": True,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "duration_seconds": duration_seconds,
    }
    pair_scores_path = args.output_dir / "pair_scores.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    write_jsonl(pair_scores_path, score_rows)
    write_json(metrics_path, metrics)
    pair_scores_sha256 = sha256_file(pair_scores_path)
    metrics_sha256 = sha256_file(metrics_path)
    write_json(
        args.output_dir / "score_manifest.json",
        {
            "protocol": f"{PROTOCOL}_pair_scoring",
            "source_tag": FROZEN_TAG,
            "source_commit": args.source_commit,
            "mode": "score",
            "split": args.score_split,
            "training_performed": False,
            "limited": False,
            "complete": True,
            "expected_pairs": pair_count,
            "completed_pairs": pair_count,
            "pair_count": pair_count,
            "preference_pairs_sha256": pair_file_hash,
            "pair_scores_sha256": pair_scores_sha256,
            # Kept at the top level because the frozen aggregator validates
            # this provenance field before it reads the nested output table.
            "metrics_sha256": metrics_sha256,
            "checkpoint_fingerprint": adapter_fingerprint,
            "model": args.model,
            "model_revision": args.model_revision,
            "adapter": {
                "path": str(args.score_adapter),
                "checkpoint_fingerprint": adapter_fingerprint,
                "file_sha256": adapter_file_hashes,
            },
            "data": {
                "strict_pair_file": str(args.pair_file),
                "strict_pair_file_sha256": pair_file_hash,
                "expected_strict_pair_file_sha256": args.expected_pair_file_sha256,
                "strict_pair_file_hash_matches_expected": True,
                "pair_audit": pair_audit,
                "split_audit": split_audit,
                "strata_audit": strata_audit,
                "token_audit": token_audit,
            },
            "scoring_contract": {
                "completion_eos_included": True,
                "primary_pair_decision": "chosen_summed_logp > rejected_summed_logp",
                "summed_margin": "chosen_summed_logp - rejected_summed_logp",
                "secondary_length_diagnostic": (
                    "chosen_summed_logp/chosen_tokens - "
                    "rejected_summed_logp/rejected_tokens"
                ),
                "runtime_truncation_count": 0,
                "dropout": 0.0,
                "quantization": "NF4 double-quantized base",
                "bf16": True,
            },
            "dropout_audit": dropout_audit,
            "runtime": {
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration_seconds,
                "peak_cuda_memory_allocated_bytes": peak_allocated,
                "peak_cuda_memory_reserved_bytes": peak_reserved,
            },
            "environment": environment_audit(torch),
            "outputs": {
                "metrics": str(metrics_path),
                "metrics_sha256": metrics_sha256,
                "pair_scores": str(pair_scores_path),
                "pair_scores_sha256": pair_scores_sha256,
                "pair_scores_rows": pair_count,
            },
        },
    )
    print(json.dumps({"status": "complete", **metrics}, indent=2))


def main() -> None:
    args = parse_args()
    if args.static_self_test:
        run_static_self_test()
        return

    # Heavy dependencies are intentionally imported only after argument and
    # static-contract parsing, so py_compile/self-test work on the Mac.
    # CUBLAS requires this value before the first CUDA context/API call.
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in (None, ":4096:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or equal ':4096:8' "
            "before importing PyTorch"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.nn.functional as functional
    import transformers
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from torch.utils.data import SequentialSampler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"V4 requires Python 3.11; found {sys.version.split()[0]}")
    frozen_packages = {
        "transformers": "4.52.4",
        "peft": "0.15.2",
        "bitsandbytes": "0.46.0",
        "datasets": "3.6.0",
        "accelerate": "1.7.0",
        "trl": "0.18.2",
    }
    for package, expected_version in frozen_packages.items():
        observed_version = package_version(package)
        if observed_version != expected_version:
            raise RuntimeError(
                f"V4 requires {package}=={expected_version}; found {observed_version}"
            )
    if not str(torch.__version__).startswith("2.7.1"):
        raise RuntimeError(f"V4 requires torch 2.7.1+cu128; found {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"V4 requires CUDA runtime 12.8; found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("V4 preference training requires one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("V4 freezes bfloat16 compute, but this GPU/PyTorch build lacks BF16 support")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"V4 runner is single-GPU only; CUDA exposes {torch.cuda.device_count()} devices. "
            "Set CUDA_VISIBLE_DEVICES to exactly one GPU."
        )
    args.source_commit = frozen_source_commit()

    if args.mode == "score":
        run_score_mode(
            args,
            torch=torch,
            AutoModelForCausalLM=AutoModelForCausalLM,
            AutoTokenizer=AutoTokenizer,
            BitsAndBytesConfig=BitsAndBytesConfig,
            PeftModel=PeftModel,
            set_seed=set_seed,
        )
        return

    # Smoke changes only the dedicated input schedule and effective step
    # count.  It must exercise every other formal model/hyperparameter value.
    validate_frozen_formal_args(args)
    require_empty_directory(args.output_dir)
    init_fingerprint, init_file_hashes = checkpoint_fingerprint(args.clean_sft_adapter)
    rows = read_jsonl(args.pair_file)

    if args.smoke_test:
        if len(rows) != SMOKE_SCHEDULE_ROWS:
            raise RuntimeError(
                f"smoke test requires a dedicated {SMOKE_SCHEDULE_ROWS}-row schedule "
                "containing the longest pair and representative modes; "
                f"found {len(rows)} rows"
            )
        effective_max_steps = SMOKE_MAX_STEPS
        effective_grad_accum = FORMAL_GRAD_ACCUM
        expected_unique_pairs = SMOKE_SCHEDULE_ROWS
    else:
        effective_max_steps = args.max_steps
        effective_grad_accum = args.grad_accum
        expected_unique_pairs = FORMAL_UNIQUE_PAIRS
    expected_rows = effective_max_steps * args.batch_size * effective_grad_accum
    schedule_audit = validate_schedule_shape(
        rows,
        expected_rows=expected_rows,
        expected_unique_pairs=expected_unique_pairs,
    )
    split_audit = validate_split_provenance(rows, expected_split="train")
    strata_audit = validate_preference_strata(
        rows,
        formal_train=not args.smoke_test,
        smoke_train=args.smoke_test,
    )
    pair_file_hash = require_expected_file_hash(
        args.pair_file,
        args.expected_pair_file_sha256,
    )
    comparison_id = comparison_contract_id(
        init_fingerprint=init_fingerprint,
        schedule_file_sha256=pair_file_hash,
        schedule_order_sha256=schedule_audit["ordered_content_sha256"],
        max_steps=effective_max_steps,
        batch_size=args.batch_size,
        grad_accum=effective_grad_accum,
        learning_rate=args.learning_rate,
    )

    set_deterministic_runtime(torch, args.seed, set_seed)
    initialize_cuda_peak_tracking(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded_rows, token_audit = tokenize_and_audit_schedule(
        rows,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_completion_tokens=args.max_completion_tokens,
        max_seq_len=args.max_seq_len,
    )
    smoke_longest_audit: dict[str, Any] | None = None
    if args.smoke_test:
        matching_indices = [
            index
            for index, row in enumerate(rows)
            if schedule_display_id(row, schedule_content_key(row)) == args.expected_longest_pair_id
        ]
        if len(matching_indices) != 1:
            raise RuntimeError(
                "dedicated smoke schedule must contain the expected global longest "
                f"pair exactly once; id={args.expected_longest_pair_id!r}, "
                f"matches={matching_indices}"
            )
        longest_index = matching_indices[0]
        encoded_longest = encoded_rows[longest_index]
        longest_sequence_tokens = len(encoded_longest["prompt_input_ids"]) + max(
            len(encoded_longest["chosen_input_ids"]),
            len(encoded_longest["rejected_input_ids"]),
        )
        observed_smoke_max = max(
            len(encoded["prompt_input_ids"])
            + max(len(encoded["chosen_input_ids"]), len(encoded["rejected_input_ids"]))
            for encoded in encoded_rows
        )
        if longest_sequence_tokens != observed_smoke_max:
            raise RuntimeError(
                "expected global longest pair is not the longest tokenized pair in "
                "the dedicated smoke schedule"
            )
        smoke_longest_audit = {
            "expected_longest_pair_id": args.expected_longest_pair_id,
            "included": True,
            "schedule_index": longest_index,
            "sequence_tokens": longest_sequence_tokens,
            "is_longest_in_smoke_schedule": True,
        }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        base,
        args.clean_sft_adapter,
        adapter_name=POLICY_ADAPTER,
        is_trainable=True,
    )
    model.set_adapter(POLICY_ADAPTER)

    reference_before: str | None = None
    policy_before = adapter_state_sha256(model, POLICY_ADAPTER, torch)
    if args.arm == "dpo":
        model.load_adapter(
            args.clean_sft_adapter,
            adapter_name=REFERENCE_ADAPTER,
            is_trainable=False,
        )
        model.set_adapter(POLICY_ADAPTER)
        set_reference_frozen(model)
        reference_before = adapter_state_sha256(model, REFERENCE_ADAPTER, torch)
        if policy_before != reference_before:
            raise RuntimeError("policy and reference adapters are not byte-identical at initialization")

    dropout_audit = disable_and_audit_dropout(model, torch)
    parameter_audit = trainable_parameter_audit(model)
    model.print_trainable_parameters()

    class SequentialTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(
                resolve_train_sampler_dataset(self, train_dataset)
            )

    class CompletionOnlyTrainer(SequentialTrainer):
        """Completion-only CE with tail logits only (same loss, lower VRAM)."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            # Our custom loss is already a per-microbatch mean and does not
            # consume num_items_in_batch.  Transformers 4.52.4 otherwise
            # infers True from Qwen2's **kwargs and skips the required /8
            # gradient-accumulation scaling.  TRL's pinned DPOTrainer applies
            # this same override for its custom loss.
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            labels = inputs["labels"]
            supervised = labels.ne(-100)
            if not bool(supervised.any()):
                raise RuntimeError("SFT microbatch has no supervised chosen tokens")
            first_supervised = int(supervised.nonzero(as_tuple=False)[:, 1].min().item())
            sequence_length = labels.shape[1]
            logits_to_keep = sequence_length - first_supervised + 1
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
            prediction_logits = outputs.logits[:, :-1, :]
            targets = labels[:, first_supervised:]
            if prediction_logits.shape[:2] != targets.shape:
                raise RuntimeError(
                    "tail-logit alignment failure: "
                    f"{tuple(prediction_logits.shape[:2])} != {tuple(targets.shape)}"
                )
            loss = functional.cross_entropy(
                prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite continued-SFT loss: {loss.detach().item()!r}")
            return (loss, outputs) if return_outputs else loss

    training_args_common = dict(
        output_dir=str(args.output_dir / "trainer_state"),
        max_steps=effective_max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        fp16=False,
        tf32=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        disable_tqdm=False,
    )

    # Create the run root before Trainer construction so Transformers can
    # safely initialize its nested output path.  The preflight empty-directory
    # check above prevents artifact mixing.
    # The orchestrator may pre-create the run directory for logs.  The
    # fail-closed check above already rejected every nonempty directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")

    dpo_runtime_audit: dict[str, Any] | None = None
    initial_dpo_audit: dict[str, Any] | None = None
    if args.arm == "continued_sft":
        sft_features = []
        for encoded in encoded_rows:
            ids = encoded["prompt_input_ids"] + encoded["chosen_input_ids"]
            prompt_length = len(encoded["prompt_input_ids"])
            sft_features.append(
                {
                    "input_ids": ids,
                    "attention_mask": [1] * len(ids),
                    "labels": [-100] * prompt_length + encoded["chosen_input_ids"],
                }
            )
        train_dataset = Dataset.from_list(sft_features)

        class SingleExampleCollator:
            def __call__(self, features):
                if len(features) != 1:
                    raise RuntimeError(
                        f"V4 freezes per-device microbatch size to 1; received {len(features)}"
                    )
                feature = features[0]
                return {
                    key: torch.tensor([feature[key]], dtype=torch.long)
                    for key in ("input_ids", "attention_mask", "labels")
                }

        train_args = TrainingArguments(
            remove_unused_columns=False,
            label_names=["labels"],
            **training_args_common,
        )
        trainer = CompletionOnlyTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=SingleExampleCollator(),
            processing_class=tokenizer,
        )
        objective = "chosen_completion_only_causal_cross_entropy"
    else:
        from trl import DPOConfig, DPOTrainer

        raw_dpo_rows = [
            {
                "schedule_index": index,
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            }
            for index, row in enumerate(rows)
        ]
        train_dataset = Dataset.from_list(raw_dpo_rows)

        class AuditedSequentialDPOTrainer(DPOTrainer):
            reference_context_entries = 0
            reference_was_frozen_in_every_context = True

            def _get_train_sampler(self, train_dataset=None):
                return SequentialSampler(
                    resolve_train_sampler_dataset(self, train_dataset)
                )

            @contextmanager
            def null_ref_context(self):
                # PEFT set_adapter normally toggles the selected adapter's
                # requires_grad flags.  Freeze it again before every reference
                # forward so the reference is immutable even transiently.
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.set_adapter(REFERENCE_ADAPTER)
                set_reference_frozen(unwrapped)
                frozen_now = all(
                    not parameter.requires_grad
                    for _, parameter in adapter_parameter_items(unwrapped, REFERENCE_ADAPTER)
                )
                self.reference_context_entries += 1
                self.reference_was_frozen_in_every_context &= frozen_now
                try:
                    yield
                finally:
                    unwrapped.set_adapter(POLICY_ADAPTER)
                    set_reference_frozen(unwrapped)

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs=False,
                num_items_in_batch=None,
            ):
                value = super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
                loss = value[0] if return_outputs else value
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"non-finite DPO loss: {loss.detach().item()!r}")
                return value

        dpo_args = DPOConfig(
            remove_unused_columns=True,
            beta=args.beta,
            loss_type=DPO_LOSS_TYPE,
            label_smoothing=DPO_LABEL_SMOOTHING,
            reference_free=False,
            model_adapter_name=POLICY_ADAPTER,
            ref_adapter_name=REFERENCE_ADAPTER,
            force_use_ref_model=False,
            disable_dropout=True,
            max_prompt_length=args.max_prompt_tokens,
            max_completion_length=args.max_completion_tokens,
            max_length=args.max_seq_len,
            truncation_mode="keep_end",
            precompute_ref_log_probs=True,
            precompute_ref_batch_size=1,
            use_logits_to_keep=True,
            generate_during_eval=False,
            padding_free=False,
            **training_args_common,
        )
        trainer = AuditedSequentialDPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        # Fail before GPU work if TRL's preprocessing altered or truncated any
        # frozen input.  This also catches future TRL API/behavior drift.
        if len(trainer.train_dataset) != len(encoded_rows):
            raise RuntimeError("TRL changed the DPO schedule cardinality")
        for index, (observed, expected) in enumerate(zip(trainer.train_dataset, encoded_rows)):
            for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
                if observed[key] != expected[key]:
                    raise RuntimeError(f"TRL tokenization/truncation drift at row {index}, field {key}")
        objective = "sigmoid_direct_preference_optimization"

    if trainer.model_accepts_loss_kwargs is not False:
        raise RuntimeError(
            "custom preference loss would bypass frozen gradient-accumulation scaling"
        )

    started_at = utc_now()
    start_time = time.monotonic()

    if args.arm == "dpo":
        # Trigger reference precomputation, then numerically verify that
        # adapter switching produces the same initial log probabilities before
        # the first optimizer step.  State hashes alone cannot catch a broken
        # adapter-switching path.
        initial_loader = trainer.get_train_dataloader()
        max_abs_logp_delta = 0.0
        initial_losses: list[float] = []
        initial_rows_checked = 0
        model.eval()
        for batch in initial_loader:
            with torch.no_grad(), trainer.compute_loss_context_manager():
                policy_output = trainer.concatenated_forward(model, batch)
                policy_chosen = policy_output["chosen_logps"]
                policy_rejected = policy_output["rejected_logps"]
                ref_chosen = batch["ref_chosen_logps"].to(policy_chosen.device)
                ref_rejected = batch["ref_rejected_logps"].to(policy_rejected.device)
                batch_delta = torch.max(
                    torch.cat(
                        (
                            (policy_chosen - ref_chosen).abs().reshape(-1),
                            (policy_rejected - ref_rejected).abs().reshape(-1),
                        )
                    )
                )
                losses, _, _ = trainer.dpo_loss(
                    policy_chosen,
                    policy_rejected,
                    ref_chosen,
                    ref_rejected,
                )
            delta_value = float(batch_delta.item())
            loss_values = [float(value) for value in losses.detach().float().cpu().tolist()]
            if not math.isfinite(delta_value) or not all(
                math.isfinite(value) for value in loss_values
            ):
                raise RuntimeError("non-finite initial DPO policy/reference audit")
            max_abs_logp_delta = max(max_abs_logp_delta, delta_value)
            initial_losses.extend(loss_values)
            initial_rows_checked += len(loss_values)
        model.train()
        initial_loss_mean = statistics.fmean(initial_losses)
        if initial_rows_checked != len(rows):
            raise RuntimeError(
                f"initial DPO audit row drift: {initial_rows_checked} != {len(rows)}"
            )
        if max_abs_logp_delta > 1.0e-4:
            raise RuntimeError(
                "initial policy/reference log-probability mismatch: "
                f"{max_abs_logp_delta} > 1e-4"
            )
        expected_initial_loss = -math.log(0.5)
        if abs(initial_loss_mean - expected_initial_loss) > 0.02:
            raise RuntimeError(
                f"initial DPO loss drift: {initial_loss_mean} is not within "
                f"0.02 of {expected_initial_loss}"
            )
        initial_dpo_audit = {
            "rows_checked_before_first_optimizer_step": initial_rows_checked,
            "max_abs_policy_reference_logp_delta": max_abs_logp_delta,
            "max_abs_logp_delta_limit": 1.0e-4,
            "policy_reference_logps_match": True,
            "mean_initial_dpo_loss": initial_loss_mean,
            "expected_initial_dpo_loss": expected_initial_loss,
            "initial_dpo_loss_tolerance": 0.02,
            "initial_dpo_loss_within_tolerance": True,
        }

    result = trainer.train()
    duration_seconds = time.monotonic() - start_time
    completed_at = utc_now()

    if trainer.state.global_step != effective_max_steps:
        raise RuntimeError(
            f"optimizer-step drift: {trainer.state.global_step} != {effective_max_steps}"
        )
    if int(trainer.state.num_input_tokens_seen or 0) not in (0,):
        # This Trainer metric is not comparable across SFT/DPO because DPO
        # concatenates two sequences.  It is retained in logs, while the
        # explicit exposure audit below is the comparison contract.
        pass
    loss_audit = finite_loss_audit(trainer.state.log_history, result.metrics)
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    smoke_memory_audit = {
        "applicable": args.smoke_test,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "limit_bytes": SMOKE_MAX_RESERVED_BYTES,
        "within_limit": peak_reserved <= SMOKE_MAX_RESERVED_BYTES,
    }
    if args.smoke_test:
        write_json(args.output_dir / "smoke_memory_audit.json", smoke_memory_audit)
        if peak_reserved > SMOKE_MAX_RESERVED_BYTES:
            raise RuntimeError(
                "preference smoke exceeded the frozen unattended-run VRAM gate: "
                f"{peak_reserved} > {SMOKE_MAX_RESERVED_BYTES} bytes"
            )

    model.set_adapter(POLICY_ADAPTER)
    if args.arm == "dpo":
        set_reference_frozen(model)
    policy_after = adapter_state_sha256(model, POLICY_ADAPTER, torch)
    if policy_after == policy_before:
        raise RuntimeError("policy adapter state did not change during continuation training")
    parameter_audit.update(
        {
            "policy_initial_state_sha256": policy_before,
            "policy_final_state_sha256": policy_after,
            "policy_state_changed": True,
        }
    )

    if args.arm == "dpo":
        reference_after = adapter_state_sha256(model, REFERENCE_ADAPTER, torch)
        if reference_after != reference_before:
            raise RuntimeError("frozen DPO reference adapter weights changed during training")
        if not trainer.reference_was_frozen_in_every_context:
            raise RuntimeError("reference adapter was trainable in at least one reference context")
        if trainer.reference_context_entries <= 0:
            raise RuntimeError("DPO never entered the frozen reference-adapter context")
        if not trainer._precomputed_train_ref_log_probs:
            raise RuntimeError("DPO reference log probabilities were not precomputed")
        ref_logprob_audit = reference_logprob_audit(trainer.train_dataset)
        if ref_logprob_audit["row_count"] != len(rows):
            raise RuntimeError("reference precompute row-count drift")
        optimizer_overlap = optimizer_reference_overlap(trainer, model)
        if optimizer_overlap:
            raise RuntimeError(
                f"optimizer unexpectedly contains {optimizer_overlap} reference tensors"
            )
        reference_requires_grad_after = sum(
            int(parameter.requires_grad)
            for _, parameter in adapter_parameter_items(model, REFERENCE_ADAPTER)
        )
        if reference_requires_grad_after:
            raise RuntimeError("reference adapter has trainable parameters after DPO")
        dpo_runtime_audit = {
            "implementation": "trl_0.18.2_single_nf4_base_dual_adapter_precomputed_reference",
            "policy_adapter_name": POLICY_ADAPTER,
            "reference_adapter_name": REFERENCE_ADAPTER,
            "policy_and_reference_identical_at_initialization": True,
            "policy_initial_state_sha256": policy_before,
            "reference_initial_state_sha256": reference_before,
            "reference_final_state_sha256": reference_after,
            "reference_state_unchanged": True,
            "reference_context_entries": trainer.reference_context_entries,
            "reference_frozen_in_every_context": True,
            "reference_tensors_in_optimizer": optimizer_overlap,
            "reference_requires_grad_tensors_after_training": reference_requires_grad_after,
            "reference_logprob_precompute": ref_logprob_audit,
            "initial_numerical_audit": initial_dpo_audit,
            "policy_final_state_sha256": policy_after,
            "policy_state_changed": True,
            "beta": args.beta,
            "loss_type": DPO_LOSS_TYPE,
            "label_smoothing": DPO_LABEL_SMOOTHING,
            "reference_free": False,
            "use_logits_to_keep": True,
        }

    checkpoint = args.output_dir / "checkpoint_final"
    model.save_pretrained(checkpoint, selected_adapters=[POLICY_ADAPTER])
    tokenizer.save_pretrained(checkpoint)
    output_fingerprint, output_file_hashes = checkpoint_fingerprint(checkpoint)

    environment = environment_audit(torch)
    training_metrics = dict(result.metrics)
    training_metrics.update(
        {
            "arm": args.arm,
            "objective": objective,
            "global_step": trainer.state.global_step,
            "finite_loss": True,
            "peak_cuda_memory_allocated_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
        }
    )
    write_json(args.output_dir / "training_metrics.json", training_metrics)
    write_json(args.output_dir / "training_log.json", trainer.state.log_history)

    if args.arm == "continued_sft":
        logical_token_compute = {
            "definition": "logical nonpadding sequence tokens; excludes gradient-checkpoint recomputation",
            "training_policy_sequence_tokens": token_audit["chosen_sequence_tokens"],
            "reference_precompute_sequence_tokens": 0,
            "initial_numerical_audit_policy_sequence_tokens": 0,
            "total_recorded_logical_sequence_tokens": token_audit["chosen_sequence_tokens"],
        }
    else:
        preference_sequence_tokens = (
            token_audit["chosen_sequence_tokens"] + token_audit["rejected_sequence_tokens"]
        )
        logical_token_compute = {
            "definition": "logical nonpadding sequence tokens; excludes gradient-checkpoint recomputation",
            "training_policy_sequence_tokens": preference_sequence_tokens,
            "reference_precompute_sequence_tokens": preference_sequence_tokens,
            "initial_numerical_audit_policy_sequence_tokens": preference_sequence_tokens,
            "total_recorded_logical_sequence_tokens": 3 * preference_sequence_tokens,
        }

    manifest = {
        "protocol": PROTOCOL,
        "source_tag": FROZEN_TAG,
        "source_commit": args.source_commit,
        "formal_result": not args.smoke_test,
        "arm": args.arm,
        "objective": objective,
        "comparison_contract_id": comparison_id,
        "model": args.model,
        "model_revision": args.model_revision,
        "clean_sft_initialization": {
            "path": str(args.clean_sft_adapter),
            "checkpoint_fingerprint": init_fingerprint,
            "file_sha256": init_file_hashes,
            "loaded_policy_state_sha256": policy_before,
        },
        "data": {
            "train_schedule": str(args.pair_file),
            "train_schedule_sha256": pair_file_hash,
            "expected_train_schedule_sha256": args.expected_pair_file_sha256,
            "train_schedule_hash_matches_expected": True,
            "schedule": schedule_audit,
            "split_audit": split_audit,
            "strata_audit": strata_audit,
            "smoke_longest_pair_audit": smoke_longest_audit,
            "token_exposure": token_audit,
            "held_out_test_accessed": False,
        },
        "compute_contract": {
            "seed": args.seed,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_completion_tokens": args.max_completion_tokens,
            "max_sequence_tokens": args.max_seq_len,
            "optimizer_steps": effective_max_steps,
            "microbatch_size": args.batch_size,
            "gradient_accumulation_steps": effective_grad_accum,
            "model_accepts_loss_kwargs": False,
            "custom_loss_gradient_accumulation_scaled_by_trainer": True,
            "scheduled_microbatches": len(rows),
            "chosen_exposures": len(rows),
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": LR_SCHEDULER,
            "warmup_steps": WARMUP_STEPS,
            "optimizer": "paged_adamw_8bit",
            "optimizer_state_reused_from_clean_sft": False,
            "scheduler_state_reused_from_clean_sft": False,
            "sampler": "SequentialSampler",
            "row_order_shuffled": False,
            "bf16": True,
            "quantization": "NF4 double-quantized base",
            "gradient_checkpointing": True,
            "dropout": 0.0,
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
        },
        "logical_token_compute_audit": logical_token_compute,
        "dropout_audit": dropout_audit,
        "parameter_audit": parameter_audit,
        "loss_audit": loss_audit,
        "dpo_reference_audit": dpo_runtime_audit,
        "smoke_memory_audit": smoke_memory_audit,
        "runtime": {
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": duration_seconds,
            "peak_cuda_memory_allocated_bytes": peak_allocated,
            "peak_cuda_memory_reserved_bytes": peak_reserved,
        },
        "output_checkpoint": {
            "path": str(checkpoint),
            "checkpoint_fingerprint": output_fingerprint,
            "file_sha256": output_file_hashes,
        },
        "environment": environment,
        "library_contract": {
            "transformers": "4.52.4",
            "peft": "0.15.2",
            "trl": "0.18.2",
        },
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "arm": args.arm,
                "checkpoint": str(checkpoint),
                "comparison_contract_id": comparison_id,
                "train_loss": loss_audit["final_train_loss"],
                "peak_cuda_memory_allocated_bytes": peak_allocated,
                "run_manifest": str(args.output_dir / "run_manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
