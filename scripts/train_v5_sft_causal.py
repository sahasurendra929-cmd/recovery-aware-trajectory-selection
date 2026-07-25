#!/usr/bin/env python3
"""Train one frozen V5 Stage-1 message-level SFT arm with 7B QLoRA.

The input contract is deliberately message structured.  Each JSONL row has
``messages`` and a parallel boolean ``label_mask``.  A true mask value is
allowed only for an assistant message.  The Qwen chat template is rendered
without truncation, and only the content/tool-call tokens (plus the assistant
end marker) of selected assistant messages enter causal-LM cross entropy.

This module keeps all GPU/ML imports inside ``main`` so its contract helpers
can be unit tested on a CPU-only coordinator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 20260722
MAX_SEQUENCE_TOKENS = 8192
FORMAL_STEPS = 64
FORMAL_BATCH_SIZE = 1
FORMAL_GRAD_ACCUM = 8
FORMAL_SCHEDULE_ROWS = FORMAL_STEPS * FORMAL_BATCH_SIZE * FORMAL_GRAD_ACCUM
FORMAL_LEARNING_RATE = 1.0e-4
SMOKE_STEPS = 2
SMOKE_GRAD_ACCUM = 8
SMOKE_ROWS = SMOKE_STEPS * FORMAL_BATCH_SIZE * SMOKE_GRAD_ACCUM
RATIO_TOLERANCE = 0.01
MIN_GPU_MEMORY_BYTES = 20 * 1024**3
ROOT = Path(__file__).resolve().parents[1]
IGNORE_INDEX = -100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DATA_AUDIT_PROTOCOL = "v5_stage1_sft_causal"
DYNAMIC_AUDIT_PROTOCOL = "v5_stage1_dynamic_injection_audit"
EXPECTED_DYNAMIC_AUDITS = {
    "generation": {
        "source_split": "inner_train",
        "verified_injections": 83,
    },
    "validation": {
        "source_split": "derived_validation",
        "verified_injections": 21,
    },
}

ARM_TARGET_RECOVERY_RATIOS = {
    "perfect_success": 0.0,
    "failure_raw": 1.0,
    "repair_25": 0.25,
    "repair_50": 0.50,
    "repair_75": 0.75,
    "repair_100": 1.0,
}
REPAIR_ARMS = {"repair_25", "repair_50", "repair_75", "repair_100"}
ROW_KEYS = {"id", "messages", "label_mask", "metadata", "token_contract"}
MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str, name: str) -> str:
    normalized = value.lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def validate_training_data_provenance(
    *,
    arm: str,
    train_file: Path,
    validation_file: Path,
    data_audit_path: Path,
    data_hashes_path: Path,
    expected_train_sha256: str,
    expected_validation_sha256: str,
) -> dict[str, Any]:
    """Bind one training run to the constructor's fail-closed audit bundle."""

    data_root = data_audit_path.resolve().parent
    if data_audit_path.resolve() != data_root / "audit.json":
        raise RuntimeError("--data-audit must name the constructor's audit.json")
    if data_hashes_path.resolve() != data_root / "hashes.json":
        raise RuntimeError(
            "--data-hashes must name hashes.json beside the constructor audit"
        )
    expected_train_path = data_root / "arms" / arm / "train.jsonl"
    expected_validation_path = data_root / "validation_loss.jsonl"
    if train_file.resolve() != expected_train_path:
        raise RuntimeError(
            f"--train-file must be the audited {arm} file: {expected_train_path}"
        )
    if validation_file.resolve() != expected_validation_path:
        raise RuntimeError(
            "--validation-file must be the audited validation_loss.jsonl: "
            f"{expected_validation_path}"
        )

    audit = read_json_object(data_audit_path, label="data audit")
    hashes = read_json_object(data_hashes_path, label="data hash manifest")
    if audit.get("status") != "PASS":
        raise RuntimeError("data audit status is not PASS")
    if audit.get("protocol") != DATA_AUDIT_PROTOCOL:
        raise RuntimeError("data audit protocol drift")
    if audit.get("official_test_used") is not False:
        raise RuntimeError("data audit does not seal the official test")
    if audit.get("derived_validation_used_for_supervision") is not False:
        raise RuntimeError("data audit used derived validation for supervision")
    if audit.get("train_validation_source_overlap") != 0:
        raise RuntimeError("data audit reports train/validation source leakage")
    label_guarantees = audit.get("label_guarantees")
    if (
        not isinstance(label_guarantees, dict)
        or label_guarantees.get(
            "official_test_and_derived_validation_label_leakage"
        )
        != 0
    ):
        raise RuntimeError("data audit lacks the frozen supervision-leakage seal")

    required_hash_keys = {
        "audit.json",
        f"arms/{arm}/train.jsonl",
        "validation_loss.jsonl",
    }
    for key in required_hash_keys:
        value = hashes.get(key)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"data hash manifest lacks valid {key} SHA-256")

    audit_sha = sha256_file(data_audit_path)
    hashes_sha = sha256_file(data_hashes_path)
    train_sha = sha256_file(train_file)
    validation_sha = sha256_file(validation_file)
    if hashes["audit.json"] != audit_sha:
        raise RuntimeError("data audit/hash manifest binding drift")
    if hashes[f"arms/{arm}/train.jsonl"] != train_sha:
        raise RuntimeError("audited arm/hash manifest binding drift")
    if hashes["validation_loss.jsonl"] != validation_sha:
        raise RuntimeError("validation-loss/hash manifest binding drift")
    if train_sha != expected_train_sha256:
        raise RuntimeError(f"training data hash drift: {train_sha}")
    if validation_sha != expected_validation_sha256:
        raise RuntimeError(f"validation data hash drift: {validation_sha}")

    arms = audit.get("arms")
    arm_audit = arms.get(arm) if isinstance(arms, dict) else None
    if not isinstance(arm_audit, dict) or arm_audit.get("sha256") != train_sha:
        raise RuntimeError("selected arm SHA is not bound into data audit")
    validation_audit = audit.get("validation_loss")
    if (
        not isinstance(validation_audit, dict)
        or validation_audit.get("sha256") != validation_sha
        or validation_audit.get("source_split") != "inner_train"
    ):
        raise RuntimeError("validation-loss SHA/source is not bound into data audit")

    dynamic = audit.get("dynamic_audits")
    if not isinstance(dynamic, dict):
        raise RuntimeError("data audit lacks dynamic audit identities")
    dynamic_identities: dict[str, dict[str, Any]] = {}
    split_sha = audit.get("split_manifest_sha256")
    if not isinstance(split_sha, str) or SHA256_RE.fullmatch(split_sha) is None:
        raise RuntimeError("data audit split-manifest SHA is invalid")
    for name, expected in EXPECTED_DYNAMIC_AUDITS.items():
        identity = dynamic.get(name)
        if not isinstance(identity, dict):
            raise RuntimeError(f"data audit lacks {name} dynamic audit identity")
        if identity.get("protocol") != DYNAMIC_AUDIT_PROTOCOL:
            raise RuntimeError(f"{name} dynamic audit protocol drift")
        if identity.get("source_split") != expected["source_split"]:
            raise RuntimeError(f"{name} dynamic audit source split drift")
        if identity.get("verified_injections") != expected["verified_injections"]:
            raise RuntimeError(f"{name} dynamic audit coverage drift")
        if identity.get("official_test_used") is not False:
            raise RuntimeError(f"{name} dynamic audit opened official test")
        if identity.get("official_test_sealed") is not True:
            raise RuntimeError(f"{name} dynamic audit lacks official-test seal")
        if identity.get("split_manifest_sha256") != split_sha:
            raise RuntimeError(f"{name} dynamic audit split SHA drift")
        for field in ("sha256", "manifest_sha256", "split_manifest_sha256"):
            value = identity.get(field)
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise RuntimeError(
                    f"{name} dynamic audit identity lacks valid {field}"
                )
        dynamic_identities[name] = dict(identity)

    return {
        "data_audit_path": str(data_audit_path),
        "data_audit_sha256": audit_sha,
        "data_hashes_path": str(data_hashes_path),
        "data_hashes_sha256": hashes_sha,
        "train_file_sha256": train_sha,
        "validation_file_sha256": validation_sha,
        "dynamic_audits": dynamic_identities,
        "official_test_used": False,
        "official_test_sealed": True,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"JSONL input does not exist: {path}")
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise RuntimeError(
                    f"{path}:{line_number}: every JSONL record must end in newline"
                )
            if not line.strip():
                raise RuntimeError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError(f"{path}:{line_number}: row must be an object")
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier:
                raise RuntimeError(f"{path}:{line_number}: missing non-empty id")
            if identifier in identifiers:
                raise RuntimeError(f"{path}:{line_number}: duplicate id {identifier!r}")
            identifiers.add(identifier)
            rows.append(row)
    if not rows:
        raise RuntimeError(f"JSONL input is empty: {path}")
    return rows


def validate_fit_partition(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Reject loss-validation examples reused by the optimization schedule."""
    def source_ids(rows: list[dict[str, Any]], split: str) -> set[str]:
        identifiers: set[str] = set()
        for row in rows:
            metadata = row.get("metadata")
            source_id = metadata.get("source_example_id") if isinstance(metadata, dict) else None
            if not isinstance(source_id, str) or not source_id:
                raise RuntimeError(
                    f"{row.get('id', '<unknown>')}: {split} source_example_id is required"
                )
            identifiers.add(source_id)
        return identifiers

    train_sources = source_ids(train_rows, "train")
    validation_sources = source_ids(validation_rows, "validation_loss")
    overlap = train_sources & validation_sources
    if overlap:
        sample = sorted(overlap)[:3]
        raise RuntimeError(
            "train/validation-loss source leakage: "
            f"{len(overlap)} overlapping source_example_id values, e.g. {sample}"
        )
    return {
        "train_source_examples": len(train_sources),
        "validation_loss_source_examples": len(validation_sources),
        "overlap": 0,
    }


def _content_text(value: Any, *, where: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return canonical(value)
    raise RuntimeError(f"{where}: content must be string, object, list, or null")


def _normalize_tool_call(call: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise RuntimeError(f"{where}: tool call must be an object")
    function = call.get("function")
    if function is not None:
        if call.get("type", "function") != "function" or not isinstance(function, dict):
            raise RuntimeError(f"{where}: malformed standard function call")
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"{where}: tool call requires a non-empty name")
    if isinstance(arguments, str):
        try:
            arguments_value = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{where}: arguments are not valid JSON") from error
    else:
        arguments_value = arguments
    if not isinstance(arguments_value, dict):
        raise RuntimeError(f"{where}: tool-call arguments must encode an object")
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            # Qwen2.5's chat template applies ``tojson`` itself.  Keeping an
            # object here avoids serializing arguments as a quoted JSON string.
            "arguments": arguments_value,
        },
    }
    call_id = call.get("id")
    if call_id is not None:
        if not isinstance(call_id, str) or not call_id:
            raise RuntimeError(f"{where}: tool-call id must be a non-empty string")
        normalized["id"] = call_id
    return normalized


def normalize_message(message: Any, *, where: str) -> dict[str, Any]:
    """Convert compact τ² or OpenAI-style messages to Qwen chat-template form."""
    if not isinstance(message, dict):
        raise RuntimeError(f"{where}: message must be an object")
    role = message.get("role")
    if role not in MESSAGE_ROLES:
        raise RuntimeError(f"{where}: unsupported role {role!r}")
    normalized: dict[str, Any] = {
        "role": role,
        "content": _content_text(message.get("content"), where=where),
    }
    calls = message.get("tool_calls") or []
    if calls:
        if role != "assistant" or not isinstance(calls, list):
            raise RuntimeError(f"{where}: only assistant messages may contain tool_calls")
        if len(calls) != 1:
            raise RuntimeError(f"{where}: Stage-1 freezes exactly one tool call per turn")
        normalized["tool_calls"] = [
            _normalize_tool_call(calls[0], where=f"{where}.tool_calls[0]")
        ]
    if role == "assistant" and not normalized["content"] and not calls:
        raise RuntimeError(f"{where}: assistant message has no content or tool call")
    if role == "tool":
        tool_call_id = message.get("tool_call_id", message.get("id"))
        if tool_call_id is not None:
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise RuntimeError(f"{where}: tool-call link id must be a string")
            normalized["tool_call_id"] = tool_call_id
    return normalized


def _call_id(message: dict[str, Any]) -> str | None:
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    call = calls[0]
    value = call.get("id")
    return value if isinstance(value, str) else None


def failed_assistant_indices(messages: list[dict[str, Any]], *, row_id: str) -> list[int]:
    """Return assistant indices whose immediately linked tool result is an error."""
    failed: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        if index == 0 or messages[index - 1].get("role") != "assistant":
            raise RuntimeError(f"{row_id}: tool message {index} lacks adjacent assistant")
        assistant = messages[index - 1]
        if not assistant.get("tool_calls"):
            raise RuntimeError(f"{row_id}: tool message {index} has no adjacent tool call")
        assistant_id = _call_id(assistant)
        tool_id = message.get("tool_call_id", message.get("id"))
        if assistant_id is not None and tool_id is not None and assistant_id != tool_id:
            raise RuntimeError(
                f"{row_id}: tool-call id mismatch at messages {index - 1}/{index}"
            )
        if not isinstance(message.get("error"), bool):
            raise RuntimeError(f"{row_id}: tool message {index} needs boolean error")
        if message["error"]:
            failed.append(index - 1)
    return failed


def validate_row(
    row: dict[str, Any],
    *,
    arm: str | None,
    split: str,
) -> dict[str, Any]:
    """Validate the message/mask and arm semantics before tokenization."""
    unknown = set(row) - ROW_KEYS
    missing = {"id", "messages", "label_mask", "metadata"} - set(row)
    if unknown:
        raise RuntimeError(f"{row.get('id', '<unknown>')}: unknown row keys {sorted(unknown)}")
    if missing:
        raise RuntimeError(f"{row.get('id', '<unknown>')}: missing row keys {sorted(missing)}")
    row_id = row["id"]
    if not isinstance(row_id, str) or not row_id:
        raise RuntimeError(f"{split}: row id must be non-empty")
    messages = row["messages"]
    mask = row["label_mask"]
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"{row_id}: messages must be a non-empty list")
    if (
        not isinstance(mask, list)
        or len(mask) != len(messages)
        or any(type(value) is not bool for value in mask)
    ):
        raise RuntimeError(
            f"{row_id}: label_mask must contain one boolean per message"
        )
    if not isinstance(row["metadata"], dict):
        raise RuntimeError(f"{row_id}: metadata must be an object")
    normalized = [
        normalize_message(message, where=f"{row_id}.messages[{index}]")
        for index, message in enumerate(messages)
    ]
    selected = [index for index, value in enumerate(mask) if value]
    if not selected:
        raise RuntimeError(f"{row_id}: no supervised assistant message")
    for index in selected:
        if messages[index].get("role") != "assistant":
            raise RuntimeError(
                f"{row_id}: label_mask[{index}]=true for non-assistant role"
            )
        if not messages[index].get("tool_calls"):
            raise RuntimeError(
                f"{row_id}: Stage-1 supervises assistant tool calls only"
            )
    if selected[-1] + 1 != len(messages) - 1:
        raise RuntimeError(
            f"{row_id}: final supervised assistant must have exactly one "
            "adjacent evidence tool result and no later messages"
        )
    for index in selected:
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
            raise RuntimeError(
                f"{row_id}: selected assistant tool call {index} lacks its "
                "adjacent evidence tool result"
            )
        assistant_id = _call_id(messages[index])
        tool_message = messages[index + 1]
        tool_id = tool_message.get("tool_call_id", tool_message.get("id"))
        if assistant_id is not None and tool_id is not None and assistant_id != tool_id:
            raise RuntimeError(
                f"{row_id}: selected tool-call/result id mismatch at "
                f"messages {index}/{index + 1}"
            )
        if not isinstance(tool_message.get("error"), bool):
            raise RuntimeError(
                f"{row_id}: selected tool result {index + 1} needs boolean error"
            )
    failed = failed_assistant_indices(messages, row_id=row_id)
    failed_set = set(failed)
    selected_set = set(selected)
    metadata = row["metadata"]
    if metadata.get("source_split") != "inner_train":
        raise RuntimeError(
            f"{row_id}: both training and loss-validation data must come from "
            "the inner_train split"
        )
    expected_fit_split = "train_schedule" if split == "train" else "validation_loss"
    if metadata.get("fit_split") != expected_fit_split:
        raise RuntimeError(
            f"{row_id}: metadata.fit_split must be {expected_fit_split!r}"
        )
    source_example_id = metadata.get("source_example_id")
    if not isinstance(source_example_id, str) or not source_example_id:
        raise RuntimeError(f"{row_id}: metadata.source_example_id is required")
    reward = metadata.get("full_trajectory_reward")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or float(reward) != 1.0
    ):
        raise RuntimeError(f"{row_id}: full_trajectory_reward must equal 1.0")
    trajectory_sha = metadata.get("source_trajectory_sha256")
    if not isinstance(trajectory_sha, str) or SHA256_RE.fullmatch(trajectory_sha) is None:
        raise RuntimeError(
            f"{row_id}: source_trajectory_sha256 must be lowercase SHA-256"
        )
    if metadata.get("official_test_used") is not False:
        raise RuntimeError(f"{row_id}: official_test_used must be false")
    metadata_arm = metadata.get("arm")
    if split == "train" and metadata_arm != arm:
        raise RuntimeError(
            f"{row_id}: metadata.arm={metadata_arm!r} does not match {arm!r}"
        )
    source = metadata.get("source")
    if source not in {"perfect_success", "failure_rich"}:
        raise RuntimeError(
            f"{row_id}: metadata.source must be perfect_success or failure_rich"
        )
    recorded_failed = metadata.get("failed_assistant_message_indices")
    if recorded_failed != failed:
        raise RuntimeError(
            f"{row_id}: failed_assistant_message_indices disagree with messages"
        )
    injected_failed = metadata.get("injected_failed_assistant_message_index")
    if failed:
        if len(failed) != 1 or injected_failed != failed[0]:
            raise RuntimeError(
                f"{row_id}: recovery row must contain exactly its one controlled "
                "injected failure"
            )
        if source != "failure_rich":
            raise RuntimeError(f"{row_id}: recovery messages need failure_rich source")
    else:
        if injected_failed is not None:
            raise RuntimeError(f"{row_id}: clean row records an injected failure")
        if source != "perfect_success":
            raise RuntimeError(f"{row_id}: clean messages need perfect_success source")
    tool_schemas = metadata.get("tool_schemas")
    if not isinstance(tool_schemas, list) or not tool_schemas:
        raise RuntimeError(f"{row_id}: metadata.tool_schemas must be a non-empty list")
    if any(not isinstance(schema, dict) for schema in tool_schemas):
        raise RuntimeError(f"{row_id}: every tool schema must be an object")
    tool_schema_hash = metadata.get("tool_schemas_sha256")
    observed_schema_hash = hashlib.sha256(
        canonical(tool_schemas).encode("utf-8")
    ).hexdigest()
    if tool_schema_hash != observed_schema_hash:
        raise RuntimeError(f"{row_id}: tool_schemas_sha256 mismatch")
    if messages[0].get("role") != "system":
        raise RuntimeError(f"{row_id}: first message must be the agent system policy")

    if split == "validation":
        if failed_set & selected_set:
            raise RuntimeError(
                f"{row_id}: validation may not positively supervise a failed action"
            )
        if failed and not any(index > failed[0] for index in selected):
            raise RuntimeError(f"{row_id}: loss-validation recovery has no repair label")
    elif arm == "perfect_success":
        if failed:
            raise RuntimeError(f"{row_id}: perfect_success contains a tool error")
    elif arm == "failure_raw":
        if not failed:
            raise RuntimeError(f"{row_id}: failure_raw row has no failed action")
        if not failed_set <= selected_set:
            raise RuntimeError(
                f"{row_id}: failure_raw must supervise every failed action"
            )
        if not any(index > failed[0] for index in selected):
            raise RuntimeError(f"{row_id}: failure_raw has no successful repair label")
    elif arm in REPAIR_ARMS:
        if failed_set & selected_set:
            raise RuntimeError(
                f"{row_id}: repair arm must mask every failed action"
            )
        if failed and not any(
            index > max(failed) and index in selected_set for index in range(len(messages))
        ):
            raise RuntimeError(
                f"{row_id}: recovery context lacks a supervised post-error repair"
            )
        if arm == "repair_100" and not failed:
            raise RuntimeError(f"{row_id}: repair_100 requires a recovery context")
    elif split == "train":
        raise RuntimeError(f"unsupported training arm: {arm!r}")
    selected_errors = {
        index for index in selected if messages[index + 1]["error"]
    }
    allowed_selected_errors = {failed[0]} if arm == "failure_raw" and failed else set()
    if selected_errors != allowed_selected_errors:
        raise RuntimeError(
            f"{row_id}: selected error labels {sorted(selected_errors)} differ "
            f"from allowed controlled failure {sorted(allowed_selected_errors)}"
        )

    token_contract = row.get("token_contract")
    if not isinstance(token_contract, dict):
        raise RuntimeError(f"{row_id}: token_contract must be an object")
    return {
        "id": row_id,
        "messages": normalized,
        "label_mask": list(mask),
        "metadata": dict(metadata),
        "failed_message_indices": failed,
        "is_recovery": bool(failed),
        "token_contract": token_contract,
    }


def _token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    generation: bool,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=generation,
    )
    if isinstance(result, dict):
        result = result.get("input_ids")
    if not isinstance(result, list) or any(type(value) is not int for value in result):
        raise RuntimeError("tokenizer chat template did not return a flat integer list")
    return result


def encode_row(
    tokenizer: Any,
    validated: dict[str, Any],
    *,
    max_seq_len: int = MAX_SEQUENCE_TOKENS,
) -> dict[str, Any]:
    """Render a row and map selected assistant messages to exact token spans."""
    row_id = validated["id"]
    messages = validated["messages"]
    mask = validated["label_mask"]
    tools = validated["metadata"]["tool_schemas"]
    full_ids = _token_ids(
        tokenizer,
        messages,
        tools=tools,
        generation=False,
    )
    if not full_ids:
        raise RuntimeError(f"{row_id}: chat template produced no tokens")
    if len(full_ids) > max_seq_len:
        raise RuntimeError(
            f"{row_id}: {len(full_ids)} tokens exceed {max_seq_len}; "
            "Stage-1 forbids truncation"
        )
    labels = [IGNORE_INDEX] * len(full_ids)
    spans: list[dict[str, int]] = []
    for message_index, selected in enumerate(mask):
        if not selected:
            continue
        before = _token_ids(
            tokenizer,
            messages[:message_index],
            tools=tools,
            generation=True,
        )
        through = _token_ids(
            tokenizer,
            messages[: message_index + 1],
            tools=tools,
            generation=False,
        )
        if len(before) >= len(through):
            raise RuntimeError(f"{row_id}: empty label span at message {message_index}")
        if through[: len(before)] != before:
            raise RuntimeError(
                f"{row_id}: chat template is not prefix-stable at message {message_index}"
            )
        if full_ids[: len(through)] != through:
            raise RuntimeError(
                f"{row_id}: future messages altered prefix tokenization at "
                f"message {message_index}"
            )
        start, end = len(before), len(through)
        if any(value != IGNORE_INDEX for value in labels[start:end]):
            raise RuntimeError(f"{row_id}: overlapping label spans")
        labels[start:end] = full_ids[start:end]
        spans.append(
            {
                "message_index": message_index,
                "token_start": start,
                "token_end": end,
            }
        )
    supervised_tokens = sum(value != IGNORE_INDEX for value in labels)
    if supervised_tokens <= 0:
        raise RuntimeError(f"{row_id}: tokenized row has no supervised tokens")
    contract = validated["token_contract"]
    required_contract = {"sequence_tokens", "supervised_tokens", "label_spans"}
    if set(contract) != required_contract:
        raise RuntimeError(
            f"{row_id}: token_contract keys must be {sorted(required_contract)}"
        )
    if contract["sequence_tokens"] != len(full_ids):
        raise RuntimeError(f"{row_id}: sequence token contract drift")
    if contract["supervised_tokens"] != supervised_tokens:
        raise RuntimeError(f"{row_id}: supervised token contract drift")
    if contract["label_spans"] != spans:
        raise RuntimeError(f"{row_id}: label-span contract drift")
    return {
        "id": row_id,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "label_spans": spans,
        "sequence_tokens": len(full_ids),
        "supervised_tokens": supervised_tokens,
        "is_recovery": validated["is_recovery"],
        "failed_message_count": len(validated["failed_message_indices"]),
        "failed_label_message_count": len(
            set(validated["failed_message_indices"])
            & {index for index, value in enumerate(mask) if value}
        ),
        "failed_label_tokens": sum(
            span["token_end"] - span["token_start"]
            for span in spans
            if span["message_index"] in set(validated["failed_message_indices"])
        ),
    }


def encode_rows(
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    *,
    arm: str | None,
    split: str,
    max_seq_len: int = MAX_SEQUENCE_TOKENS,
) -> list[dict[str, Any]]:
    return [
        encode_row(
            tokenizer,
            validate_row(row, arm=arm, split=split),
            max_seq_len=max_seq_len,
        )
        for row in rows
    ]


def arm_audit(encoded_rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    loss_tokens = sum(row["supervised_tokens"] for row in encoded_rows)
    recovery_loss_tokens = sum(
        row["supervised_tokens"] for row in encoded_rows if row["is_recovery"]
    )
    if loss_tokens <= 0:
        raise RuntimeError("training schedule has zero supervised tokens")
    observed_ratio = recovery_loss_tokens / loss_tokens
    expected_ratio = ARM_TARGET_RECOVERY_RATIOS[arm]
    if abs(observed_ratio - expected_ratio) > RATIO_TOLERANCE:
        raise RuntimeError(
            f"{arm} recovery supervised-token ratio is {observed_ratio:.6f}; "
            f"expected {expected_ratio:.2f} +/- {RATIO_TOLERANCE:.2f}"
        )
    failed_labels = sum(row["failed_label_message_count"] for row in encoded_rows)
    failed_label_tokens = sum(row["failed_label_tokens"] for row in encoded_rows)
    if arm == "failure_raw" and failed_labels <= 0:
        raise RuntimeError("failure_raw produced no supervised failed-action label")
    if arm != "failure_raw" and failed_labels != 0:
        raise RuntimeError(f"{arm} unexpectedly supervises failed actions")
    return {
        "rows": len(encoded_rows),
        "nonpad_tokens": sum(row["sequence_tokens"] for row in encoded_rows),
        "supervised_tokens": loss_tokens,
        "recovery_rows": sum(row["is_recovery"] for row in encoded_rows),
        "recovery_supervised_tokens": recovery_loss_tokens,
        "realized_recovery_supervised_token_ratio": observed_ratio,
        "expected_recovery_supervised_token_ratio": expected_ratio,
        "failed_action_context_messages": sum(
            row["failed_message_count"] for row in encoded_rows
        ),
        "failed_action_label_messages": failed_labels,
        "failed_action_label_tokens": failed_label_tokens,
        "max_sequence_tokens": max(row["sequence_tokens"] for row in encoded_rows),
        "max_supervised_tokens": max(row["supervised_tokens"] for row in encoded_rows),
    }


def resolve_train_sampler_dataset(trainer: Any, train_dataset: Any = None) -> Any:
    return trainer.train_dataset if train_dataset is None else train_dataset


def tail_logit_contract(sequence_length: int, first_supervised: int) -> tuple[int, int]:
    """Return Qwen's tail-logit count and matching target start.

    To predict the label at position ``s``, causal LM needs the logit at
    position ``s - 1``.  Retaining ``L - s + 1`` final logits and dropping
    their last element therefore aligns exactly with ``labels[:, s:]``.
    """
    if sequence_length <= 1:
        raise ValueError("causal sequence must contain at least two tokens")
    if not 1 <= first_supervised < sequence_length:
        raise ValueError("first supervised token must follow context")
    return sequence_length - first_supervised + 1, first_supervised


def initialize_cuda_peak_tracking(torch_module: Any) -> int:
    device = 0
    torch_module.cuda.init()
    torch_module.cuda.set_device(device)
    torch_module.empty(1, device=f"cuda:{device}")
    torch_module.cuda.synchronize(device)
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats(device)
    return device


def finite_training_audit(log_history: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    loss_values: list[float] = []
    grad_norms: list[float] = []
    eval_losses: list[float] = []
    checked = 0
    for record in [*log_history, metrics]:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise RuntimeError(f"non-finite training metric {key}: {value!r}")
                checked += 1
                if "loss" in key.lower():
                    loss_values.append(numeric)
                if key == "grad_norm":
                    grad_norms.append(numeric)
                if key == "eval_loss":
                    eval_losses.append(numeric)
    if not loss_values:
        raise RuntimeError("training emitted no finite loss")
    if not grad_norms:
        raise RuntimeError("training emitted no finite grad_norm")
    if not eval_losses:
        raise RuntimeError("training emitted no finite validation loss")
    return {
        "finite": True,
        "numeric_values_checked": checked,
        "loss_values_checked": len(loss_values),
        "grad_norm_values_checked": len(grad_norms),
        "validation_loss_values_checked": len(eval_losses),
        "final_train_loss": float(metrics["train_loss"]),
        "final_validation_loss": eval_losses[-1],
    }


def checkpoint_fingerprint(path: Path) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        target = path / filename
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"incomplete adapter checkpoint: {target}")
        file_hash = sha256_file(target)
        file_hashes[filename] = file_hash
        digest.update(filename.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest(), file_hashes


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def require_clean_tracked_source() -> None:
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"repository contains {label} tracked source drift")


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARM_TARGET_RECOVERY_RATIOS), required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--data-hashes", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.expected_source_commit = args.expected_source_commit.lower()
    if COMMIT_RE.fullmatch(args.expected_source_commit) is None:
        raise RuntimeError("--expected-source-commit must be a full lowercase commit")
    if COMMIT_RE.fullmatch(args.model_revision) is None:
        raise RuntimeError("--model-revision must be a full lowercase model commit")
    expected_train_sha = validate_sha256(
        args.expected_train_sha256, "--expected-train-sha256"
    )
    expected_validation_sha = validate_sha256(
        args.expected_validation_sha256, "--expected-validation-sha256"
    )
    require_clean_tracked_source()
    source_commit = git_commit()
    if source_commit != args.expected_source_commit:
        raise RuntimeError(
            f"source commit drift: HEAD={source_commit}, expected={args.expected_source_commit}"
        )
    data_provenance = validate_training_data_provenance(
        arm=args.arm,
        train_file=args.train_file,
        validation_file=args.validation_file,
        data_audit_path=args.data_audit,
        data_hashes_path=args.data_hashes,
        expected_train_sha256=expected_train_sha,
        expected_validation_sha256=expected_validation_sha,
    )
    train_sha = data_provenance["train_file_sha256"]
    validation_sha = data_provenance["validation_file_sha256"]
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise RuntimeError(f"output directory must be absent or empty: {args.output_dir}")

    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cublas_workspace not in (None, ":4096:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or ':4096:8' before importing torch"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import torch.nn.functional as functional
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import SequentialSampler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Stage-1 requires Python 3.12; found {sys.version.split()[0]}"
        )
    frozen_packages = {
        "transformers": "4.52.4",
        "peft": "0.15.2",
        "bitsandbytes": "0.46.0",
        "datasets": "3.6.0",
        "accelerate": "1.7.0",
    }
    for package, expected in frozen_packages.items():
        observed = package_version(package)
        if observed != expected:
            raise RuntimeError(f"Stage-1 requires {package}=={expected}; found {observed}")
    if not str(torch.__version__).startswith("2.7.1"):
        raise RuntimeError(f"Stage-1 requires torch 2.7.1+cu128; found {torch.__version__}")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Stage-1 requires CUDA runtime 12.8; found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("V5 Stage-1 7B QLoRA requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("V5 Stage-1 freezes BF16 compute, but BF16 is unavailable")
    if torch.cuda.get_device_properties(0).total_memory < MIN_GPU_MEMORY_BYTES:
        raise RuntimeError(
            "V5 Stage-1 7B/8192 contract requires a GPU with at least 20 GiB"
        )

    train_rows = read_jsonl(args.train_file)
    validation_rows = read_jsonl(args.validation_file)
    fit_partition_audit = validate_fit_partition(train_rows, validation_rows)
    if len(train_rows) != FORMAL_SCHEDULE_ROWS:
        raise RuntimeError(
            f"formal schedule must contain exactly {FORMAL_SCHEDULE_ROWS} rows; "
            f"found {len(train_rows)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer has no usable pad token")
    tokenizer.padding_side = "right"

    formal_encoded = encode_rows(
        tokenizer,
        train_rows,
        arm=args.arm,
        split="train",
    )
    validation_encoded = encode_rows(
        tokenizer,
        validation_rows,
        arm=None,
        split="validation",
    )
    formal_arm_audit = arm_audit(formal_encoded, args.arm)
    if args.mode == "smoke":
        encoded_rows = sorted(
            formal_encoded,
            key=lambda row: (row["sequence_tokens"], row["id"]),
            reverse=True,
        )[:SMOKE_ROWS]
        effective_steps = SMOKE_STEPS
        effective_grad_accum = SMOKE_GRAD_ACCUM
        effective_validation = sorted(
            validation_encoded,
            key=lambda row: (row["sequence_tokens"], row["id"]),
            reverse=True,
        )[: min(16, len(validation_encoded))]
    else:
        encoded_rows = formal_encoded
        effective_steps = FORMAL_STEPS
        effective_grad_accum = FORMAL_GRAD_ACCUM
        effective_validation = validation_encoded

    set_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = initialize_cuda_peak_tracking(torch)

    def dataset_features(rows: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list(
            [
                {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                    "labels": row["labels"],
                }
                for row in rows
            ]
        )

    class DynamicCompletionCollator:
        def __call__(self, features):
            maximum = max(len(feature["input_ids"]) for feature in features)
            ids, attention, labels = [], [], []
            for feature in features:
                pad = maximum - len(feature["input_ids"])
                ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * pad)
                attention.append(feature["attention_mask"] + [0] * pad)
                labels.append(feature["labels"] + [IGNORE_INDEX] * pad)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )

    class SequentialTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(
                resolve_train_sampler_dataset(self, train_dataset)
            )

    class MaskedCausalTrainer(SequentialTrainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
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
            supervised = labels.ne(IGNORE_INDEX)
            if not bool(supervised.any()):
                raise RuntimeError("SFT microbatch has no supervised tokens")
            first_supervised = int(supervised.nonzero(as_tuple=False)[:, 1].min().item())
            sequence_length = labels.shape[1]
            logits_to_keep, target_start = tail_logit_contract(
                sequence_length,
                first_supervised,
            )
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
            prediction_logits = outputs.logits[:, :-1, :]
            targets = labels[:, target_start:]
            if prediction_logits.shape[:2] != targets.shape:
                raise RuntimeError(
                    "tail-logit alignment failure: "
                    f"{tuple(prediction_logits.shape[:2])} != {tuple(targets.shape)}"
                )
            loss = functional.cross_entropy(
                prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite SFT loss: {loss.detach().item()!r}")
            return (loss, outputs) if return_outputs else loss

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join(shlex.quote(argument) for argument in [sys.executable, *sys.argv])
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_state"),
        max_steps=effective_steps,
        per_device_train_batch_size=FORMAL_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=effective_grad_accum,
        learning_rate=FORMAL_LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=effective_steps,
        save_strategy="no",
        report_to=[],
        bf16=True,
        fp16=False,
        tf32=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        label_names=["labels"],
    )
    trainer = MaskedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_features(encoded_rows),
        eval_dataset=dataset_features(effective_validation),
        data_collator=DynamicCompletionCollator(),
    )
    result = trainer.train()
    loss_audit = finite_training_audit(trainer.state.log_history, result.metrics)
    checkpoint = args.output_dir / "checkpoint_final"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    checkpoint_hash, checkpoint_files = checkpoint_fingerprint(checkpoint)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "bf16": True,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    manifest = {
        "protocol": "v5_stage1_message_masked_sft_7b",
        "source_commit": source_commit,
        "arm": args.arm,
        "mode": args.mode,
        "objective": "message_masked_causal_language_model_cross_entropy",
        "model": MODEL,
        "model_revision": args.model_revision,
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "seed": SEED,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "truncation": False,
        "formal_steps": FORMAL_STEPS,
        "formal_batch_size": FORMAL_BATCH_SIZE,
        "formal_grad_accum": FORMAL_GRAD_ACCUM,
        "effective_steps": effective_steps,
        "effective_grad_accum": effective_grad_accum,
        "learning_rate": FORMAL_LEARNING_RATE,
        "train_file": str(args.train_file),
        "train_file_sha256": train_sha,
        "validation_file": str(args.validation_file),
        "validation_file_sha256": validation_sha,
        "data_provenance": data_provenance,
        "formal_schedule": formal_arm_audit,
        "fit_partition": fit_partition_audit,
        "effective_rows": len(encoded_rows),
        "effective_validation_rows": len(effective_validation),
        "loss_audit": loss_audit,
        "held_out_test_accessed": False,
        "checkpoint": {
            "path": str(checkpoint),
            "fingerprint": checkpoint_hash,
            "file_sha256": checkpoint_files,
        },
        "environment": environment,
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(result.metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "arm": args.arm,
                "mode": args.mode,
                "checkpoint": str(checkpoint),
                "metrics": result.metrics,
                "environment": environment,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
