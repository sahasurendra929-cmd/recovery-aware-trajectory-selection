#!/usr/bin/env python3
"""Auditable V6 directional-screen SFT trainer.

This trainer consumes the JSONL emitted by ``materialize_v6_sft.py``.  It is
deliberately independent of V5 arm semantics: a selected fresh recovery suffix
may contain both assistant tool calls and assistant text.  The controlled
failed call and its error result are context only and can never enter the
positive causal-LM labels.

All heavyweight ML imports live in ``main``.  Contract validation and
token-span construction can therefore be tested on a CPU-only coordinator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "v6_directional_sft_training_v1"
MANIFEST_PROTOCOL = "v6_matched_task_selector_manifests_v1"
SOURCE_BINDING_PROTOCOL = "v6_selector_manifest_training_binding_v1"
MATERIALIZATION_PROTOCOL = "v6_sft_materialization_v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TOKENIZER_REVISION = MODEL_REVISION
ARMS = ("flawless_only", "random_stratified", "full_proposed")
TRAIN_SEED = 20260722
RANDOM_SELECTION_SEED = 20260806
IGNORE_INDEX = -100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ROLES = {"system", "user", "assistant", "tool"}
ROOT = Path(__file__).resolve().parents[1]


class V6TrainingError(RuntimeError):
    """A frozen V6 training or provenance contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise V6TrainingError(f"{label} must be a lowercase SHA-256")
    return value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def model_config_fingerprint(config: Any) -> str:
    """Hash the architecture/config separately from the pinned weight revision."""
    payload = dict(config.to_dict())
    for unstable in ("_name_or_path", "_commit_hash", "transformers_version"):
        payload.pop(unstable, None)
    return canonical_sha256(_json_safe(payload))


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Hash all token IDs and rendering state that define V6 token contracts."""
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise V6TrainingError("tokenizer vocabulary is empty")
    payload = {
        "vocab": sorted((str(token), int(index)) for token, index in vocab.items()),
        "added_vocab": sorted(
            (str(token), int(index))
            for token, index in tokenizer.get_added_vocab().items()
        ),
        "special_tokens_map": _json_safe(tokenizer.special_tokens_map),
        "chat_template": tokenizer.chat_template,
    }
    return canonical_sha256(payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise V6TrainingError(f"training JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise V6TrainingError(f"{path}:{line_number}: blank JSONL line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise V6TrainingError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(row, dict):
                raise V6TrainingError(
                    f"{path}:{line_number}: row must be an object"
                )
            rows.append(row)
    if not rows:
        raise V6TrainingError("training JSONL is empty")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(row_id, str) or not row_id for row_id in ids):
        raise V6TrainingError("every training row needs a non-empty id")
    if len(ids) != len(set(ids)):
        raise V6TrainingError("training row ids are not unique")
    return rows


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise V6TrainingError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6TrainingError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise V6TrainingError(f"{label} root must be an object: {path}")
    return value


def _content_text(value: Any, *, where: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        # Candidate measurement deliberately reuses the frozen V5
        # normalization contract.  τ² tool results are commonly structured
        # JSON objects, so canonicalize both objects and arrays identically
        # here or the trainer would reject valid materialized rows (and, for
        # arrays, could silently retokenize a different string).
        return canonical(value)
    raise V6TrainingError(f"{where}.content has an unsupported shape")


def _normalize_tool_call(call: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise V6TrainingError(f"{where} must be an object")
    function = call.get("function")
    if function is not None:
        if not isinstance(function, dict):
            raise V6TrainingError(f"{where}.function must be an object")
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise V6TrainingError(f"{where}.name must be non-empty")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise V6TrainingError(
                f"{where}.arguments is not JSON"
            ) from error
    if not isinstance(arguments, dict):
        raise V6TrainingError(f"{where}.arguments must be an object")
    normalized: dict[str, Any] = {
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    call_id = call.get("id")
    if call_id is not None:
        if not isinstance(call_id, str) or not call_id:
            raise V6TrainingError(f"{where}.id must be non-empty")
        normalized["id"] = call_id
    return normalized


def normalize_message(message: Any, *, where: str) -> dict[str, Any]:
    """Normalize compact τ²/OpenAI messages for Qwen's chat template."""
    if not isinstance(message, dict):
        raise V6TrainingError(f"{where} must be an object")
    role = message.get("role")
    if role not in ROLES:
        raise V6TrainingError(f"{where}.role is unsupported: {role!r}")
    calls = message.get("tool_calls") or []
    if calls and (role != "assistant" or not isinstance(calls, list)):
        raise V6TrainingError(
            f"{where}: only assistant messages may contain tool calls"
        )
    if len(calls) > 1:
        raise V6TrainingError(f"{where}: V6 freezes one tool call per turn")
    normalized: dict[str, Any] = {
        "role": role,
        "content": _content_text(message.get("content"), where=where),
    }
    if calls:
        normalized["tool_calls"] = [
            _normalize_tool_call(calls[0], where=f"{where}.tool_calls[0]")
        ]
    if role == "assistant" and not normalized["content"] and not calls:
        raise V6TrainingError(
            f"{where}: assistant message has neither text nor a tool call"
        )
    if role == "tool":
        call_id = message.get("tool_call_id", message.get("id"))
        if call_id is not None:
            if not isinstance(call_id, str) or not call_id:
                raise V6TrainingError(f"{where}.tool_call_id must be non-empty")
            normalized["tool_call_id"] = call_id
    return normalized


def _recursive_official_test_true(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "official_test_used" and item is not False:
                return True
            if _recursive_official_test_true(item):
                return True
    elif isinstance(value, list):
        return any(_recursive_official_test_true(item) for item in value)
    return False


def failed_assistant_indices(
    messages: Sequence[dict[str, Any]], *, row_id: str
) -> list[int]:
    failed: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        if (
            index == 0
            or messages[index - 1].get("role") != "assistant"
            or not messages[index - 1].get("tool_calls")
        ):
            raise V6TrainingError(f"{row_id}: orphan tool result at {index}")
        if type(message.get("error")) is not bool:
            raise V6TrainingError(
                f"{row_id}: tool result {index} needs explicit boolean error"
            )
        if message["error"]:
            failed.append(index - 1)
    return failed


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V6TrainingError(f"{label} must be a non-empty string")
    return value


def _validate_row_source_provenance(
    *,
    row_id: str,
    metadata: Mapping[str, Any],
    arm: str,
) -> dict[str, Any]:
    """Validate provenance duplicated within one materialized training row."""
    manifest_sha256 = checked_sha256(
        metadata.get("selector_manifest_sha256"),
        f"{row_id}: metadata.selector_manifest_sha256",
    )
    pool_sha256 = checked_sha256(
        metadata.get("candidate_pool_sha256"),
        f"{row_id}: metadata.candidate_pool_sha256",
    )
    source_trajectory_sha256 = checked_sha256(
        metadata.get("source_trajectory_sha256"),
        f"{row_id}: metadata.source_trajectory_sha256",
    )
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise V6TrainingError(f"{row_id}: metadata.source_hashes must be an object")
    if source_hashes.get("selector_manifest_sha256") != manifest_sha256:
        raise V6TrainingError(
            f"{row_id}: source_hashes selector manifest hash drift"
        )
    if source_hashes.get("candidate_pool_sha256") != pool_sha256:
        raise V6TrainingError(f"{row_id}: source_hashes candidate pool hash drift")

    task_identity = _nonempty_string(
        metadata.get("task_identity"),
        label=f"{row_id}: metadata.task_identity",
    )
    domain = _nonempty_string(
        metadata.get("domain"), label=f"{row_id}: metadata.domain"
    )
    task_id = _nonempty_string(
        metadata.get("task_id"), label=f"{row_id}: metadata.task_id"
    )
    if task_identity != f"{domain}:{task_id}":
        raise V6TrainingError(f"{row_id}: task identity/domain/task_id drift")
    if metadata.get("source_example_id") != row_id:
        raise V6TrainingError(f"{row_id}: source_example_id drift")
    if metadata.get("source_split") != "inner_train":
        raise V6TrainingError(f"{row_id}: source split drift")
    if metadata.get("fit_split") != "train_schedule":
        raise V6TrainingError(f"{row_id}: fit split drift")

    if arm == "flawless_only":
        expected_hash_keys = {
            "selector_manifest_sha256",
            "candidate_pool_sha256",
            "clean_view_sha256",
            "source_candidate_pair_ids_sha256",
        }
        if set(source_hashes) != expected_hash_keys:
            raise V6TrainingError(f"{row_id}: flawless source_hashes shape drift")
        clean_sha256 = checked_sha256(
            source_hashes.get("clean_view_sha256"),
            f"{row_id}: source_hashes.clean_view_sha256",
        )
        source_pair_ids_sha256 = checked_sha256(
            source_hashes.get("source_candidate_pair_ids_sha256"),
            f"{row_id}: source_hashes.source_candidate_pair_ids_sha256",
        )
        clean_id = _nonempty_string(
            metadata.get("clean_id"), label=f"{row_id}: metadata.clean_id"
        )
        if metadata.get("arm") != "v6_flawless_control":
            raise V6TrainingError(f"{row_id}: materialized trainer arm drift")
        if metadata.get("candidate_pair_id") is not None:
            raise V6TrainingError(f"{row_id}: flawless row has candidate_pair_id")
        if metadata.get("branch_id") is not None:
            raise V6TrainingError(f"{row_id}: flawless row has branch_id")
        if metadata.get("source_pair_id") != clean_id:
            raise V6TrainingError(f"{row_id}: flawless source_pair_id drift")
        if metadata.get("trial") != clean_id:
            raise V6TrainingError(f"{row_id}: flawless trial drift")
        if row_id != f"v6:{arm}:{clean_id}":
            raise V6TrainingError(f"{row_id}: flawless row id drift")
        if source_trajectory_sha256 != clean_sha256:
            raise V6TrainingError(
                f"{row_id}: clean source trajectory hash drift"
            )
        source_pair_ids = metadata.get("source_candidate_pair_ids")
        if (
            not isinstance(source_pair_ids, list)
            or any(not isinstance(value, str) or not value for value in source_pair_ids)
            or source_pair_ids != sorted(source_pair_ids)
            or len(source_pair_ids) != len(set(source_pair_ids))
        ):
            raise V6TrainingError(
                f"{row_id}: source_candidate_pair_ids must be sorted and unique"
            )
        if canonical_sha256(source_pair_ids) != source_pair_ids_sha256:
            raise V6TrainingError(
                f"{row_id}: source_candidate_pair_ids hash drift"
            )
        return {
            "selector_manifest_sha256": manifest_sha256,
            "candidate_pool_sha256": pool_sha256,
            "task_identity": task_identity,
            "domain": domain,
            "clean_id": clean_id,
            "clean_view_sha256": clean_sha256,
            "source_candidate_pair_ids": list(source_pair_ids),
        }

    expected_hash_keys = {
        "selector_manifest_sha256",
        "candidate_pool_sha256",
        "candidate_pair_sha256",
        "branch_sha256",
        "prefix_sha256",
        "environment_snapshot_sha256",
    }
    if set(source_hashes) != expected_hash_keys:
        raise V6TrainingError(f"{row_id}: recovery source_hashes shape drift")
    candidate_pair_sha256 = checked_sha256(
        source_hashes.get("candidate_pair_sha256"),
        f"{row_id}: source_hashes.candidate_pair_sha256",
    )
    branch_sha256 = checked_sha256(
        source_hashes.get("branch_sha256"),
        f"{row_id}: source_hashes.branch_sha256",
    )
    prefix_sha256 = checked_sha256(
        source_hashes.get("prefix_sha256"),
        f"{row_id}: source_hashes.prefix_sha256",
    )
    environment_sha256 = checked_sha256(
        source_hashes.get("environment_snapshot_sha256"),
        f"{row_id}: source_hashes.environment_snapshot_sha256",
    )
    candidate_pair_id = _nonempty_string(
        metadata.get("candidate_pair_id"),
        label=f"{row_id}: metadata.candidate_pair_id",
    )
    branch_id = _nonempty_string(
        metadata.get("branch_id"), label=f"{row_id}: metadata.branch_id"
    )
    if metadata.get("arm") != "v6_recovery_selected":
        raise V6TrainingError(f"{row_id}: materialized trainer arm drift")
    if metadata.get("clean_id") is not None:
        raise V6TrainingError(f"{row_id}: recovery row has clean_id")
    if metadata.get("source_pair_id") != candidate_pair_id:
        raise V6TrainingError(f"{row_id}: recovery source_pair_id drift")
    if metadata.get("trial") != branch_id:
        raise V6TrainingError(f"{row_id}: recovery trial drift")
    if row_id != f"v6:{arm}:{candidate_pair_id}:{branch_id}":
        raise V6TrainingError(f"{row_id}: recovery row id drift")
    if source_trajectory_sha256 != branch_sha256:
        raise V6TrainingError(f"{row_id}: branch source trajectory hash drift")
    return {
        "selector_manifest_sha256": manifest_sha256,
        "candidate_pool_sha256": pool_sha256,
        "task_identity": task_identity,
        "domain": domain,
        "candidate_pair_id": candidate_pair_id,
        "branch_id": branch_id,
        "candidate_pair_sha256": candidate_pair_sha256,
        "branch_sha256": branch_sha256,
        "prefix_sha256": prefix_sha256,
        "environment_snapshot_sha256": environment_sha256,
    }


def validate_row(
    row: dict[str, Any],
    *,
    arm: str,
    tokenizer_name: str = MODEL_ID,
    tokenizer_revision: str = TOKENIZER_REVISION,
) -> dict[str, Any]:
    """Fail closed on leakage, failed labels, arm drift, and schema drift."""
    required = {"id", "messages", "label_mask", "metadata", "token_contract"}
    if set(row) != required:
        raise V6TrainingError(
            f"{row.get('id', '<unknown>')}: row keys must be {sorted(required)}"
        )
    row_id = row["id"]
    if not isinstance(row_id, str) or not row_id:
        raise V6TrainingError("row id must be non-empty")
    if arm not in ARMS:
        raise V6TrainingError(f"unsupported V6 arm: {arm}")
    messages = row["messages"]
    mask = row["label_mask"]
    metadata = row["metadata"]
    contract = row["token_contract"]
    if not isinstance(messages, list) or not messages:
        raise V6TrainingError(f"{row_id}: messages must be non-empty")
    if (
        not isinstance(mask, list)
        or len(mask) != len(messages)
        or any(type(value) is not bool for value in mask)
    ):
        raise V6TrainingError(
            f"{row_id}: label_mask must be one boolean per message"
        )
    if not isinstance(metadata, dict):
        raise V6TrainingError(f"{row_id}: metadata must be an object")
    if _recursive_official_test_true(row):
        raise V6TrainingError(f"{row_id}: official test use is forbidden")
    if metadata.get("official_test_used") is not False:
        raise V6TrainingError(
            f"{row_id}: metadata.official_test_used must be false"
        )
    if metadata.get("selector") != arm or metadata.get("paper_arm") != arm:
        raise V6TrainingError(f"{row_id}: selector/arm provenance drift")
    if (
        arm == "random_stratified"
        and metadata.get("selection_seed") != RANDOM_SELECTION_SEED
    ):
        raise V6TrainingError(
            f"{row_id}: random arm must use frozen selection seed "
            f"{RANDOM_SELECTION_SEED}"
        )
    if metadata.get("materialization_protocol") != MATERIALIZATION_PROTOCOL:
        raise V6TrainingError(f"{row_id}: materialization protocol drift")
    if metadata.get("tokenizer_name") != tokenizer_name:
        raise V6TrainingError(f"{row_id}: materialized tokenizer name drift")
    if metadata.get("tokenizer_revision") != tokenizer_revision:
        raise V6TrainingError(f"{row_id}: materialized tokenizer revision drift")
    source_provenance = _validate_row_source_provenance(
        row_id=row_id,
        metadata=metadata,
        arm=arm,
    )
    if messages[0].get("role") != "system":
        raise V6TrainingError(f"{row_id}: first message must be system")
    if any(message.get("role") == "system" for message in messages[1:]):
        raise V6TrainingError(f"{row_id}: system message appears after index zero")

    normalized = [
        normalize_message(message, where=f"{row_id}.messages[{index}]")
        for index, message in enumerate(messages)
    ]
    selected = [index for index, value in enumerate(mask) if value]
    if not selected:
        raise V6TrainingError(f"{row_id}: no supervised assistant messages")
    for index in selected:
        if messages[index].get("role") != "assistant":
            raise V6TrainingError(
                f"{row_id}: non-assistant message {index} is labeled"
            )
    failed = failed_assistant_indices(messages, row_id=row_id)
    if set(failed) & set(selected):
        raise V6TrainingError(f"{row_id}: a failed call is positively labeled")
    if metadata.get("failed_assistant_message_indices") != failed:
        raise V6TrainingError(f"{row_id}: recorded failed-call indices drift")
    if metadata.get("failed_action_label_messages") != 0:
        raise V6TrainingError(f"{row_id}: failed-action label count is not zero")

    if arm == "flawless_only":
        if failed or metadata.get("source") != "perfect_success":
            raise V6TrainingError(f"{row_id}: flawless arm contains an error")
    else:
        if len(failed) != 1 or metadata.get("source") != "failure_rich":
            raise V6TrainingError(
                f"{row_id}: recovery arm needs one controlled real error"
            )
        error_result_index = failed[0] + 1
        if any(index <= error_result_index for index in selected):
            raise V6TrainingError(
                f"{row_id}: only the fresh suffix after the error may be labeled"
            )
        if metadata.get("future_clean_suffix_used") is not False:
            raise V6TrainingError(f"{row_id}: future clean suffix leakage")
        suffix_origin = metadata.get("recovery_suffix_origin")
        if suffix_origin == "sanitized_deterministic_reference_plan":
            if (
                metadata.get("fresh_recovery_suffix") is not False
                or metadata.get("gold_reference_suffix_used") is not True
            ):
                raise V6TrainingError(
                    f"{row_id}: sanitized-oracle suffix provenance drift"
                )
        elif suffix_origin in (None, "fresh_teacher_generation"):
            # ``None`` preserves the frozen legacy V6 materialization schema.
            if (
                metadata.get("fresh_recovery_suffix") is not True
                or metadata.get("gold_reference_suffix_used") is True
            ):
                raise V6TrainingError(
                    f"{row_id}: fresh recovery suffix provenance drift"
                )
        else:
            raise V6TrainingError(
                f"{row_id}: unsupported recovery suffix origin {suffix_origin!r}"
            )

    tools = metadata.get("tool_schemas")
    if (
        not isinstance(tools, list)
        or not tools
        or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise V6TrainingError(f"{row_id}: tool_schemas must be non-empty")
    if metadata.get("tool_schemas_sha256") != canonical_sha256(tools):
        raise V6TrainingError(f"{row_id}: tool schema hash drift")
    contract_input = {
        "messages": messages,
        "label_mask": mask,
        "tool_schemas": tools,
    }
    if metadata.get("token_contract_input_sha256") != canonical_sha256(
        contract_input
    ):
        raise V6TrainingError(f"{row_id}: token-contract input hash drift")
    if not isinstance(contract, dict) or set(contract) != {
        "sequence_tokens",
        "supervised_tokens",
        "label_spans",
    }:
        raise V6TrainingError(f"{row_id}: token_contract shape drift")
    return {
        "id": row_id,
        "messages": normalized,
        "raw_messages": messages,
        "label_mask": list(mask),
        "metadata": dict(metadata),
        "token_contract": dict(contract),
        "failed_message_indices": failed,
        "source_provenance": source_provenance,
    }


def validate_dataset_provenance(
    validated_rows: Sequence[dict[str, Any]],
    selector_manifest: Mapping[str, Any],
    *,
    arm: str,
    expected_selector_manifest_sha256: str,
    observed_selector_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind every materialized row to one exact selector manifest.

    This closes the gap left by hashing only the training JSONL: task
    membership, complete recovery-pair atoms, and per-source hashes are
    independently recovered from the frozen upstream manifest.
    """
    if not validated_rows:
        raise V6TrainingError("materialized dataset is empty")
    expected_manifest_sha256 = checked_sha256(
        expected_selector_manifest_sha256,
        "--expected-selector-manifest-sha256",
    )
    observed_manifest_sha256 = checked_sha256(
        observed_selector_manifest_sha256,
        "selector manifest file SHA-256",
    )
    if observed_manifest_sha256 != expected_manifest_sha256:
        raise V6TrainingError("selector manifest file SHA-256 drift")
    if selector_manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise V6TrainingError("selector manifest protocol drift")
    if selector_manifest.get("official_test_used") is not False:
        raise V6TrainingError("selector manifest official-test marker drift")
    if _recursive_official_test_true(selector_manifest):
        raise V6TrainingError("selector manifest contains official test use")
    if selector_manifest.get("selector") != arm:
        raise V6TrainingError("selector manifest arm drift")

    pool_sha256 = checked_sha256(
        selector_manifest.get("candidate_pool_sha256"),
        "selector manifest candidate_pool_sha256",
    )
    for row in validated_rows:
        provenance = row["source_provenance"]
        if provenance["selector_manifest_sha256"] != expected_manifest_sha256:
            raise V6TrainingError(
                f"{row['id']}: row is not bound to the expected selector manifest"
            )
        if provenance["candidate_pool_sha256"] != pool_sha256:
            raise V6TrainingError(
                f"{row['id']}: row candidate pool differs from selector manifest"
            )
        if canonical(row["metadata"].get("selection_seed")) != canonical(
            selector_manifest.get("selection_seed")
        ):
            raise V6TrainingError(f"{row['id']}: selection seed differs from manifest")

    matched = selector_manifest.get("matched_task_ids")
    if (
        not isinstance(matched, list)
        or not matched
        or any(not isinstance(value, str) or not value for value in matched)
        or len(matched) != len(set(matched))
    ):
        raise V6TrainingError(
            "selector manifest matched_task_ids must be non-empty and unique"
        )
    selected = selector_manifest.get("selected")
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(record, Mapping) for record in selected)
    ):
        raise V6TrainingError("selector manifest selected records are malformed")
    selected_tasks = [
        _nonempty_string(
            record.get("task_identity"),
            label="selector manifest selected.task_identity",
        )
        for record in selected
    ]
    if (
        len(selected_tasks) != len(set(selected_tasks))
        or set(selected_tasks) != set(matched)
    ):
        raise V6TrainingError(
            "selector manifest selected tasks differ from matched_task_ids"
        )
    row_tasks = {
        row["source_provenance"]["task_identity"] for row in validated_rows
    }
    if row_tasks != set(matched):
        raise V6TrainingError(
            "materialized tasks differ from selector manifest matched_task_ids"
        )

    if arm == "flawless_only":
        clean_records: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for record in selected:
            view = record.get("clean_view")
            if not isinstance(view, Mapping):
                raise V6TrainingError(
                    f"{record.get('task_identity')}: manifest clean_view is missing"
                )
            clean_id = _nonempty_string(
                view.get("clean_id"),
                label="selector manifest clean_view.clean_id",
            )
            if record.get("clean_id") != clean_id:
                raise V6TrainingError(f"{clean_id}: selected clean_id drift")
            if clean_id in clean_records:
                raise V6TrainingError("selector manifest clean IDs are not unique")
            clean_records[clean_id] = (record, view)
        observed_clean_ids: set[str] = set()
        observed_tasks: set[str] = set()
        for row in validated_rows:
            provenance = row["source_provenance"]
            clean_id = provenance["clean_id"]
            if clean_id not in clean_records:
                raise V6TrainingError(
                    f"{row['id']}: clean source is absent from selector manifest"
                )
            if clean_id in observed_clean_ids:
                raise V6TrainingError(
                    f"{clean_id}: flawless source was materialized more than once"
                )
            observed_clean_ids.add(clean_id)
            record, view = clean_records[clean_id]
            task_identity = provenance["task_identity"]
            if (
                record.get("task_identity") != task_identity
                or record.get("domain") != provenance["domain"]
            ):
                raise V6TrainingError(f"{clean_id}: clean task/domain drift")
            if task_identity in observed_tasks:
                raise V6TrainingError(
                    f"{task_identity}: flawless task was materialized more than once"
                )
            observed_tasks.add(task_identity)
            if canonical_sha256(view) != provenance["clean_view_sha256"]:
                raise V6TrainingError(f"{clean_id}: clean_view source hash drift")
            source_pair_ids = view.get("source_candidate_pair_ids", [])
            if (
                not isinstance(source_pair_ids, list)
                or any(
                    not isinstance(value, str) or not value
                    for value in source_pair_ids
                )
                or sorted(source_pair_ids)
                != provenance["source_candidate_pair_ids"]
            ):
                raise V6TrainingError(
                    f"{clean_id}: clean source candidate-pair provenance drift"
                )
        if observed_clean_ids != set(clean_records):
            raise V6TrainingError(
                "materialized flawless rows do not cover every selected clean view"
            )
        candidate_pairs = 0
        expected_rows = len(matched)
    else:
        pair_records: dict[
            str,
            tuple[
                Mapping[str, Any],
                Mapping[str, Any],
                dict[str, Mapping[str, Any]],
            ],
        ] = {}
        for record in selected:
            pair = record.get("candidate_pair")
            if not isinstance(pair, Mapping):
                raise V6TrainingError(
                    f"{record.get('task_identity')}: complete candidate_pair is missing"
                )
            pair_id = _nonempty_string(
                pair.get("candidate_pair_id"),
                label="selector manifest candidate_pair.candidate_pair_id",
            )
            if pair_id in pair_records:
                raise V6TrainingError(
                    "selector manifest candidate pair IDs are not unique"
                )
            for field in ("candidate_pair_id", "task_identity", "domain"):
                if record.get(field) != pair.get(field):
                    raise V6TrainingError(
                        f"{pair_id}: selected-record {field} differs from pair"
                    )
            branches = pair.get("branches")
            if (
                not isinstance(branches, list)
                or len(branches) != 2
                or any(not isinstance(branch, Mapping) for branch in branches)
            ):
                raise V6TrainingError(
                    f"{pair_id}: selector atom must contain two sibling branches"
                )
            branches_by_id: dict[str, Mapping[str, Any]] = {}
            for branch in branches:
                branch_id = _nonempty_string(
                    branch.get("branch_id"),
                    label=f"{pair_id}: branch_id",
                )
                if branch_id in branches_by_id:
                    raise V6TrainingError(
                        f"{pair_id}: selector branch IDs are not unique"
                    )
                branches_by_id[branch_id] = branch
            declared_branch_ids = record.get("branch_ids")
            if (
                not isinstance(declared_branch_ids, list)
                or any(
                    not isinstance(value, str) or not value
                    for value in declared_branch_ids
                )
                or sorted(declared_branch_ids) != sorted(branches_by_id)
                or len(declared_branch_ids) != len(set(declared_branch_ids))
            ):
                raise V6TrainingError(
                    f"{pair_id}: selected-record branch_ids drift"
                )
            pair_records[pair_id] = (record, pair, branches_by_id)

        rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in validated_rows:
            pair_id = row["source_provenance"]["candidate_pair_id"]
            rows_by_pair[pair_id].append(row)
        if set(rows_by_pair) != set(pair_records):
            raise V6TrainingError(
                "materialized candidate pairs differ from selector manifest"
            )
        for pair_id, (record, pair, branches_by_id) in pair_records.items():
            pair_rows = rows_by_pair[pair_id]
            if len(pair_rows) != 2:
                raise V6TrainingError(
                    f"{pair_id}: complete two-branch candidate pair is required"
                )
            observed_branch_ids = [
                row["source_provenance"]["branch_id"] for row in pair_rows
            ]
            if (
                len(observed_branch_ids) != len(set(observed_branch_ids))
                or set(observed_branch_ids) != set(branches_by_id)
            ):
                raise V6TrainingError(
                    f"{pair_id}: materialized sibling branches are incomplete"
                )
            pair_sha256 = canonical_sha256(pair)
            prefix_sha256 = checked_sha256(
                pair.get("prefix_sha256"), f"{pair_id}: prefix_sha256"
            )
            environment_sha256 = checked_sha256(
                pair.get("environment_snapshot_sha256"),
                f"{pair_id}: environment_snapshot_sha256",
            )
            for row in pair_rows:
                provenance = row["source_provenance"]
                if (
                    provenance["task_identity"] != record.get("task_identity")
                    or provenance["domain"] != record.get("domain")
                ):
                    raise V6TrainingError(f"{pair_id}: materialized task/domain drift")
                if provenance["candidate_pair_sha256"] != pair_sha256:
                    raise V6TrainingError(
                        f"{pair_id}: candidate_pair source hash drift"
                    )
                if provenance["prefix_sha256"] != prefix_sha256:
                    raise V6TrainingError(f"{pair_id}: prefix source hash drift")
                if provenance["environment_snapshot_sha256"] != environment_sha256:
                    raise V6TrainingError(
                        f"{pair_id}: environment snapshot source hash drift"
                    )
                branch = branches_by_id[provenance["branch_id"]]
                if provenance["branch_sha256"] != canonical_sha256(branch):
                    raise V6TrainingError(
                        f"{provenance['branch_id']}: branch source hash drift"
                    )
        candidate_pairs = len(pair_records)
        expected_rows = 2 * len(matched)

    if len(validated_rows) != expected_rows:
        raise V6TrainingError(
            "materialized row count does not cover every matched selector atom"
        )
    return {
        "protocol": SOURCE_BINDING_PROTOCOL,
        "status": "PASS",
        "selector_manifest_sha256": expected_manifest_sha256,
        "candidate_pool_sha256": pool_sha256,
        "selector": arm,
        "selection_seed": selector_manifest.get("selection_seed"),
        "matched_task_ids": sorted(matched),
        "matched_task_ids_sha256": canonical_sha256(sorted(matched)),
        "tasks": len(matched),
        "candidate_pairs": candidate_pairs,
        "rows": len(validated_rows),
        "expected_rows": expected_rows,
        "candidate_pair_atom_split": False,
        "source_hashes_recomputed_against_manifest": True,
        "official_test_used": False,
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
        raise V6TrainingError(
            "tokenizer chat template did not return a flat integer list"
        )
    return result


def encode_row(
    tokenizer: Any,
    validated: dict[str, Any],
    *,
    max_seq_len: int,
) -> dict[str, Any]:
    """Retokenize, reproduce every label span, and forbid truncation."""
    row_id = validated["id"]
    messages = validated["messages"]
    tools = validated["metadata"]["tool_schemas"]
    full_ids = _token_ids(
        tokenizer, messages, tools=tools, generation=False
    )
    if not full_ids:
        raise V6TrainingError(f"{row_id}: tokenizer emitted no tokens")
    if len(full_ids) > max_seq_len:
        raise V6TrainingError(
            f"{row_id}: {len(full_ids)} tokens exceed max_seq_len={max_seq_len}; "
            "V6 forbids truncation"
        )
    labels = [IGNORE_INDEX] * len(full_ids)
    spans: list[dict[str, int]] = []
    for message_index, selected in enumerate(validated["label_mask"]):
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
            raise V6TrainingError(
                f"{row_id}: empty label span at message {message_index}"
            )
        if through[: len(before)] != before:
            raise V6TrainingError(
                f"{row_id}: chat template is not prefix-stable at {message_index}"
            )
        if full_ids[: len(through)] != through:
            raise V6TrainingError(
                f"{row_id}: future messages altered prefix tokenization"
            )
        start, end = len(before), len(through)
        if any(value != IGNORE_INDEX for value in labels[start:end]):
            raise V6TrainingError(f"{row_id}: overlapping label spans")
        labels[start:end] = full_ids[start:end]
        spans.append(
            {
                "message_index": message_index,
                "token_start": start,
                "token_end": end,
            }
        )
    supervised = sum(value != IGNORE_INDEX for value in labels)
    expected = validated["token_contract"]
    observed = {
        "sequence_tokens": len(full_ids),
        "supervised_tokens": supervised,
        "label_spans": spans,
    }
    if observed != expected:
        raise V6TrainingError(
            f"{row_id}: retokenized token_contract does not match materialization"
        )
    if supervised <= 0:
        raise V6TrainingError(f"{row_id}: no supervised tokens")
    first_supervised = next(
        index for index, value in enumerate(labels) if value != IGNORE_INDEX
    )
    if first_supervised == 0:
        raise V6TrainingError(f"{row_id}: causal label has no preceding context")
    failed = set(validated["failed_message_indices"])
    return {
        "id": row_id,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "sequence_tokens": len(full_ids),
        "supervised_tokens": supervised,
        "selected_assistant_text_messages": sum(
            validated["raw_messages"][index].get("role") == "assistant"
            and not validated["raw_messages"][index].get("tool_calls")
            for index, selected in enumerate(validated["label_mask"])
            if selected
        ),
        "selected_assistant_tool_messages": sum(
            bool(validated["raw_messages"][index].get("tool_calls"))
            for index, selected in enumerate(validated["label_mask"])
            if selected
        ),
        "failed_context_messages": len(failed),
    }


def encode_rows(
    tokenizer: Any,
    rows: Iterable[dict[str, Any]],
    *,
    arm: str,
    max_seq_len: int,
    tokenizer_name: str = MODEL_ID,
    tokenizer_revision: str = TOKENIZER_REVISION,
) -> list[dict[str, Any]]:
    return [
        encode_row(
            tokenizer,
            validate_row(
                row,
                arm=arm,
                tokenizer_name=tokenizer_name,
                tokenizer_revision=tokenizer_revision,
            ),
            max_seq_len=max_seq_len,
        )
        for row in rows
    ]


def data_audit(
    encoded: Sequence[dict[str, Any]],
    *,
    arm: str,
    train_file_sha256: str,
    max_seq_len: int,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if not encoded:
        raise V6TrainingError("encoded training set is empty")
    return {
        "protocol": PROTOCOL,
        "status": "PASS",
        "arm": arm,
        "train_file_sha256": train_file_sha256,
        "rows": len(encoded),
        "sequence_tokens": sum(row["sequence_tokens"] for row in encoded),
        "supervised_tokens": sum(row["supervised_tokens"] for row in encoded),
        "max_sequence_tokens_observed": max(
            row["sequence_tokens"] for row in encoded
        ),
        "max_sequence_tokens_allowed": max_seq_len,
        "failed_calls_as_context": sum(
            row["failed_context_messages"] for row in encoded
        ),
        "failed_call_positive_labels": 0,
        "assistant_tool_messages_supervised": sum(
            row["selected_assistant_tool_messages"] for row in encoded
        ),
        "assistant_text_messages_supervised": sum(
            row["selected_assistant_text_messages"] for row in encoded
        ),
        "token_contracts_retokenized": len(encoded),
        "truncation_used": False,
        "official_test_used": False,
        "source_provenance": dict(source_provenance),
    }


def tail_logit_contract(
    sequence_length: int, first_supervised: int
) -> tuple[int, int]:
    if sequence_length <= 1:
        raise ValueError("causal sequence must contain at least two tokens")
    if not 1 <= first_supervised < sequence_length:
        raise ValueError("first supervised token must follow context")
    return sequence_length - first_supervised + 1, first_supervised


def finite_training_audit(
    log_history: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, Any]:
    losses: list[float] = []
    grad_norms: list[float] = []
    checked = 0
    for record in [*log_history, metrics]:
        for key, value in record.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise V6TrainingError(
                        f"non-finite training metric {key}: {value!r}"
                    )
                checked += 1
                if "loss" in key.lower():
                    losses.append(numeric)
                if key == "grad_norm":
                    grad_norms.append(numeric)
    if not losses:
        raise V6TrainingError("training emitted no finite loss")
    if not grad_norms:
        raise V6TrainingError("training emitted no finite grad_norm")
    return {
        "finite": True,
        "numeric_values_checked": checked,
        "loss_values_checked": len(losses),
        "grad_norm_values_checked": len(grad_norms),
        "final_train_loss": float(metrics["train_loss"]),
    }


def checkpoint_hashes(path: Path) -> dict[str, str]:
    required = ("adapter_config.json", "adapter_model.safetensors")
    hashes: dict[str, str] = {}
    for name in required:
        target = path / name
        if not target.is_file() or target.stat().st_size <= 0:
            raise V6TrainingError(f"incomplete adapter checkpoint: {target}")
        hashes[name] = sha256_file(target)
    return hashes


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD^{commit}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def require_clean_tracked_source() -> None:
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged"),
    ):
        if subprocess.run(command, cwd=ROOT, check=False).returncode != 0:
            raise V6TrainingError(
                f"repository contains {label} tracked source drift"
            )


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--expected-train-sha256")
    parser.add_argument("--selector-manifest", type=Path)
    parser.add_argument("--expected-selector-manifest-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--expected-model-config-sha256")
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--smoke-steps", type=int, default=2)
    parser.add_argument("--smoke-rows", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument("--optimizer", choices=("paged_adamw_8bit",), default="paged_adamw_8bit")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.model != MODEL_ID:
        raise V6TrainingError(f"V6 directional screen freezes model={MODEL_ID}")
    for value, label in (
        (args.model_revision, "--model-revision"),
        (args.tokenizer_revision, "--tokenizer-revision"),
    ):
        if COMMIT_RE.fullmatch(value or "") is None:
            raise V6TrainingError(f"{label} must be a full lowercase commit")
    if args.model_revision != MODEL_REVISION:
        raise V6TrainingError(
            f"V6 directional screen freezes model revision {MODEL_REVISION}"
        )
    if args.tokenizer_revision != TOKENIZER_REVISION:
        raise V6TrainingError(
            f"V6 directional screen freezes tokenizer revision {TOKENIZER_REVISION}"
        )
    if args.train_seed != TRAIN_SEED:
        raise V6TrainingError(
            f"directional screen freezes one train seed: {TRAIN_SEED}"
        )
    integer_positive = (
        "max_seq_len",
        "max_steps",
        "smoke_steps",
        "smoke_rows",
        "batch_size",
        "gradient_accumulation_steps",
        "lora_rank",
        "lora_alpha",
    )
    for name in integer_positive:
        if getattr(args, name) <= 0:
            raise V6TrainingError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise V6TrainingError("--learning-rate must be finite and positive")
    if not 0 <= args.warmup_ratio < 1:
        raise V6TrainingError("--warmup-ratio must be in [0,1)")
    if args.weight_decay < 0 or args.max_grad_norm <= 0:
        raise V6TrainingError("weight decay/gradient norm contract is invalid")
    if not 0 <= args.lora_dropout < 1:
        raise V6TrainingError("--lora-dropout must be in [0,1)")
    if len(args.lora_target_modules) != len(set(args.lora_target_modules)):
        raise V6TrainingError("LoRA target modules contain duplicates")
    if not args.identity_only:
        for name in (
            "train_file",
            "expected_train_sha256",
            "selector_manifest",
            "expected_selector_manifest_sha256",
            "arm",
            "expected_model_config_sha256",
            "expected_tokenizer_sha256",
            "expected_source_commit",
        ):
            if getattr(args, name) in (None, ""):
                raise V6TrainingError(f"--{name.replace('_', '-')} is required")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise V6TrainingError(
            f"output directory is not empty; archive it first: {args.output_dir}"
        )

    # Heavy imports begin here so pure contract tests require no ML stack.
    import transformers
    from transformers import (
        AutoConfig,
        AutoTokenizer,
    )

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    observed_model_config_hash = model_config_fingerprint(config)
    observed_tokenizer_hash = tokenizer_fingerprint(tokenizer)
    identities = {
        "model": args.model,
        "model_revision": args.model_revision,
        "model_config_sha256": observed_model_config_hash,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_sha256": observed_tokenizer_hash,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.identity_only:
        write_json(args.output_dir / "model_tokenizer_identities.json", identities)
        print(json.dumps(identities, sort_keys=True, indent=2))
        return

    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    expected_model_hash = checked_sha256(
        args.expected_model_config_sha256, "--expected-model-config-sha256"
    )
    expected_tokenizer_hash = checked_sha256(
        args.expected_tokenizer_sha256, "--expected-tokenizer-sha256"
    )
    if observed_model_config_hash != expected_model_hash:
        raise V6TrainingError("model config fingerprint drift")
    if observed_tokenizer_hash != expected_tokenizer_hash:
        raise V6TrainingError("tokenizer fingerprint drift")
    expected_train_hash = checked_sha256(
        args.expected_train_sha256, "--expected-train-sha256"
    )
    if sha256_file(args.train_file) != expected_train_hash:
        raise V6TrainingError("training JSONL SHA-256 drift")
    expected_manifest_hash = checked_sha256(
        args.expected_selector_manifest_sha256,
        "--expected-selector-manifest-sha256",
    )
    observed_manifest_hash = sha256_file(args.selector_manifest)
    if observed_manifest_hash != expected_manifest_hash:
        raise V6TrainingError("selector manifest file SHA-256 drift")
    selector_manifest = read_json_object(
        args.selector_manifest, label="selector manifest"
    )
    if COMMIT_RE.fullmatch(args.expected_source_commit or "") is None:
        raise V6TrainingError("--expected-source-commit must be a full commit")
    require_clean_tracked_source()
    source_commit = git_commit()
    if source_commit != args.expected_source_commit:
        raise V6TrainingError(
            f"source commit drift: {source_commit} != {args.expected_source_commit}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise V6TrainingError("tokenizer has no pad token")
    tokenizer.padding_side = "right"
    raw_rows = read_jsonl(args.train_file)
    validated_rows = [
        validate_row(
            row,
            arm=args.arm,
            tokenizer_name=args.model,
            tokenizer_revision=args.tokenizer_revision,
        )
        for row in raw_rows
    ]
    source_provenance = validate_dataset_provenance(
        validated_rows,
        selector_manifest,
        arm=args.arm,
        expected_selector_manifest_sha256=expected_manifest_hash,
        observed_selector_manifest_sha256=observed_manifest_hash,
    )
    encoded = [
        encode_row(tokenizer, row, max_seq_len=args.max_seq_len)
        for row in validated_rows
    ]
    audit = data_audit(
        encoded,
        arm=args.arm,
        train_file_sha256=expected_train_hash,
        max_seq_len=args.max_seq_len,
        source_provenance=source_provenance,
    )
    if args.mode == "smoke":
        selected_rows = sorted(
            encoded,
            key=lambda row: (row["sequence_tokens"], row["id"]),
            reverse=True,
        )[: args.smoke_rows]
        effective_steps = args.smoke_steps
    else:
        selected_rows = encoded
        effective_steps = args.max_steps
    audit["mode"] = args.mode
    audit["scheduled_rows"] = len(selected_rows)
    write_json(args.output_dir / "data_audit.json", audit)

    if not torch.cuda.is_available():
        raise V6TrainingError("7B 4-bit QLoRA requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise V6TrainingError("V6 freezes bfloat16 compute")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.empty(1, device="cuda:0")
    torch.cuda.synchronize(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    set_seed(args.train_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items: Sequence[dict[str, Any]]) -> None:
            self.items = list(items)

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.items[index]
            return {
                "input_ids": row["input_ids"],
                "attention_mask": row["attention_mask"],
                "labels": row["labels"],
            }

    class DynamicMaskedCollator:
        def __init__(self) -> None:
            self.batches = 0
            self.examples = 0
            self.supervised_tokens = 0
            self.nonpadding_tokens = 0

        def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
            maximum = max(len(feature["input_ids"]) for feature in features)
            ids: list[list[int]] = []
            attention: list[list[int]] = []
            labels: list[list[int]] = []
            for feature in features:
                padding = maximum - len(feature["input_ids"])
                ids.append(
                    feature["input_ids"] + [tokenizer.pad_token_id] * padding
                )
                attention.append(feature["attention_mask"] + [0] * padding)
                labels.append(feature["labels"] + [IGNORE_INDEX] * padding)
                self.examples += 1
                self.supervised_tokens += sum(
                    token != IGNORE_INDEX for token in feature["labels"]
                )
                self.nonpadding_tokens += sum(feature["attention_mask"])
            self.batches += 1
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
        args.model,
        revision=args.model_revision,
        quantization_config=quantization,
        device_map={"": 0},
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
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules,
        ),
    )

    class MaskedCausalTrainer(Trainer):
        def __init__(self, *trainer_args: Any, **trainer_kwargs: Any) -> None:
            super().__init__(*trainer_args, **trainer_kwargs)
            self.model_accepts_loss_kwargs = False

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs["labels"]
            supervised = labels.ne(IGNORE_INDEX)
            if not bool(supervised.any()):
                raise V6TrainingError("microbatch has no supervised tokens")
            first = int(supervised.nonzero(as_tuple=False)[:, 1].min().item())
            keep, target_start = tail_logit_contract(labels.shape[1], first)
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                logits_to_keep=keep,
            )
            logits = outputs.logits[:, :-1, :]
            targets = labels[:, target_start:]
            if logits.shape[:2] != targets.shape:
                raise V6TrainingError("tail-logit/target alignment failure")
            loss = functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite V6 SFT loss: {loss.detach().item()!r}"
                )
            return (loss, outputs) if return_outputs else loss

    hyperparameters = {
        "train_seed": args.train_seed,
        "max_seq_len": args.max_seq_len,
        "max_steps": effective_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "optimizer": args.optimizer,
        "quantization": {
            "bits": 4,
            "type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(args.lora_target_modules),
        },
    }
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer_state"),
        max_steps=effective_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        bf16=True,
        fp16=False,
        tf32=False,
        optim=args.optimizer,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.train_seed,
        data_seed=args.train_seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=False,
        label_names=["labels"],
        logging_nan_inf_filter=False,
    )
    command = " ".join(
        shlex.quote(argument) for argument in [sys.executable, *sys.argv]
    )
    (args.output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    collator = DynamicMaskedCollator()
    trainer = MaskedCausalTrainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(selected_rows),
        data_collator=collator,
    )
    result = trainer.train()
    finite_audit = finite_training_audit(
        trainer.state.log_history, result.metrics
    )
    checkpoint = args.output_dir / "checkpoint_final"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    adapter_hashes = checkpoint_hashes(checkpoint)

    training_metrics = {
        "protocol": PROTOCOL,
        "arm": args.arm,
        "mode": args.mode,
        "trainer_metrics": _json_safe(result.metrics),
        "finite_audit": finite_audit,
        "log_history": _json_safe(trainer.state.log_history),
        "actual_consumed_token_accounting": {
            "microbatches_collated": collator.batches,
            "examples_collated": collator.examples,
            "supervised_target_tokens": collator.supervised_tokens,
            "nonpadding_tokens": collator.nonpadding_tokens,
            "padding_excluded": True,
        },
    }
    write_json(args.output_dir / "training_metrics.json", training_metrics)
    final_audit = {
        **audit,
        "training_finite": True,
        "adapter_files_sha256": adapter_hashes,
        "actual_consumed_token_accounting": training_metrics[
            "actual_consumed_token_accounting"
        ],
    }
    write_json(args.output_dir / "audit.json", final_audit)
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    run_manifest = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "source_commit": source_commit,
        "arm": args.arm,
        "mode": args.mode,
        "identities": identities,
        "train_file": str(args.train_file.resolve()),
        "train_file_sha256": expected_train_hash,
        "selector_manifest": str(args.selector_manifest.resolve()),
        "selector_manifest_sha256": expected_manifest_hash,
        "candidate_pool_sha256": source_provenance["candidate_pool_sha256"],
        "source_provenance": source_provenance,
        "data_audit_file_sha256": sha256_file(
            args.output_dir / "data_audit.json"
        ),
        "audit_file_sha256": sha256_file(args.output_dir / "audit.json"),
        "hyperparameters": hyperparameters,
        "hyperparameters_sha256": canonical_sha256(hyperparameters),
        "adapter_files_sha256": adapter_hashes,
        "environment": environment,
        "official_test_used": False,
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    hash_targets = [
        args.output_dir / "command.txt",
        args.output_dir / "data_audit.json",
        args.output_dir / "training_metrics.json",
        args.output_dir / "audit.json",
        args.output_dir / "run_manifest.json",
        checkpoint / "adapter_config.json",
        checkpoint / "adapter_model.safetensors",
    ]
    write_json(
        args.output_dir / "files_sha256.json",
        {
            str(path.relative_to(args.output_dir)): sha256_file(path)
            for path in hash_targets
        },
    )


if __name__ == "__main__":
    main()
