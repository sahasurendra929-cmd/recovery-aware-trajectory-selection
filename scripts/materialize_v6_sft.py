#!/usr/bin/env python3
"""Materialize a frozen V6 selector manifest into leakage-safe SFT JSONL.

The selector atom is a complete ``candidate_pair`` with exactly two sibling
error branches.  A recovery manifest therefore produces two training rows per
selected task.  The failed assistant tool call and its adjacent error result
remain in the prompt as context only; every successful assistant output in the
recovery suffix is supervised.  Legacy V6 suffixes are fresh policy rollouts;
V6.10 suffixes are explicitly marked as sanitized deterministic oracle data.
The materializer preserves that distinction instead of relabeling oracle data
as freshly generated.

``flawless_only`` is a matched-task control.  Its selected records must carry
the complete immutable ``clean_view`` (messages, label mask, tool schemas,
token counts, and success evidence).  A clean ID plus precomputed costs is not
enough to authorize training.

The output retains the message-level ``messages``/``label_mask`` shape, but its
objective is deliberately broader than the V5 tool-action trainer: every
non-failed assistant output in the fresh suffix is a label, including natural
language.  A V6 full-suffix trainer is therefore required.  The CLI computes
an exact token contract with a pinned Hugging Face tokenizer.  Tests may inject
a contract builder, but the builder's identity and the hash of its complete
input are always recorded.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

try:
    import prepare_v5_sft_causal as v5_data
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts import prepare_v5_sft_causal as v5_data


PROTOCOL = "v6_sft_materialization_v1"
MANIFEST_PROTOCOL = "v6_matched_task_selector_manifests_v1"
RECOVERY_TRAINER_ARM = "v6_recovery_selected"
FLAWLESS_TRAINER_ARM = "v6_flawless_control"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLES = {"system", "user", "assistant", "tool"}

ContractBuilder = Callable[
    [list[dict[str, Any]], list[bool], list[dict[str, Any]]],
    dict[str, Any],
]


class V6MaterializationError(RuntimeError):
    """A selector manifest cannot safely be converted to SFT examples."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V6MaterializationError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise V6MaterializationError(
            f"{label} must be a lowercase 64-character SHA-256"
        )
    return text


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise V6MaterializationError(f"{label} must be a positive integer")
    return value


def _boolean_mask(value: Any, length: int, label: str) -> list[bool]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(type(item) is not bool for item in value)
    ):
        raise V6MaterializationError(
            f"{label} must contain exactly {length} booleans"
        )
    return list(value)


def _tool_call(call: Any, label: str) -> tuple[str, dict[str, Any]]:
    if not isinstance(call, Mapping):
        raise V6MaterializationError(f"{label} must be an object")
    function = call.get("function")
    if function is not None:
        if not isinstance(function, Mapping):
            raise V6MaterializationError(f"{label}.function must be an object")
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = call.get("name")
        arguments = call.get("arguments", {})
    name = _nonempty(name, f"{label}.name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise V6MaterializationError(
                f"{label}.arguments is not valid JSON"
            ) from error
    if not isinstance(arguments, dict):
        raise V6MaterializationError(
            f"{label}.arguments must encode an object"
        )
    return name, deepcopy(arguments)


def _normalize_messages(
    raw_messages: Any, *, label: str
) -> list[dict[str, Any]]:
    """Validate messages while preserving the already-measured call IDs."""
    if not isinstance(raw_messages, list) or not raw_messages:
        raise V6MaterializationError(f"{label} must be a non-empty message list")
    output: list[dict[str, Any]] = []
    next_call = 1
    pending_call: tuple[str, str] | None = None
    for index, raw in enumerate(raw_messages):
        where = f"{label}[{index}]"
        if not isinstance(raw, Mapping):
            raise V6MaterializationError(f"{where} must be an object")
        role = raw.get("role")
        if role not in ROLES:
            raise V6MaterializationError(f"{where}.role is unsupported: {role!r}")
        calls = raw.get("tool_calls") or []
        if role == "tool":
            if pending_call is None:
                raise V6MaterializationError(
                    f"{where}: tool result lacks an adjacent assistant tool call"
                )
            if calls:
                raise V6MaterializationError(
                    f"{where}: tool result may not contain tool_calls"
                )
            if not isinstance(raw.get("error"), bool):
                raise V6MaterializationError(
                    f"{where}.error must be an explicit boolean"
                )
            call_id, call_name = pending_call
            supplied_link = raw.get("tool_call_id", raw.get("id"))
            if supplied_link is not None and supplied_link != call_id:
                raise V6MaterializationError(
                    f"{where}: tool-call link differs from adjacent assistant ID"
                )
            item: dict[str, Any] = {
                "role": "tool",
                "content": deepcopy(raw.get("content")),
                "tool_call_id": call_id,
                "error": raw["error"],
            }
            if raw.get("name") is not None:
                observed_name = _nonempty(raw.get("name"), f"{where}.name")
                if observed_name != call_name:
                    raise V6MaterializationError(
                        f"{where}: tool name differs from adjacent call"
                    )
                item["name"] = observed_name
            output.append(item)
            pending_call = None
            continue
        if pending_call is not None:
            raise V6MaterializationError(
                f"{where}: previous assistant tool call lacks adjacent result"
            )
        if role != "assistant" and calls:
            raise V6MaterializationError(
                f"{where}: only assistant messages may contain tool_calls"
            )
        item = {"role": role, "content": deepcopy(raw.get("content"))}
        if calls:
            if not isinstance(calls, list) or len(calls) != 1:
                raise V6MaterializationError(
                    f"{where}: exactly one tool call per assistant turn is required"
                )
            name, arguments = _tool_call(calls[0], f"{where}.tool_calls[0]")
            source_call = calls[0]
            call_id = source_call.get("id")
            if call_id is None:
                call_id = f"call_{next_call:04d}"
                next_call += 1
            elif not isinstance(call_id, str) or not call_id:
                raise V6MaterializationError(
                    f"{where}.tool_calls[0].id must be a non-empty string"
                )
            item["tool_calls"] = [
                {"id": call_id, "name": name, "arguments": arguments}
            ]
            pending_call = (call_id, name)
        elif role == "assistant" and raw.get("content") in (None, ""):
            raise V6MaterializationError(
                f"{where}: assistant message has neither content nor a tool call"
            )
        output.append(item)
    if pending_call is not None:
        raise V6MaterializationError(
            f"{label}: final assistant tool call lacks an adjacent result"
        )
    if output[0].get("role") != "system":
        raise V6MaterializationError(
            f"{label}: first message must be the frozen agent system policy"
        )
    if any(message.get("role") == "system" for message in output[1:]):
        raise V6MaterializationError(
            f"{label}: system policy may appear only as the first message"
        )
    return output


def _outcomes(
    messages: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[list[int], list[int]]:
    successful: list[int] = []
    failed: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if (
                index + 1 >= len(messages)
                or messages[index + 1].get("role") != "tool"
            ):
                raise V6MaterializationError(
                    f"{label}[{index}]: tool call lacks adjacent result"
                )
            if messages[index + 1].get("error") is True:
                failed.append(index)
            elif messages[index + 1].get("error") is False:
                successful.append(index)
            else:
                raise V6MaterializationError(
                    f"{label}[{index + 1}].error must be boolean"
                )
        elif message.get("role") == "tool":
            if (
                index == 0
                or messages[index - 1].get("role") != "assistant"
                or not messages[index - 1].get("tool_calls")
            ):
                raise V6MaterializationError(
                    f"{label}[{index}]: orphan tool result"
                )
    return successful, failed


def _schema_names(schemas: Any, *, label: str) -> tuple[list[dict[str, Any]], set[str]]:
    if (
        not isinstance(schemas, list)
        or not schemas
        or any(not isinstance(row, dict) for row in schemas)
    ):
        raise V6MaterializationError(
            f"{label} must be a non-empty list of tool-schema objects"
        )
    names: set[str] = set()
    for index, schema in enumerate(schemas):
        function = schema.get("function")
        name = (
            function.get("name")
            if isinstance(function, Mapping)
            else schema.get("name")
        )
        names.add(_nonempty(name, f"{label}[{index}].name"))
    return deepcopy(schemas), names


def _resolve_tool_schemas(
    owners: Sequence[Mapping[str, Any]], *, label: str
) -> list[dict[str, Any]]:
    candidates = [
        owner["tool_schemas"]
        for owner in owners
        if isinstance(owner, Mapping) and "tool_schemas" in owner
    ]
    if not candidates:
        raise V6MaterializationError(
            f"{label}: immutable tool_schemas are missing from the selected record"
        )
    canonical_candidates = {canonical(value) for value in candidates}
    if len(canonical_candidates) != 1:
        raise V6MaterializationError(
            f"{label}: tool_schemas differ across selected-record layers"
        )
    schemas, _ = _schema_names(candidates[0], label=f"{label}.tool_schemas")
    return schemas


def _require_calls_in_schemas(
    messages: Sequence[Mapping[str, Any]],
    schemas: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    _, names = _schema_names(list(schemas), label=f"{label}.tool_schemas")
    for index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            name, _ = _tool_call(call, f"{label}.messages[{index}].tool_calls[0]")
            if name not in names:
                raise V6MaterializationError(
                    f"{label}.messages[{index}]: unknown tool {name!r}"
                )


def _nonfailed_assistant_indices(
    messages: Sequence[Mapping[str, Any]],
    *,
    start: int,
    label: str,
) -> set[int]:
    """Return every assistant output eligible for full-suffix supervision."""
    selected: set[int] = set()
    for index in range(start, len(messages)):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        if message.get("tool_calls"):
            if (
                index + 1 >= len(messages)
                or messages[index + 1].get("role") != "tool"
            ):
                raise V6MaterializationError(
                    f"{label}[{index}]: assistant tool call lacks evidence"
                )
            if messages[index + 1].get("error") is False:
                selected.add(index)
            elif messages[index + 1].get("error") is not True:
                raise V6MaterializationError(
                    f"{label}[{index + 1}].error must be boolean"
                )
        elif message.get("content") not in (None, ""):
            selected.add(index)
        else:  # pragma: no cover - normalized messages already reject this
            raise V6MaterializationError(
                f"{label}[{index}]: empty assistant output"
            )
    return selected


def _validate_contract(
    value: Any,
    *,
    selected_indices: Sequence[int],
    label: str,
) -> dict[str, Any]:
    required = {"sequence_tokens", "supervised_tokens", "label_spans"}
    if not isinstance(value, dict) or set(value) != required:
        raise V6MaterializationError(
            f"{label} must contain exactly {sorted(required)}"
        )
    sequence_tokens = _positive_int(
        value["sequence_tokens"], f"{label}.sequence_tokens"
    )
    supervised_tokens = _positive_int(
        value["supervised_tokens"], f"{label}.supervised_tokens"
    )
    spans = value["label_spans"]
    if not isinstance(spans, list) or len(spans) != len(selected_indices):
        raise V6MaterializationError(
            f"{label}.label_spans must contain one span per selected message"
        )
    observed_indices: list[int] = []
    total = 0
    previous_end = -1
    checked_spans: list[dict[str, int]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != {
            "message_index",
            "token_start",
            "token_end",
        }:
            raise V6MaterializationError(
                f"{label}.label_spans[{index}] has an invalid shape"
            )
        message_index = span["message_index"]
        start = span["token_start"]
        end = span["token_end"]
        if (
            isinstance(message_index, bool)
            or not isinstance(message_index, int)
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > sequence_tokens
            or start < previous_end
        ):
            raise V6MaterializationError(
                f"{label}.label_spans[{index}] has invalid token bounds"
            )
        observed_indices.append(message_index)
        total += end - start
        previous_end = end
        checked_spans.append(
            {
                "message_index": message_index,
                "token_start": start,
                "token_end": end,
            }
        )
    if observed_indices != list(selected_indices):
        raise V6MaterializationError(
            f"{label}: label spans do not match the message-level label mask"
        )
    if total != supervised_tokens or supervised_tokens > sequence_tokens:
        raise V6MaterializationError(
            f"{label}: supervised-token total is inconsistent"
        )
    return {
        "sequence_tokens": sequence_tokens,
        "supervised_tokens": supervised_tokens,
        "label_spans": checked_spans,
    }


def _build_contract(
    messages: list[dict[str, Any]],
    mask: list[bool],
    schemas: list[dict[str, Any]],
    *,
    tokenizer: Any | None,
    contract_builder: ContractBuilder | None,
    label: str,
) -> dict[str, Any]:
    if contract_builder is None:
        if tokenizer is None:
            raise V6MaterializationError(
                f"{label}: tokenizer or contract_builder is required"
            )
        raw = v5_data.token_contract(tokenizer, messages, mask, schemas)
    else:
        raw = contract_builder(deepcopy(messages), list(mask), deepcopy(schemas))
    return _validate_contract(
        raw,
        selected_indices=[
            index for index, selected in enumerate(mask) if selected
        ],
        label=f"{label}.token_contract",
    )


def _validate_quality(pair: Mapping[str, Any], *, label: str) -> None:
    quality = pair.get("quality")
    if not isinstance(quality, Mapping):
        raise V6MaterializationError(f"{label}.quality must be an object")
    for field in (
        "real_error_executed",
        "matched_recovery_replay_success",
        "no_future_leakage",
        "independent_replay_audited",
    ):
        if quality.get(field) is not True:
            raise V6MaterializationError(f"{label}.quality.{field} must be true")
    if quality.get("official_test_used") is not False:
        raise V6MaterializationError(
            f"{label}.quality.official_test_used must be false"
        )
    if quality.get("failed_positive_labels") != 0:
        raise V6MaterializationError(
            f"{label}.quality.failed_positive_labels must equal zero"
        )


def _recovery_source_mask(
    branch: Mapping[str, Any],
    *,
    prompt_length: int,
    suffix_length: int,
    label: str,
) -> list[bool]:
    has_combined = "loss_mask" in branch
    has_suffix = "fresh_recovery_label_mask" in branch
    if has_combined == has_suffix:
        raise V6MaterializationError(
            f"{label}: provide exactly one of loss_mask (combined prompt+suffix "
            "scope) or fresh_recovery_label_mask (suffix scope)"
        )
    if has_combined:
        return _boolean_mask(
            branch["loss_mask"],
            prompt_length + suffix_length,
            f"{label}.loss_mask",
        )
    suffix_mask = _boolean_mask(
        branch["fresh_recovery_label_mask"],
        suffix_length,
        f"{label}.fresh_recovery_label_mask",
    )
    return [False] * prompt_length + suffix_mask


def _v6_validate_row(row: Mapping[str, Any]) -> None:
    """Fail-close the V6 full-assistant-suffix message objective locally."""
    row_id = _nonempty(row.get("id"), "row.id")
    messages = row.get("messages")
    mask = row.get("label_mask")
    if not isinstance(messages, list) or not messages:
        raise V6MaterializationError(f"{row_id}: messages are missing")
    checked_mask = _boolean_mask(mask, len(messages), f"{row_id}.label_mask")
    selected = {index for index, value in enumerate(checked_mask) if value}
    if not selected:
        raise V6MaterializationError(f"{row_id}: no assistant suffix labels")
    _, failed = _outcomes(messages, label=row_id)
    if selected & set(failed):
        raise V6MaterializationError(f"{row_id}: failed assistant call was labeled")
    for index in selected:
        message = messages[index]
        if message.get("role") != "assistant":
            raise V6MaterializationError(
                f"{row_id}: non-assistant message {index} was labeled"
            )
        if message.get("tool_calls") and messages[index + 1].get("error") is not False:
            raise V6MaterializationError(
                f"{row_id}: labeled tool call {index} did not succeed"
            )
        if not message.get("tool_calls") and message.get("content") in (None, ""):
            raise V6MaterializationError(
                f"{row_id}: labeled assistant text {index} is empty"
            )
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise V6MaterializationError(f"{row_id}: metadata is missing")
    if metadata.get("official_test_used") is not False:
        raise V6MaterializationError(f"{row_id}: official test was opened")


def _metadata(
    *,
    row_id: str,
    selector: str,
    selection_seed: Any,
    domain: str,
    task_identity: str,
    source: str,
    schemas: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    mask: list[bool],
    source_trajectory_sha256: str,
    manifest_sha256: str,
    candidate_pool_sha256: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    candidate_pair_id: str | None,
    branch_id: str | None,
    clean_id: str | None,
    source_hashes: Mapping[str, Any],
    recovery_suffix_origin: str = "not_applicable",
    fresh_recovery_generated: bool = False,
    gold_reference_suffix_used: bool = False,
) -> dict[str, Any]:
    try:
        task_id = task_identity.split(":", 1)[1]
    except IndexError as error:
        raise V6MaterializationError(
            f"{task_identity}: task identity is not namespaced"
        ) from error
    selected = [index for index, value in enumerate(mask) if value]
    _, failed = _outcomes(messages, label=row_id)
    return {
        "arm": (
            FLAWLESS_TRAINER_ARM
            if source == "perfect_success"
            else RECOVERY_TRAINER_ARM
        ),
        "paper_arm": selector,
        "selector": selector,
        "selection_seed": selection_seed,
        "source": source,
        "domain": domain,
        "task_id": task_id,
        "task_identity": task_identity,
        "trial": branch_id or clean_id or row_id,
        "seed": selection_seed,
        "candidate_pair_id": candidate_pair_id,
        "branch_id": branch_id,
        "clean_id": clean_id,
        "source_example_id": row_id,
        "source_pair_id": candidate_pair_id or clean_id,
        "source_split": "inner_train",
        "fit_split": "train_schedule",
        "tool_schemas": schemas,
        "tool_schemas_sha256": canonical_sha256(schemas),
        "call_id_policy": "source_preserved_or_deterministic_when_missing",
        "full_trajectory_reward": 1.0,
        "source_trajectory_sha256": source_trajectory_sha256,
        "source_hashes": deepcopy(dict(source_hashes)),
        "selector_manifest_sha256": manifest_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "token_contract_input_sha256": canonical_sha256(
            {
                "messages": messages,
                "label_mask": mask,
                "tool_schemas": schemas,
            }
        ),
        "failed_assistant_message_indices": failed,
        "injected_failed_assistant_message_index": (
            failed[0] if failed else None
        ),
        "failed_action_label_messages": len(set(failed) & set(selected)),
        "full_nonfailed_assistant_suffix_supervised": True,
        "assistant_text_outputs_supervised": True,
        "fresh_recovery_suffix": fresh_recovery_generated,
        "recovery_suffix_origin": recovery_suffix_origin,
        "gold_reference_suffix_used": gold_reference_suffix_used,
        "future_clean_suffix_used": False,
        "official_test_used": False,
        "materialization_protocol": PROTOCOL,
    }


def _recovery_suffix_provenance(
    branch: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, bool, bool]:
    """Return truthful suffix provenance for legacy and V6.10 candidates."""

    v610 = isinstance(branch.get("reference_preflight_binding"), Mapping)
    if v610:
        audited = branch.get("recovery_suffix_provenance")
        binding = branch["reference_preflight_binding"]
        if (
            not isinstance(audited, Mapping)
            or audited.get("protocol")
            != "v6_10_audited_recovery_suffix_provenance_v1"
            or audited.get("origin")
            != "sanitized_deterministic_reference_plan"
            or audited.get("fresh_recovery_generated") is not False
            or audited.get("gold_suffix_used") is not True
            or audited.get("reference_preflight_receipt_sha256")
            != binding.get("reference_preflight_receipt_sha256")
            or audited.get("reference_task_preflight_sha256")
            != binding.get("reference_task_preflight_sha256")
            or audited.get("sanitized_reference_plan_sha256")
            != binding.get("sanitized_reference_plan_sha256")
            or SHA256_RE.fullmatch(
                str(audited.get("matched_recovery_evidence_sha256"))
            )
            is None
        ):
            raise V6MaterializationError(
                f"{label}: V6.10 recovery suffix lacks canonical audited "
                "sanitized-oracle provenance"
            )
        return (
            "sanitized_deterministic_reference_plan",
            False,
            True,
        )

    producer = branch.get("producer_evidence")
    matched = (
        producer.get("matched_recovery")
        if isinstance(producer, Mapping)
        else None
    )
    if isinstance(matched, Mapping):
        fresh = matched.get("fresh_recovery_generated")
        gold = matched.get("gold_suffix_used")
        if fresh is False or gold is True:
            raise V6MaterializationError(
                f"{label}: legacy recovery suffix unexpectedly uses oracle "
                "provenance"
            )
    return "fresh_teacher_generation", True, False


def _materialize_recovery_branch(
    *,
    record: Mapping[str, Any],
    pair: Mapping[str, Any],
    branch: Mapping[str, Any],
    manifest_sha256: str,
    candidate_pool_sha256: str,
    selector: str,
    selection_seed: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    tokenizer: Any | None,
    contract_builder: ContractBuilder | None,
) -> dict[str, Any]:
    pair_id = _nonempty(pair.get("candidate_pair_id"), "candidate_pair_id")
    branch_id = _nonempty(
        branch.get("branch_id"), f"{pair_id}.branch.branch_id"
    )
    label = f"{pair_id}:{branch_id}"
    prompt_fields = [
        field for field in ("prompt", "training_prompt") if field in branch
    ]
    if not prompt_fields:
        raise V6MaterializationError(
            f"{label}: prompt/training_prompt is missing"
        )
    raw_prompt = branch[prompt_fields[0]]
    if any(
        canonical(branch[field]) != canonical(raw_prompt)
        for field in prompt_fields[1:]
    ):
        raise V6MaterializationError(
            f"{label}: prompt and training_prompt differ"
        )
    raw_suffix = branch.get("fresh_recovery_suffix")
    if not isinstance(raw_prompt, list) or not isinstance(raw_suffix, list):
        raise V6MaterializationError(
            f"{label}: prompt and fresh_recovery_suffix must be lists"
        )
    if not raw_suffix:
        raise V6MaterializationError(f"{label}: fresh recovery suffix is empty")
    (
        recovery_suffix_origin,
        fresh_recovery_generated,
        gold_reference_suffix_used,
    ) = _recovery_suffix_provenance(branch, label=label)
    raw_mask = _recovery_source_mask(
        branch,
        prompt_length=len(raw_prompt),
        suffix_length=len(raw_suffix),
        label=label,
    )
    if any(raw_mask[: len(raw_prompt)]):
        raise V6MaterializationError(
            f"{label}: prompt context, including the failed call, was labeled"
        )
    messages = _normalize_messages(
        [*deepcopy(raw_prompt), *deepcopy(raw_suffix)], label=f"{label}.messages"
    )
    schemas = _resolve_tool_schemas(
        [branch, pair], label=label
    )
    _require_calls_in_schemas(messages, schemas, label=label)
    successful, failed = _outcomes(messages, label=label)
    prompt_successful, prompt_failed = _outcomes(
        messages[: len(raw_prompt)], label=f"{label}.prompt"
    )
    del prompt_successful
    if (
        len(prompt_failed) != 1
        or prompt_failed[0] != len(raw_prompt) - 2
        or messages[len(raw_prompt) - 1].get("role") != "tool"
        or messages[len(raw_prompt) - 1].get("error") is not True
    ):
        raise V6MaterializationError(
            f"{label}: prompt must end with exactly one failed assistant call "
            "and its adjacent real error result"
        )
    del successful
    expected_labels = _nonfailed_assistant_indices(
        messages,
        start=len(raw_prompt),
        label=f"{label}.fresh_recovery_suffix",
    )
    observed_labels = {
        index for index, selected in enumerate(raw_mask) if selected
    }
    if not expected_labels or observed_labels != expected_labels:
        raise V6MaterializationError(
            f"{label}: labels must be exactly all non-failed assistant outputs "
            "(tool calls and text) in the fresh recovery suffix"
        )
    mask = raw_mask
    contract = _build_contract(
        messages,
        mask,
        schemas,
        tokenizer=tokenizer,
        contract_builder=contract_builder,
        label=label,
    )
    measured_contract = branch.get("token_contract")
    if (
        not isinstance(measured_contract, Mapping)
        or canonical(measured_contract) != canonical(contract)
    ):
        raise V6MaterializationError(
            f"{label}: materialized token contract differs from frozen "
            "candidate measurement"
        )
    if contract["supervised_tokens"] != _positive_int(
        branch.get("supervised_target_tokens"),
        f"{label}.supervised_target_tokens",
    ):
        raise V6MaterializationError(
            f"{label}: materialized supervised tokens differ from registered c_sup"
        )
    if contract["sequence_tokens"] != _positive_int(
        branch.get("nonpadding_tokens"), f"{label}.nonpadding_tokens"
    ):
        raise V6MaterializationError(
            f"{label}: materialized sequence tokens differ from registered c_nonpad"
        )
    task_identity = _nonempty(pair.get("task_identity"), f"{pair_id}.task_identity")
    domain = _nonempty(pair.get("domain"), f"{pair_id}.domain")
    if not task_identity.startswith(f"{domain}:"):
        raise V6MaterializationError(
            f"{pair_id}: task identity is not namespaced by domain"
        )
    row_id = f"v6:{selector}:{pair_id}:{branch_id}"
    branch_sha = canonical_sha256(branch)
    source_hashes = {
        "selector_manifest_sha256": manifest_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
        "candidate_pair_sha256": canonical_sha256(pair),
        "branch_sha256": branch_sha,
        "prefix_sha256": _sha256(
            pair.get("prefix_sha256"), f"{pair_id}.prefix_sha256"
        ),
        "environment_snapshot_sha256": _sha256(
            pair.get("environment_snapshot_sha256"),
            f"{pair_id}.environment_snapshot_sha256",
        ),
    }
    row = {
        "id": row_id,
        "messages": messages,
        "label_mask": mask,
        "metadata": _metadata(
            row_id=row_id,
            selector=selector,
            selection_seed=selection_seed,
            domain=domain,
            task_identity=task_identity,
            source="failure_rich",
            schemas=schemas,
            messages=messages,
            mask=mask,
            source_trajectory_sha256=branch_sha,
            manifest_sha256=manifest_sha256,
            candidate_pool_sha256=candidate_pool_sha256,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            candidate_pair_id=pair_id,
            branch_id=branch_id,
            clean_id=None,
            source_hashes=source_hashes,
            recovery_suffix_origin=recovery_suffix_origin,
            fresh_recovery_generated=fresh_recovery_generated,
            gold_reference_suffix_used=gold_reference_suffix_used,
        ),
        "token_contract": contract,
    }
    _v6_validate_row(row)
    return row


def _clean_mask(view: Mapping[str, Any], *, message_count: int, label: str) -> list[bool]:
    fields = [field for field in ("label_mask", "loss_mask") if field in view]
    if len(fields) != 1:
        raise V6MaterializationError(
            f"{label}: provide exactly one of label_mask or loss_mask"
        )
    return _boolean_mask(view[fields[0]], message_count, f"{label}.{fields[0]}")


def _materialize_clean(
    *,
    record: Mapping[str, Any],
    manifest_sha256: str,
    candidate_pool_sha256: str,
    selector: str,
    selection_seed: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    tokenizer: Any | None,
    contract_builder: ContractBuilder | None,
) -> dict[str, Any]:
    task_identity = _nonempty(record.get("task_identity"), "clean.task_identity")
    domain = _nonempty(record.get("domain"), f"{task_identity}.domain")
    view = record.get("clean_view")
    if not isinstance(view, Mapping):
        raise V6MaterializationError(
            f"{task_identity}: flawless selected record lacks complete clean_view"
        )
    clean_id = _nonempty(view.get("clean_id"), f"{task_identity}.clean_view.clean_id")
    raw_messages = view.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise V6MaterializationError(
            f"{clean_id}: clean_view.messages must be complete and non-empty"
        )
    mask = _clean_mask(view, message_count=len(raw_messages), label=clean_id)
    messages = _normalize_messages(deepcopy(raw_messages), label=f"{clean_id}.messages")
    schemas = _resolve_tool_schemas([view, record], label=clean_id)
    _require_calls_in_schemas(messages, schemas, label=clean_id)
    successful, failed = _outcomes(messages, label=clean_id)
    if failed:
        raise V6MaterializationError(f"{clean_id}: flawless view contains a tool error")
    expected = _nonfailed_assistant_indices(
        messages, start=0, label=f"{clean_id}.messages"
    )
    observed = {index for index, selected in enumerate(mask) if selected}
    if not expected or observed != expected:
        raise V6MaterializationError(
            f"{clean_id}: labels must be exactly all non-failed assistant "
            "outputs, including natural-language responses"
        )
    success_value = view.get(
        "full_trajectory_reward",
        view.get("reward", view.get("official_task_success")),
    )
    if (
        not isinstance(success_value, (int, float))
        or isinstance(success_value, bool)
        or float(success_value) != 1.0
    ):
        raise V6MaterializationError(
            f"{clean_id}: flawless view lacks full-trajectory success evidence"
        )
    contract = _build_contract(
        messages,
        mask,
        schemas,
        tokenizer=tokenizer,
        contract_builder=contract_builder,
        label=clean_id,
    )
    measured_contract = view.get("token_contract")
    if (
        not isinstance(measured_contract, Mapping)
        or canonical(measured_contract) != canonical(contract)
    ):
        raise V6MaterializationError(
            f"{clean_id}: materialized token contract differs from frozen "
            "candidate measurement"
        )
    if contract["supervised_tokens"] != _positive_int(
        view.get("c_sup"), f"{clean_id}.c_sup"
    ):
        raise V6MaterializationError(
            f"{clean_id}: materialized supervised tokens differ from registered c_sup"
        )
    if contract["sequence_tokens"] != _positive_int(
        view.get("c_nonpad"), f"{clean_id}.c_nonpad"
    ):
        raise V6MaterializationError(
            f"{clean_id}: materialized sequence tokens differ from registered c_nonpad"
        )
    if record.get("c_sup") != contract["supervised_tokens"]:
        raise V6MaterializationError(f"{clean_id}: selected-record c_sup drift")
    if record.get("c_nonpad") != contract["sequence_tokens"]:
        raise V6MaterializationError(f"{clean_id}: selected-record c_nonpad drift")
    row_id = f"v6:{selector}:{clean_id}"
    view_sha = canonical_sha256(view)
    source_pair_ids = view.get("source_candidate_pair_ids", [])
    if not isinstance(source_pair_ids, list) or any(
        not isinstance(value, str) or not value for value in source_pair_ids
    ):
        raise V6MaterializationError(
            f"{clean_id}: source_candidate_pair_ids must be a string list"
        )
    source_hashes = {
        "selector_manifest_sha256": manifest_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
        "clean_view_sha256": view_sha,
        "source_candidate_pair_ids_sha256": canonical_sha256(
            sorted(source_pair_ids)
        ),
    }
    row = {
        "id": row_id,
        "messages": messages,
        "label_mask": mask,
        "metadata": _metadata(
            row_id=row_id,
            selector=selector,
            selection_seed=selection_seed,
            domain=domain,
            task_identity=task_identity,
            source="perfect_success",
            schemas=schemas,
            messages=messages,
            mask=mask,
            source_trajectory_sha256=view_sha,
            manifest_sha256=manifest_sha256,
            candidate_pool_sha256=candidate_pool_sha256,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            candidate_pair_id=None,
            branch_id=None,
            clean_id=clean_id,
            source_hashes=source_hashes,
        ),
        "token_contract": contract,
    }
    row["metadata"]["source_candidate_pair_ids"] = sorted(source_pair_ids)
    _v6_validate_row(row)
    return row


def materialize_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str | None = None,
    tokenizer_name: str,
    tokenizer_revision: str,
    tokenizer: Any | None = None,
    contract_builder: ContractBuilder | None = None,
    contract_builder_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert one selector manifest into V6 full-suffix SFT rows."""
    if manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise V6MaterializationError("selector manifest protocol drift")
    if manifest.get("official_test_used") is not False:
        raise V6MaterializationError("selector manifest opened the official test")
    selector = _nonempty(manifest.get("selector"), "manifest.selector")
    candidate_pool_sha256 = _sha256(
        manifest.get("candidate_pool_sha256"),
        "manifest.candidate_pool_sha256",
    )
    manifest_digest = (
        canonical_sha256(manifest)
        if manifest_sha256 is None
        else _sha256(manifest_sha256, "manifest_sha256")
    )
    tokenizer_name = _nonempty(tokenizer_name, "tokenizer_name")
    tokenizer_revision = _nonempty(tokenizer_revision, "tokenizer_revision")
    if contract_builder is not None and not contract_builder_name:
        raise V6MaterializationError(
            "contract_builder_name is required for an injected builder"
        )
    builder_identity = (
        contract_builder_name
        if contract_builder is not None
        else "scripts.prepare_v5_sft_causal.token_contract"
    )
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise V6MaterializationError("selector manifest has no selected records")
    task_ids = [_nonempty(row.get("task_identity"), "selected.task_identity")
                if isinstance(row, Mapping)
                else "" for row in selected]
    if len(task_ids) != len(set(task_ids)):
        raise V6MaterializationError(
            "selector manifest must contain exactly one selected atom per task"
        )
    matched = manifest.get("matched_task_ids")
    if (
        not isinstance(matched, list)
        or sorted(task_ids) != sorted(str(value) for value in matched)
    ):
        raise V6MaterializationError("selected tasks differ from matched_task_ids")
    selection_seed = manifest.get("selection_seed")
    rows: list[dict[str, Any]] = []
    pair_ids: list[str] = []
    if selector == "flawless_only":
        for record in selected:
            if not isinstance(record, Mapping):
                raise V6MaterializationError("selected clean record must be an object")
            rows.append(
                _materialize_clean(
                    record=record,
                    manifest_sha256=manifest_digest,
                    candidate_pool_sha256=candidate_pool_sha256,
                    selector=selector,
                    selection_seed=selection_seed,
                    tokenizer_name=tokenizer_name,
                    tokenizer_revision=tokenizer_revision,
                    tokenizer=tokenizer,
                    contract_builder=contract_builder,
                )
            )
    else:
        for record in selected:
            if not isinstance(record, Mapping):
                raise V6MaterializationError(
                    "selected recovery record must be an object"
                )
            pair = record.get("candidate_pair")
            if not isinstance(pair, Mapping):
                raise V6MaterializationError(
                    f"{record.get('task_identity')}: complete candidate_pair is missing"
                )
            _validate_quality(pair, label=str(pair.get("candidate_pair_id")))
            pair_id = _nonempty(
                pair.get("candidate_pair_id"), "candidate_pair.candidate_pair_id"
            )
            pair_ids.append(pair_id)
            for field in ("candidate_pair_id", "task_identity", "domain"):
                if str(record.get(field)) != str(pair.get(field)):
                    raise V6MaterializationError(
                        f"{pair_id}: selected-record {field} differs from embedded pair"
                    )
            branches = pair.get("branches")
            if not isinstance(branches, list) or len(branches) != 2:
                raise V6MaterializationError(
                    f"{pair_id}: selector atom must contain exactly two sibling branches"
                )
            branch_ids = sorted(
                _nonempty(
                    branch.get("branch_id"),
                    f"{pair_id}.branches[{index}].branch_id",
                )
                for index, branch in enumerate(branches)
                if isinstance(branch, Mapping)
            )
            if len(branch_ids) != 2 or len(set(branch_ids)) != 2:
                raise V6MaterializationError(
                    f"{pair_id}: sibling branch IDs are incomplete or duplicated"
                )
            if sorted(str(value) for value in record.get("branch_ids", [])) != branch_ids:
                raise V6MaterializationError(
                    f"{pair_id}: selected-record branch_ids drift"
                )
            pair_rows = [
                _materialize_recovery_branch(
                    record=record,
                    pair=pair,
                    branch=branch,
                    manifest_sha256=manifest_digest,
                    candidate_pool_sha256=candidate_pool_sha256,
                    selector=selector,
                    selection_seed=selection_seed,
                    tokenizer_name=tokenizer_name,
                    tokenizer_revision=tokenizer_revision,
                    tokenizer=tokenizer,
                    contract_builder=contract_builder,
                )
                for branch in branches
            ]
            observed_sup = sum(
                row["token_contract"]["supervised_tokens"] for row in pair_rows
            )
            observed_nonpad = sum(
                row["token_contract"]["sequence_tokens"] for row in pair_rows
            )
            if record.get("c_sup") != observed_sup:
                raise V6MaterializationError(
                    f"{pair_id}: selected-record c_sup differs from both branches"
                )
            if record.get("c_nonpad") != observed_nonpad:
                raise V6MaterializationError(
                    f"{pair_id}: selected-record c_nonpad differs from both branches"
                )
            cost = pair.get("cost")
            if isinstance(cost, Mapping) and (
                cost.get("c_sup") != observed_sup
                or cost.get("c_nonpad") != observed_nonpad
            ):
                raise V6MaterializationError(
                    f"{pair_id}: embedded scored cost differs from materialization"
                )
            rows.extend(pair_rows)
    budget = manifest.get("budget")
    if not isinstance(budget, Mapping):
        raise V6MaterializationError("manifest budget is missing")
    actual_sup = sum(row["token_contract"]["supervised_tokens"] for row in rows)
    actual_nonpad = sum(row["token_contract"]["sequence_tokens"] for row in rows)
    if budget.get("actual_c_sup") != actual_sup:
        raise V6MaterializationError(
            "materialized supervised-token mass differs from manifest budget"
        )
    if budget.get("actual_c_nonpad") != actual_nonpad:
        raise V6MaterializationError(
            "materialized non-padding-token mass differs from manifest budget"
        )
    if selector == "flawless_only":
        target = budget.get("target_c_sup")
        remaining = budget.get("remaining_c_sup")
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or isinstance(remaining, bool)
            or not isinstance(remaining, int)
            or remaining != target - actual_sup
            or budget.get("budget_comparable_primary") is not False
        ):
            raise V6MaterializationError(
                "flawless natural-token control budget declaration is invalid"
            )
    elif budget.get("remaining_c_sup") != 0:
        raise V6MaterializationError(
            "recovery selector manifest has an unfilled c_sup budget"
        )
    failed_labels = sum(
        row["metadata"]["failed_action_label_messages"] for row in rows
    )
    if failed_labels != 0:
        raise V6MaterializationError("a failed call entered positive supervision")
    row_ids = [row["id"] for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise V6MaterializationError("materialized row IDs are not unique")
    rows.sort(key=lambda row: row["id"])
    audit = {
        "protocol": PROTOCOL,
        "status": "PASS",
        "selector_manifest_sha256": manifest_digest,
        "candidate_pool_sha256": candidate_pool_sha256,
        "selector": selector,
        "selection_seed": selection_seed,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "token_contract_builder": builder_identity,
        "tasks": len(task_ids),
        "candidate_pairs": len(pair_ids),
        "candidate_pair_atom_split": False,
        "rows": len(rows),
        "expected_rows": (
            len(task_ids) if selector == "flawless_only" else 2 * len(task_ids)
        ),
        "supervised_tokens": actual_sup,
        "nonpadding_tokens": actual_nonpad,
        "budget_comparable_primary": budget.get(
            "budget_comparable_primary", selector != "flawless_only"
        ),
        "failed_action_positive_labels": failed_labels,
        "failed_call_and_error_result_are_context_only": True,
        "full_nonfailed_assistant_suffix_supervised": True,
        "assistant_tool_calls_and_text_supervised": True,
        "v5_tool_action_only_trainer_compatible": False,
        "required_trainer_objective": "v6_full_assistant_suffix",
        "future_clean_suffix_used": False,
        "official_test_used": False,
        "row_content_sha256": canonical_sha256(rows),
    }
    if audit["rows"] != audit["expected_rows"]:
        raise V6MaterializationError(
            "materialized row count proves that a candidate pair was split"
        )
    return rows, audit


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V6MaterializationError(f"{path}: invalid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise V6MaterializationError(f"{path}: root must be an object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(canonical(row) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.audit_output.exists():
        raise V6MaterializationError("output and audit paths must both be absent")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    manifest = read_json(args.manifest)
    rows, audit = materialize_manifest(
        manifest,
        manifest_sha256=sha256_file(args.manifest),
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer=tokenizer,
    )
    write_jsonl(args.output, rows)
    audit["output_jsonl_sha256"] = sha256_file(args.output)
    audit["source_manifest_path"] = str(args.manifest)
    write_json(args.audit_output, audit)


if __name__ == "__main__":
    main()
