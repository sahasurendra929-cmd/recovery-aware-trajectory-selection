#!/usr/bin/env python3
"""Materialize the complete V5.5 five-arm, message-masked SFT dataset.

The input is an independently audited counterfactual-pair JSONL.  Every pair
provides two views of the same successful suffix:

* clean: shared prefix -> successful suffix
* recovery: shared prefix -> failed read-only call/result -> successful suffix

Only assistant messages in the successful suffix are labels.  The failed call
and failed tool result are always context.  A 512-row schedule is built from
128 four-row blocks.  Each block repeats one source pair four times, so
selecting 0/1/2/3/4 recovery rows per block realizes exactly
0/25/50/75/100 percent recovery *supervised-token mass*, not merely an
approximate row ratio.

The reference-grounded V5.5 pair set may be used for a diagnostic screen.
Natural-conversation pairs that implement the same audited schema may be used
for the confirmatory training pool.  The output audit records the claim
boundary and fails closed before training artifacts are authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    import prepare_v5_sft_causal as v5
    import v5_5_full_protocol as full
    import v5_5_protocol as pair_protocol
except ModuleNotFoundError:
    from scripts import prepare_v5_sft_causal as v5
    from scripts import v5_5_full_protocol as full
    from scripts import v5_5_protocol as pair_protocol


DATA_PROTOCOL = "v5_5_full_sft_data_v1"
SCHEDULE_ROWS = 512
BLOCK_SIZE = 4
BLOCK_COUNT = SCHEDULE_ROWS // BLOCK_SIZE
ARM_MAP = {
    "perfect_success": ("r0_perfect", 0),
    "repair_25": ("r25", 1),
    "repair_50": ("r50", 2),
    "repair_75": ("r75", 3),
    "repair_100": ("r100_recovery", 4),
}
PAIR_MODES = {"reference", "natural"}


class V55DataError(RuntimeError):
    """V5.5 data cannot safely authorize training."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(v5.canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise V55DataError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V55DataError(f"{label} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise V55DataError(f"{label} must be a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise V55DataError(f"pair JSONL does not exist: {path}")
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise V55DataError(
                    f"{path}:{line_number}: every JSONL row must end in newline"
                )
            if not line.strip():
                raise V55DataError(f"{path}:{line_number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise V55DataError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(row, dict):
                raise V55DataError(f"{path}:{line_number}: row must be an object")
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise V55DataError(f"{path}:{line_number}: missing pair_id")
            if pair_id in identifiers:
                raise V55DataError(f"{path}:{line_number}: duplicate pair_id")
            identifiers.add(pair_id)
            rows.append(row)
    if not rows:
        raise V55DataError("pair JSONL is empty")
    return rows


def _synthetic_reference_user(pair: dict[str, Any]) -> dict[str, Any]:
    """Expose the reference-task metadata only for the diagnostic screen."""
    context = pair.get("task_context")
    if not isinstance(context, dict):
        raise V55DataError(f"{pair.get('pair_id')}: reference pair lacks task_context")
    return {
        "role": "user",
        "content": (
            "Controlled reference-action diagnostic. Complete the following "
            "customer-service task. This structured task metadata is used only "
            "for a mechanism screen and is not a natural-dialogue claim.\n"
            + v5.canonical(context)
        ),
    }


def _conversation_prefix(
    pair: dict[str, Any],
    *,
    pair_mode: str,
    field: str,
) -> list[dict[str, Any]]:
    value = pair.get(field)
    if not isinstance(value, list):
        raise V55DataError(f"{pair.get('pair_id')}: {field} must be a list")
    prefix = deepcopy(value)
    if pair_mode == "reference":
        if any(message.get("role") == "user" for message in prefix):
            raise V55DataError(
                f"{pair.get('pair_id')}: reference prefix unexpectedly contains user"
            )
        prefix.insert(0, _synthetic_reference_user(pair))
    elif not any(
        isinstance(message, dict) and message.get("role") == "user"
        for message in prefix
    ):
        raise V55DataError(
            f"{pair.get('pair_id')}: natural pair has no observed user message"
        )
    return prefix


def _selected_suffix_indices(
    messages: list[dict[str, Any]],
    *,
    suffix_start: int,
    pair_id: str,
) -> list[int]:
    selected: list[int] = []
    for index in range(suffix_start, len(messages)):
        message = messages[index]
        if not isinstance(message, dict):
            raise V55DataError(f"{pair_id}: message {index} is not an object")
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        if calls:
            if (
                len(calls) != 1
                or index + 1 >= len(messages)
                or messages[index + 1].get("role") != "tool"
                or messages[index + 1].get("error") is not False
            ):
                raise V55DataError(
                    f"{pair_id}: supervised tool call {index} lacks one "
                    "adjacent successful result"
                )
            selected.append(index)
        elif message.get("content") in (None, ""):
            raise V55DataError(f"{pair_id}: empty supervised assistant message")
        # V5.5 retains final natural-language responses as context/evidence,
        # but the frozen Stage-1 objective labels tool actions only.
    if not selected:
        raise V55DataError(f"{pair_id}: successful suffix has no assistant label")
    return selected


def _materialize_view(
    pair: dict[str, Any],
    *,
    pair_mode: str,
    recovery: bool,
    context: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    pair_id = str(pair["pair_id"])
    prefix_field = "recovery_prompt" if recovery else "clean_prefix"
    prefix = _conversation_prefix(pair, pair_mode=pair_mode, field=prefix_field)
    suffix = pair.get("supervised_messages")
    if not isinstance(suffix, list) or not suffix:
        raise V55DataError(f"{pair_id}: supervised_messages must be non-empty")
    raw_messages = [*prefix, *deepcopy(suffix)]
    selected_old = _selected_suffix_indices(
        raw_messages,
        suffix_start=len(prefix),
        pair_id=pair_id,
    )
    # The frozen trainer requires the final supervised tool call to retain
    # exactly its adjacent result and no later messages.  Later assistant text
    # is not part of the tool-action objective.
    raw_messages = raw_messages[: selected_old[-1] + 2]
    messages, index_map = v5.neutralize_messages(raw_messages, context)
    selected_new = {index_map[index] for index in selected_old}
    label_mask = [
        index in selected_new for index in range(len(messages))
    ]
    failed_indices = [
        index
        for index in range(len(messages) - 1)
        if messages[index].get("role") == "assistant"
        and messages[index].get("tool_calls")
        and messages[index + 1].get("role") == "tool"
        and messages[index + 1].get("error") is True
    ]
    if recovery:
        if len(failed_indices) != 1:
            raise V55DataError(
                f"{pair_id}: recovery view requires exactly one failed tool call"
            )
        if label_mask[failed_indices[0]]:
            raise V55DataError(f"{pair_id}: failed call entered the positive label")
        source = "failure_rich"
    else:
        if failed_indices:
            raise V55DataError(f"{pair_id}: clean view contains a tool error")
        source = "perfect_success"
    if any(
        messages[index].get("role") != "assistant"
        for index, selected in enumerate(label_mask)
        if selected
    ):
        raise V55DataError(f"{pair_id}: a non-assistant message entered the label")
    schemas = context["tool_schemas"]
    token_contract = v5.token_contract(
        tokenizer,
        messages,
        label_mask,
        schemas,
    )
    variant = "recovery" if recovery else "clean"
    source_id = f"{pair_id}:{variant}"
    row = {
        "id": source_id,
        "messages": messages,
        "label_mask": label_mask,
        "metadata": {
            "arm": "repair_100" if recovery else "perfect_success",
            "source": source,
            "domain": str(pair["domain"]),
            "task_id": str(pair["task_id"]),
            "trial": str(pair.get("mutation_variant", pair_id)),
            "seed": full.TRAINING_SEEDS[0],
            "pair_id": pair_id,
            "source_example_id": source_id,
            "source_pair_id": pair_id,
            "source_split": "inner_train",
            "fit_split": "train_schedule",
            "fault_family": str(
                pair.get("fault_family", "identifier_lookup_counterfactual")
            ),
            "fault_tool_name": str(pair["injected_call"]["name"]),
            "fault_relevance": "reference_path_or_operation_aligned",
            "fault_on_reference_path": True,
            "failed_assistant_message_indices": failed_indices,
            "injected_failed_assistant_message_index": (
                failed_indices[0] if failed_indices else None
            ),
            "tool_schemas": schemas,
            "tool_schemas_sha256": v5.sha256_text(v5.canonical(schemas)),
            "call_id_policy": "condition_neutral_sequential",
            "full_trajectory_reward": 1.0,
            "source_trajectory_sha256": v5.sha256_text(v5.canonical(pair)),
            "pair_mode": pair_mode,
            "failed_action_label_messages": 0,
            "official_test_used": False,
        },
        "token_contract": token_contract,
    }
    # Exercise the exact trainer-side message contract before scheduling.
    v5_train = __import__(
        "scripts.train_v5_sft_causal",
        fromlist=["validate_row"],
    )
    v5_train.validate_row(
        row,
        arm="repair_100" if recovery else "perfect_success",
        split="train",
    )
    return row


def materialize_source_pairs(
    pairs: list[dict[str, Any]],
    *,
    pair_mode: str,
    contexts: dict[str, dict[str, Any]],
    tokenizer: Any,
    independent_replay_authorized: bool,
) -> list[dict[str, Any]]:
    source_pairs: list[dict[str, Any]] = []
    for pair in sorted(pairs, key=lambda item: str(item["pair_id"])):
        audit_view = deepcopy(pair)
        # The independent auditor intentionally does not rewrite the producer
        # JSONL.  Its PASS bundle is the authority for this one field.
        if independent_replay_authorized:
            audit_view["independent_environment_replay_pass"] = True
        checks = pair_protocol.audit_pair(audit_view)
        if not checks or not all(checks.values()):
            failed = sorted(key for key, value in checks.items() if not value)
            raise V55DataError(
                f"{pair['pair_id']}: pair audit failed: {failed}"
            )
        domain = str(pair.get("domain"))
        if domain not in contexts:
            raise V55DataError(f"{pair['pair_id']}: unsupported domain {domain!r}")
        clean = _materialize_view(
            pair,
            pair_mode=pair_mode,
            recovery=False,
            context=contexts[domain],
            tokenizer=tokenizer,
        )
        recovery = _materialize_view(
            pair,
            pair_mode=pair_mode,
            recovery=True,
            context=contexts[domain],
            tokenizer=tokenizer,
        )
        if clean["token_contract"]["supervised_tokens"] != recovery[
            "token_contract"
        ]["supervised_tokens"]:
            raise V55DataError(
                f"{pair['pair_id']}: clean/recovery supervised-token mass differs"
            )
        clean_targets = [
            message
            for message, selected in zip(
                clean["messages"], clean["label_mask"]
            )
            if selected
        ]
        recovery_targets = [
            message
            for message, selected in zip(
                recovery["messages"], recovery["label_mask"]
            )
            if selected
        ]
        def semantic_targets(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for message in messages:
                item = deepcopy(message)
                for call in item.get("tool_calls") or []:
                    call.pop("id", None)
                item.pop("tool_call_id", None)
                item.pop("id", None)
                result.append(item)
            return result

        canonical_targets = semantic_targets(clean_targets)
        if v5.canonical(canonical_targets) != v5.canonical(
            semantic_targets(recovery_targets)
        ):
            raise V55DataError(
                f"{pair['pair_id']}: clean/recovery supervised targets differ"
            )
        source_pairs.append(
            {
                "pair_id": str(pair["pair_id"]),
                "task_identity": str(pair["task_identity"]),
                "clean": clean,
                "recovery": recovery,
                "supervised_tokens": clean["token_contract"]["supervised_tokens"],
                "target_sha256": v5.sha256_text(v5.canonical(canonical_targets)),
            }
        )
    return source_pairs


def deterministic_pair_order(
    source_pairs: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    return sorted(
        source_pairs,
        key=lambda pair: (
            hashlib.sha256(
                f"{DATA_PROTOCOL}|{seed}|{pair['pair_id']}".encode("utf-8")
            ).hexdigest(),
            pair["pair_id"],
        ),
    )


def build_arm_schedules(
    source_pairs: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if not source_pairs:
        raise V55DataError("cannot schedule an empty source-pair pool")
    ordered = deterministic_pair_order(source_pairs, seed=seed)
    schedules: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, Any] = {}
    exposure_template: list[str] = []
    for block_index in range(BLOCK_COUNT):
        exposure_template.append(ordered[block_index % len(ordered)]["pair_id"])
    for trainer_arm, (paper_arm, recovery_per_block) in ARM_MAP.items():
        rows: list[dict[str, Any]] = []
        for block_index, pair_id in enumerate(exposure_template):
            source = next(pair for pair in ordered if pair["pair_id"] == pair_id)
            recovery_slots = {
                (slot + block_index) % BLOCK_SIZE
                for slot in range(recovery_per_block)
            }
            for slot in range(BLOCK_SIZE):
                recovery = slot in recovery_slots
                row = deepcopy(source["recovery" if recovery else "clean"])
                row["id"] = (
                    f"{paper_arm}:block-{block_index:03d}:slot-{slot}:"
                    f"{source['pair_id']}"
                )
                row["metadata"]["arm"] = trainer_arm
                row["metadata"]["paper_arm"] = paper_arm
                row["metadata"]["schedule_block"] = block_index
                row["metadata"]["schedule_slot"] = slot
                row["metadata"]["schedule_seed"] = seed
                rows.append(row)
        supervised = sum(
            row["token_contract"]["supervised_tokens"] for row in rows
        )
        recovery_tokens = sum(
            row["token_contract"]["supervised_tokens"]
            for row in rows
            if row["metadata"]["source"] == "failure_rich"
        )
        recovery_rows = sum(
            row["metadata"]["source"] == "failure_rich" for row in rows
        )
        expected_ratio = full.RECOVERY_RATIOS[paper_arm]
        observed_ratio = recovery_tokens / supervised
        if (
            len(rows) != SCHEDULE_ROWS
            or recovery_rows != BLOCK_COUNT * recovery_per_block
            or abs(observed_ratio - expected_ratio) > 1e-12
        ):
            raise V55DataError(f"{paper_arm}: exact dose construction failed")
        if any(
            row["metadata"]["failed_action_label_messages"] != 0
            for row in rows
        ):
            raise V55DataError(f"{paper_arm}: failed action became a label")
        schedules[trainer_arm] = rows
        audits[trainer_arm] = {
            "paper_arm": paper_arm,
            "rows": len(rows),
            "recovery_rows": recovery_rows,
            "supervised_tokens": supervised,
            "recovery_supervised_tokens": recovery_tokens,
            "realized_recovery_supervised_token_ratio": observed_ratio,
            "expected_recovery_supervised_token_ratio": expected_ratio,
            "distinct_pairs": len({row["metadata"]["pair_id"] for row in rows}),
            "distinct_tasks": len(
                {
                    f"{row['metadata']['domain']}:{row['metadata']['task_id']}"
                    for row in rows
                }
            ),
            "failed_action_label_messages": 0,
        }
    exposure_by_arm = {
        arm: [
            row["metadata"]["pair_id"]
            for index, row in enumerate(rows)
            if index % BLOCK_SIZE == 0
        ]
        for arm, rows in schedules.items()
    }
    if len({tuple(values) for values in exposure_by_arm.values()}) != 1:
        raise V55DataError("arm source-pair exposure drift")
    return schedules, audits


def _validate_input_authorization(
    *,
    pair_mode: str,
    pairs: list[dict[str, Any]],
    pair_audit: dict[str, Any],
    pair_manifest: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    if pair_mode not in PAIR_MODES:
        raise V55DataError(f"unsupported pair mode {pair_mode!r}")
    if pair_audit.get("status") != "PASS_TRAINING_AUTHORIZED":
        raise V55DataError("pair audit did not authorize training")
    if pair_audit.get("checks", {}).get("official_test_sealed") is not True:
        raise V55DataError("pair audit lacks the official-test seal")
    if pair_manifest.get("official_test_used") is not False:
        raise V55DataError("pair manifest opened the official test")
    task_ids = {str(pair.get("task_identity")) for pair in pairs}
    domains = {str(pair.get("domain")) for pair in pairs}
    if strict:
        if pair_mode == "reference":
            if (
                pair_manifest.get("protocol") != pair_protocol.PROTOCOL
                or len(pairs) != full.REFERENCE_PILOT_PAIRS
                or len(task_ids) != full.REFERENCE_PILOT_TASKS
            ):
                raise V55DataError("reference pair pool differs from the frozen 24/48 set")
        elif (
            len(pairs) < full.MIN_NATURAL_PAIRS
            or len(task_ids) < full.MIN_NATURAL_TRAIN_TASKS
        ):
            raise V55DataError(
                "natural pair pool is below the frozen 24-task/48-pair gate"
            )
        if domains != {"airline", "retail"}:
            raise V55DataError("training pool must cover airline and retail")
    return {
        "pair_mode": pair_mode,
        "pairs": len(pairs),
        "tasks": len(task_ids),
        "domains": sorted(domains),
        "scientific_claim_level": (
            "diagnostic_only"
            if pair_mode == "reference"
            else "natural_counterfactual_training_candidate"
        ),
    }


def prepare(
    *,
    tau2_root: Path,
    pairs_path: Path,
    pair_audit_path: Path,
    pair_manifest_path: Path,
    output_dir: Path,
    tokenizer: Any,
    tokenizer_name: str,
    tokenizer_revision: str,
    pair_mode: str,
    schedule_seed: int = full.TRAINING_SEEDS[0],
    strict: bool = True,
) -> dict[str, Any]:
    full.validate_design()
    if output_dir.exists():
        raise V55DataError(f"output directory must be absent: {output_dir}")
    if schedule_seed not in full.TRAINING_SEEDS:
        raise V55DataError("schedule seed is not a frozen V5.5 training seed")
    pairs = read_jsonl(pairs_path)
    pair_audit = read_json(pair_audit_path, label="pair audit")
    pair_manifest = read_json(pair_manifest_path, label="pair manifest")
    input_audit = _validate_input_authorization(
        pair_mode=pair_mode,
        pairs=pairs,
        pair_audit=pair_audit,
        pair_manifest=pair_manifest,
        strict=strict,
    )
    contexts = v5.load_tau2_contexts(tau2_root)
    source_pairs = materialize_source_pairs(
        pairs,
        pair_mode=pair_mode,
        contexts=contexts,
        tokenizer=tokenizer,
        independent_replay_authorized=True,
    )
    schedules, arm_audits = build_arm_schedules(
        source_pairs,
        seed=schedule_seed,
    )
    output_dir.mkdir(parents=True)
    for arm, rows in schedules.items():
        path = output_dir / "arms" / arm / "train.jsonl"
        write_jsonl(path, rows)
        arm_audits[arm]["sha256"] = sha256_file(path)
    validation_path = output_dir / "validation_loss.jsonl"
    validation_path.write_bytes(b"")
    source_index = [
        {
            "pair_id": pair["pair_id"],
            "task_identity": pair["task_identity"],
            "supervised_tokens": pair["supervised_tokens"],
            "target_sha256": pair["target_sha256"],
        }
        for pair in source_pairs
    ]
    write_json(output_dir / "source_pairs.json", source_index)
    audit = {
        "protocol": DATA_PROTOCOL,
        "design_protocol": full.PROTOCOL,
        "design_version": full.DESIGN_VERSION,
        "status": "PASS",
        "schedule_seed": schedule_seed,
        "model_tokenizer": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "pair_mode": pair_mode,
        "input_authorization": input_audit,
        "pair_manifest_sha256": sha256_file(pair_manifest_path),
        "pair_audit_sha256": sha256_file(pair_audit_path),
        "pairs_jsonl_sha256": sha256_file(pairs_path),
        "source_pairs_sha256": sha256_file(output_dir / "source_pairs.json"),
        "official_test_used": False,
        "official_test_sealed": True,
        "derived_validation_used_for_supervision": False,
        "train_validation_source_overlap": 0,
        "training_mixture_basis": full.TRAINING_MIXTURE_BASIS,
        "same_source_pair_exposure_across_arms": True,
        "same_supervised_target_within_clean_recovery_pair": True,
        "training_fault_tools": sorted(
            {
                f"{pair['domain']}:{pair['injected_call']['name']}"
                for pair in pairs
            }
        ),
        "training_fault_families": sorted(
            {
                str(
                    pair.get(
                        "fault_family",
                        "identifier_lookup_counterfactual",
                    )
                )
                for pair in pairs
            }
        ),
        "arms": arm_audits,
        "validation_loss": {
            "rows": 0,
            "sha256": sha256_file(validation_path),
            "source_split": "not_applicable_fixed_step_screen",
            "used_for_checkpoint_selection": False,
            "validation_disabled_reason": "fixed_steps_v5_5",
        },
        "label_guarantees": {
            "failed_action_positive_labels": 0,
            "official_test_and_derived_validation_label_leakage": 0,
            "failed_call_and_result_are_context_only": True,
            "post_error_successful_assistant_actions_only": True,
        },
        "claim_boundary": {
            "reference_mode_is_diagnostic_only": pair_mode == "reference",
            "task_success_requires_end_to_end_tau2_evaluation": True,
            "training_authorization_is_not_a_positive_scientific_result": True,
        },
    }
    write_json(output_dir / "audit.json", audit)
    hashes = {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    }
    write_json(output_dir / "hashes.json", hashes)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-audit", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-mode", choices=sorted(PAIR_MODES), required=True)
    parser.add_argument("--tokenizer", default=full.MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--schedule-seed", type=int, default=full.TRAINING_SEEDS[0])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    result = prepare(
        tau2_root=args.tau2_root.resolve(),
        pairs_path=args.pairs.resolve(),
        pair_audit_path=args.pair_audit.resolve(),
        pair_manifest_path=args.pair_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        pair_mode=args.pair_mode,
        schedule_seed=args.schedule_seed,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir.resolve()),
                "pair_mode": args.pair_mode,
                "arms": sorted(result["arms"]),
                "rows_per_arm": SCHEDULE_ROWS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
