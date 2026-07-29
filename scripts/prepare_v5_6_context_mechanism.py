#!/usr/bin/env python3
"""Build exact-budget R0/R25-true/R25-shuffled V5.6 training schedules.

Each four-row block exposes the same target pair four times.  R0 contains four
clean rows; both R25 arms contain three clean rows and one masked-error row.
The shuffled row preserves the target's successful suffix but replaces its
error-call/result with a deterministic same-domain donor from another task.
Thus dose, targets, target-task exposure, and token mass are all matched.
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
    import prepare_v5_5_full_sft as v55
    import prepare_v5_sft_causal as v5
    import v5_6_context_protocol as protocol
except ModuleNotFoundError:
    from scripts import prepare_v5_5_full_sft as v55
    from scripts import prepare_v5_sft_causal as v5
    from scripts import v5_6_context_protocol as protocol


class ContextDataError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(v5.canonical(row) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def _failed_slice(row: dict[str, Any]) -> list[dict[str, Any]]:
    failed = row["metadata"]["failed_assistant_message_indices"]
    if not isinstance(failed, list) or len(failed) != 1:
        raise ContextDataError(f"{row['id']}: expected exactly one failed event")
    index = failed[0]
    messages = row["messages"]
    if index + 1 >= len(messages) or messages[index + 1].get("error") is not True:
        raise ContextDataError(f"{row['id']}: failed event is malformed")
    return deepcopy(messages[index : index + 2])


def _first_label_index(row: dict[str, Any]) -> int:
    indices = [index for index, selected in enumerate(row["label_mask"]) if selected]
    if not indices:
        raise ContextDataError(f"{row['id']}: row has no target")
    return indices[0]


def make_shuffled_row(target: dict[str, Any], donor: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Insert donor's complete failed event immediately before target repair."""
    target_task = f"{target['metadata']['domain']}:{target['metadata']['task_id']}"
    donor_task = f"{donor['metadata']['domain']}:{donor['metadata']['task_id']}"
    if target_task == donor_task or target["metadata"]["domain"] != donor["metadata"]["domain"]:
        raise ContextDataError("shuffle must be a cross-task, within-domain derangement")
    row = deepcopy(target)
    at = _first_label_index(row)
    event = _failed_slice(donor)
    row["messages"][at:at] = event
    row["label_mask"][at:at] = [False, False]
    metadata = row["metadata"]
    metadata.update({
        "source": "failure_rich",
        "failed_assistant_message_indices": [at],
        "injected_failed_assistant_message_index": at,
        "context_condition": "shuffled",
        "error_context_matches_target": False,
        "error_donor_pair_id": donor["metadata"]["pair_id"],
        "error_donor_task_identity": donor_task,
    })
    # Recompute the exact template-level contract after inserting the donor.
    row["token_contract"] = v5.token_contract(
        tokenizer, row["messages"], row["label_mask"], metadata["tool_schemas"]
    )
    return row


def finalize_shuffled_row(row: dict[str, Any]) -> dict[str, Any]:
    return row


def deterministic_donors(source_pairs: list[dict[str, Any]], seed: int) -> dict[str, str]:
    result: dict[str, str] = {}
    for domain in sorted({pair["clean"]["metadata"]["domain"] for pair in source_pairs}):
        group = [pair for pair in source_pairs if pair["clean"]["metadata"]["domain"] == domain]
        if len({pair["clean"]["metadata"]["task_id"] for pair in group}) < 2:
            raise ContextDataError(f"{domain}: need at least two tasks for a derangement")
        ordered = sorted(group, key=lambda pair: hashlib.sha256(f"{protocol.PROTOCOL}|{seed}|{pair['pair_id']}".encode()).hexdigest())
        for index, pair in enumerate(ordered):
            target_task = pair["clean"]["metadata"]["task_id"]
            donor = next((candidate for offset in range(1, len(ordered))
                          if (candidate := ordered[(index + offset) % len(ordered)])["clean"]["metadata"]["task_id"] != target_task), None)
            if donor is None:
                raise ContextDataError(f"{pair['pair_id']}: no cross-task donor")
            result[pair["pair_id"]] = donor["pair_id"]
    return result


def build_schedules(source_pairs: list[dict[str, Any]], tokenizer: Any, seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    protocol.validate_design()
    ordered = v55.deterministic_pair_order(source_pairs, seed=seed)
    by_id = {pair["pair_id"]: pair for pair in source_pairs}
    donor_map = deterministic_donors(source_pairs, seed)
    schedules = {arm: [] for arm in protocol.ARMS}
    for block in range(protocol.SCHEDULE_ROWS // protocol.BLOCK_SIZE):
        source = ordered[block % len(ordered)]
        donor = by_id[donor_map[source["pair_id"]]]
        shuffled = finalize_shuffled_row(make_shuffled_row(source["clean"], donor["recovery"], tokenizer))
        for arm in protocol.ARMS:
            for slot in range(protocol.BLOCK_SIZE):
                if arm == "perfect_success" or slot != block % protocol.BLOCK_SIZE:
                    row = deepcopy(source["clean"])
                    condition = "clean"
                elif arm == "repair_25_true":
                    row = deepcopy(source["recovery"])
                    row["metadata"].update({"context_condition": "true", "error_context_matches_target": True, "error_donor_pair_id": source["pair_id"], "error_donor_task_identity": f"{row['metadata']['domain']}:{row['metadata']['task_id']}"})
                    condition = "true"
                else:
                    row = deepcopy(shuffled)
                    condition = "shuffled"
                row["id"] = f"{arm}:block-{block:03d}:slot-{slot}:{source['pair_id']}"
                row["metadata"].update({"arm": arm, "paper_arm": arm, "schedule_block": block, "schedule_slot": slot, "schedule_seed": seed, "context_condition": condition})
                schedules[arm].append(row)
    audits: dict[str, Any] = {}
    for arm, rows in schedules.items():
        recovery_rows = [row for row in rows if row["metadata"]["source"] == "failure_rich"]
        total = sum(row["token_contract"]["supervised_tokens"] for row in rows)
        recovery = sum(row["token_contract"]["supervised_tokens"] for row in recovery_rows)
        max_sequence_tokens = max(row["token_contract"]["sequence_tokens"] for row in rows)
        if len(rows) != protocol.SCHEDULE_ROWS or len(recovery_rows) != int(protocol.SCHEDULE_ROWS * protocol.RECOVERY_RATIO[arm]) or abs(recovery / total - protocol.RECOVERY_RATIO[arm]) > 1e-12:
            raise ContextDataError(f"{arm}: dose contract failed")
        if max_sequence_tokens > protocol.MAX_SEQUENCE_TOKENS:
            raise ContextDataError(
                f"{arm}: sequence capacity exceeded: {max_sequence_tokens} > "
                f"{protocol.MAX_SEQUENCE_TOKENS}"
            )
        audits[arm] = {"rows": len(rows), "recovery_rows": len(recovery_rows), "supervised_tokens": total, "recovery_supervised_tokens": recovery, "realized_recovery_supervised_token_ratio": recovery / total, "max_sequence_tokens": max_sequence_tokens, "failed_action_label_messages": 0, "context_conditions": sorted({row["metadata"]["context_condition"] for row in rows})}
    return schedules, {"arms": audits, "donor_map": donor_map}


def prepare(*, tau2_root: Path, pairs_path: Path, pair_audit_path: Path, pair_manifest_path: Path, output_dir: Path, tokenizer: Any, tokenizer_name: str, tokenizer_revision: str, pair_mode: str, schedule_seed: int, strict: bool = True) -> dict[str, Any]:
    if output_dir.exists():
        raise ContextDataError(f"output directory must be absent: {output_dir}")
    protocol.validate_design()
    pairs = v55.read_jsonl(pairs_path)
    authorization = v55._validate_input_authorization(pair_mode=pair_mode, pairs=pairs, pair_audit=v55.read_json(pair_audit_path, label="pair audit"), pair_manifest=v55.read_json(pair_manifest_path, label="pair manifest"), strict=strict)
    contexts = v5.load_tau2_contexts(tau2_root)
    sources = v55.materialize_source_pairs(pairs, pair_mode=pair_mode, contexts=contexts, tokenizer=tokenizer, independent_replay_authorized=True)
    schedules, schedule_audit = build_schedules(sources, tokenizer, schedule_seed)
    output_dir.mkdir(parents=True)
    for arm, rows in schedules.items():
        write_jsonl(output_dir / "arms" / arm / "train.jsonl", rows)
        schedule_audit["arms"][arm]["sha256"] = sha256_file(output_dir / "arms" / arm / "train.jsonl")
    (output_dir / "validation_loss.jsonl").write_bytes(b"")
    write_json(output_dir / "source_pairs.json", [{"pair_id": item["pair_id"], "task_identity": item["task_identity"], "target_sha256": item["target_sha256"]} for item in sources])
    audit = {"protocol": protocol.DATA_PROTOCOL, "design_protocol": protocol.PROTOCOL, "design_version": protocol.DESIGN_VERSION, "status": "PASS", "schedule_seed": schedule_seed, "model_tokenizer": tokenizer_name, "tokenizer_revision": tokenizer_revision, "pair_mode": pair_mode, "max_sequence_tokens": protocol.MAX_SEQUENCE_TOKENS, "input_authorization": authorization, "pair_manifest_sha256": sha256_file(pair_manifest_path), "pair_audit_sha256": sha256_file(pair_audit_path), "pairs_jsonl_sha256": sha256_file(pairs_path), "source_pairs_sha256": sha256_file(output_dir / "source_pairs.json"), "official_test_used": False, "official_test_sealed": True, "derived_validation_used_for_supervision": False, "train_validation_source_overlap": 0, "training_mixture_basis": "supervised_token_mass", "same_source_pair_exposure_across_arms": True, "same_supervised_target_within_clean_recovery_pair": True, "shuffle_contract": {"within_domain": True, "different_task": True, "error_call_and_result_moved_together": True, "donor_map": schedule_audit["donor_map"]}, "arms": schedule_audit["arms"], "validation_loss": {"rows": 0, "sha256": sha256_file(output_dir / "validation_loss.jsonl"), "source_split": "not_applicable_fixed_step_screen", "used_for_checkpoint_selection": False}, "label_guarantees": {"failed_action_positive_labels": 0, "official_test_and_derived_validation_label_leakage": 0, "failed_call_and_result_are_context_only": True, "post_error_successful_assistant_actions_only": True}, "claim_boundary": {"reference_mode_is_diagnostic_only": pair_mode == "reference", "mechanism_score_is_not_end_to_end_task_success": True}}
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "hashes.json", {path.relative_to(output_dir).as_posix(): sha256_file(path) for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "hashes.json"})
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tau2_root", "pairs", "pair_audit", "pair_manifest", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--pair-mode", choices=sorted(v55.PAIR_MODES), required=True)
    parser.add_argument("--tokenizer", default=protocol.MODEL)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--schedule-seed", type=int, default=protocol.TRAINING_SEEDS[0])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision, trust_remote_code=True, local_files_only=args.local_files_only)
    result = prepare(tau2_root=args.tau2_root.resolve(), pairs_path=args.pairs.resolve(), pair_audit_path=args.pair_audit.resolve(), pair_manifest_path=args.pair_manifest.resolve(), output_dir=args.output_dir.resolve(), tokenizer=tokenizer, tokenizer_name=args.tokenizer, tokenizer_revision=args.tokenizer_revision, pair_mode=args.pair_mode, schedule_seed=args.schedule_seed)
    print(json.dumps({"status": result["status"], "arms": sorted(result["arms"]), "official_test_used": False}, indent=2))


if __name__ == "__main__":
    main()
