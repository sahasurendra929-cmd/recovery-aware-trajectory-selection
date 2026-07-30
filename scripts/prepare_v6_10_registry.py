#!/usr/bin/env python3
"""Build the V6.10 executable, reference-preflight-bound pair registry.

The legacy V6 registry selected actions from reference *syntax*.  V6.10 first
executes every formal reference trace and admits only exact action indices
that:

1. remain in the zero-error sanitized reference plan;
2. succeed when moved to the first recovery position;
3. still reach the same final state; and
4. retain official tau2 reward 1.0.

This builder is outcome-independent with respect to every model.  It consumes
only the frozen split/config/task catalog and a PASS model-free reference
preflight receipt.  Injected-error execution is still rechecked at candidate
generation time before a model-derived trajectory can be accepted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

if __package__:
    from scripts import prepare_v6_candidate_registry as legacy
    from scripts import v6_10_selection_protocol as closure
    from scripts import v6_reference_contract as reference
    from scripts import v6_selection_protocol as base
else:  # pragma: no cover - direct script execution
    import prepare_v6_candidate_registry as legacy
    import v6_10_selection_protocol as closure
    import v6_reference_contract as reference
    import v6_selection_protocol as base


REGISTRY_PROTOCOL = closure.REGISTRY_PROTOCOL
REGISTRY_VERSION = "2.0"
PAIRS_PER_TASK = closure.PAIRS_PER_TASK
PHASE_SEED_SALTS = {
    "compatibility": 20260826,
    "pilot": 20260827,
    "formal": 20260828,
}
PHASE_TASK_IDS = {
    "compatibility": closure.COMPATIBILITY_TASK_IDS,
    "pilot": closure.PROSPECTIVE_PILOT_TASK_IDS,
    "formal": closure.FORMAL_TASK_IDS,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class V610RegistryError(RuntimeError):
    """The V6.10 executable registry cannot be frozen."""


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        "|".join(str(value) for value in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 1_000_001


def _task_key(identity: str) -> tuple[str, int]:
    domain, task_id = identity.split(":", 1)
    return domain, int(task_id)


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise V610RegistryError(
            f"cannot inspect git provenance in {root}: {arguments}"
        ) from error
    return completed.stdout.strip()


def source_provenance(root: Path) -> dict[str, Any]:
    commit = _git_output(root, "rev-parse", "HEAD")
    if GIT_COMMIT_RE.fullmatch(commit) is None:
        raise V610RegistryError("source HEAD is not a full 40-hex commit")
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise V610RegistryError(
            "source worktree contains tracked or untracked drift"
        )
    return {
        "commit": commit,
        "tracked_worktree_clean": True,
        "worktree_scope": "tracked_and_untracked_files",
    }


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("protocol") != closure.PROTOCOL:
        raise V610RegistryError("config does not freeze V6.10")
    benchmark = config.get("benchmark")
    populations = config.get("task_populations")
    reference_audit = config.get("reference_plan_audit")
    if not all(
        isinstance(value, Mapping)
        for value in (benchmark, populations, reference_audit)
    ):
        raise V610RegistryError("V6.10 config lacks required sections")
    if benchmark.get("commit") != base.TAU2_COMMIT:
        raise V610RegistryError("tau2 commit drift")
    official = benchmark.get("official_test")
    if (
        not isinstance(official, Mapping)
        or official.get("sealed") is not True
        or official.get("ids_or_content_exported_to_workers_before_unseal")
        is not False
    ):
        raise V610RegistryError("official test is not fail-closed")
    configured = {
        "compatibility": populations.get("compatibility_development"),
        "pilot": populations.get("prospective_pilot"),
        "formal": populations.get("formal_pool"),
    }
    for phase, row in configured.items():
        expected = list(PHASE_TASK_IDS[phase])
        if (
            not isinstance(row, Mapping)
            or row.get("task_ids") != expected
            or row.get("exact_tasks") != len(expected)
            or row.get("ordered_task_ids_sha256") != sha256(expected)
        ):
            raise V610RegistryError(f"{phase}: task population drift")
    if reference_audit.get("receipt_protocol") != reference.RECEIPT_PROTOCOL:
        raise V610RegistryError("reference preflight protocol drift")
    return {
        "tau2_commit": str(benchmark["commit"]),
        "partition_sha256": str(benchmark["partition_sha256"]),
        "split_manifest": str(benchmark["split_manifest"]),
    }


def _slot_by_index(
    preflight_row: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    slots = preflight_row.get("reference_slots")
    if not isinstance(slots, list) or not slots:
        raise V610RegistryError(
            f"{preflight_row.get('task_identity')}: missing reference slots"
        )
    by_index: dict[int, Mapping[str, Any]] = {}
    for position, slot in enumerate(slots):
        if (
            not isinstance(slot, Mapping)
            or slot.get("reference_action_index") != position
            or not isinstance(slot.get("reference_slot_id"), str)
            or not isinstance(slot.get("reference_slot_sha256"), str)
        ):
            raise V610RegistryError("preflight reference-slot identity drift")
        by_index[position] = slot
    return by_index


def executable_actions(
    *,
    task_identity: str,
    task: Mapping[str, Any],
    preflight_row: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return identifier-bearing, exact-index actions admitted by preflight."""
    if (
        preflight_row.get("task_identity") != task_identity
        or preflight_row.get("status") != "PASS"
    ):
        return []
    domain = task_identity.split(":", 1)[0]
    eligible = preflight_row.get("eligible_forced_first_reference_indices")
    sanitized = preflight_row.get("sanitized_successful_reference_indices")
    if not isinstance(eligible, list) or not isinstance(sanitized, list):
        raise V610RegistryError(f"{task_identity}: missing exact-index sets")
    if not set(eligible).issubset(set(sanitized)):
        raise V610RegistryError(
            f"{task_identity}: eligible slot lies outside sanitized plan"
        )
    slots = _slot_by_index(preflight_row)
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for action in legacy._reference_actions(domain=domain, task=task):
        index = int(action["reference_action_index"])
        if index not in set(eligible):
            continue
        slot = slots.get(index)
        if slot is None:
            raise V610RegistryError(
                f"{task_identity}: eligible action lacks exact slot"
            )
        call = action["reference_call"]
        semantics = sha256(
            {"requestor": call.get("requestor", "assistant"),
             "name": call["name"],
             "arguments": call["arguments"]}
        )
        if semantics != slot.get("call_semantics_sha256"):
            raise V610RegistryError(
                f"{task_identity}:{index}: task catalog/preflight call drift"
            )
        unique_key = (str(action["family"]), semantics)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        row = deepcopy(action)
        row.update(
            {
                "task_identity": task_identity,
                "reference_slot_id": slot["reference_slot_id"],
                "reference_slot_sha256": slot["reference_slot_sha256"],
                "call_semantics_sha256": semantics,
                "task_preflight_sha256": preflight_row[
                    "task_preflight_sha256"
                ],
                "sanitized_successful_plan_sha256": preflight_row[
                    "sanitized_successful_plan_sha256"
                ],
            }
        )
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            str(row["family"]),
            str(row["call_semantics_sha256"]),
            int(row["reference_action_index"]),
        ),
    )


def _pair_options(
    task_identity: str,
    actions: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    pairs = [
        pair
        for pair in combinations(actions, 2)
        if pair[0]["family"] != pair[1]["family"]
        and pair[0]["call_semantics_sha256"]
        != pair[1]["call_semantics_sha256"]
        and pair[0]["reference_action_index"]
        != pair[1]["reference_action_index"]
    ]
    if not pairs:
        raise V610RegistryError(
            f"{task_identity}: fewer than two distinct executable actions"
        )
    return sorted(
        pairs,
        key=lambda pair: (
            hashlib.sha256(
                canonical(
                    {
                        "protocol": REGISTRY_PROTOCOL,
                        "task_identity": task_identity,
                        "left": pair[0]["call_semantics_sha256"],
                        "right": pair[1]["call_semantics_sha256"],
                    }
                ).encode("utf-8")
            ).hexdigest(),
            str(pair[0]["family"]),
            str(pair[1]["family"]),
        ),
    )


def _branch(
    *,
    phase: str,
    task_identity: str,
    choice_set_id: str,
    candidate_pair_id: str,
    pair_index: int,
    branch_index: int,
    action: Mapping[str, Any],
    prefix_sha256: str,
    snapshot_sha256: str,
    reference_preflight_receipt_sha256: str,
    reference_preflight_file_sha256: str,
) -> dict[str, Any]:
    branch_id = f"{candidate_pair_id}:branch:{branch_index + 1}"
    mutation_salt = (
        f"{REGISTRY_PROTOCOL}|{phase}|{task_identity}|"
        f"{pair_index}|{branch_index}|{action['reference_slot_id']}"
    )
    original = str(action["original_identifier"])
    mutated = legacy.mutate_identifier(original, salt=mutation_salt)
    reference_call = deepcopy(dict(action["reference_call"]))
    error_call = deepcopy(reference_call)
    error_call["id"] = f"v610-error-{sha256([branch_id, mutation_salt])[:16]}"
    error_call["arguments"][str(action["argument_key"])] = mutated
    row = {
        "branch_id": branch_id,
        "candidate_pair_id": candidate_pair_id,
        "choice_set_id": choice_set_id,
        "phase": phase,
        "task_identity": task_identity,
        "branch_index": branch_index,
        "prefix_spec_sha256": prefix_sha256,
        "snapshot_spec_sha256": snapshot_sha256,
        "tool_name": str(action["tool_name"]),
        "identifier_key": str(action["argument_key"]),
        "original_identifier": original,
        "mutated_identifier": mutated,
        "corrective_family": str(action["family"]),
        "canonical_corrective_family": str(action["family"]),
        "reference_action_kind": str(action["reference_action_kind"]),
        "reference_arguments": deepcopy(reference_call["arguments"]),
        "reference_action_key": base.canonical_reference_action_key(
            str(action["tool_name"]),
            reference_call["arguments"],
        ),
        "reference_preflight_binding": {
            "reference_preflight_receipt_sha256": (
                reference_preflight_receipt_sha256
            ),
            "reference_preflight_file_sha256": (
                reference_preflight_file_sha256
            ),
            "reference_task_preflight_sha256": action[
                "task_preflight_sha256"
            ],
            "sanitized_reference_plan_sha256": action[
                "sanitized_successful_plan_sha256"
            ],
            "reference_action_index": int(action["reference_action_index"]),
            "reference_slot_id": action["reference_slot_id"],
            "reference_slot_sha256": action["reference_slot_sha256"],
            "call_semantics_sha256": action["call_semantics_sha256"],
        },
        "identifier_field_allowlisted": True,
        "identifier_mutation_type_preserving": True,
        "recovery_seed": _seed(
            REGISTRY_PROTOCOL,
            PHASE_SEED_SALTS[phase],
            branch_id,
        ),
        "injection_spec": {
            "status": "FROZEN_RUNTIME_VERIFICATION_REQUIRED",
            "site_tool_allowlist": [str(action["tool_name"])],
            "selected_site_tool": str(action["tool_name"]),
            "operator": "replace_one_allowlisted_identifier_argument",
            "argument_key": str(action["argument_key"]),
            "original_identifier": original,
            "replacement": {
                "source": "deterministic_type_preserving_mutation",
                "value": mutated,
                "mutation_salt_sha256": hashlib.sha256(
                    mutation_salt.encode("utf-8")
                ).hexdigest(),
            },
            "error_call": error_call,
            "expected_error_predicate": {
                "json_path": "$.error",
                "operator": "is",
                "value": True,
            },
            "expected_state_mutating": False,
            "required_runtime_evidence": [
                "actual_tool_error",
                "agent_db_hash_before_equals_after_error",
                (
                    "user_db_hash_before_equals_after_error_"
                    "or_explicitly_unavailable"
                ),
            ],
        },
        "corrective_action_spec": {
            "canonical_family": str(action["family"]),
            "forced_first_action_constructor": {
                "kind": "exact_registered_reference_tool_call",
                "tool_call": reference_call,
            },
            "reference_action_index": int(action["reference_action_index"]),
            "reference_slot_id": action["reference_slot_id"],
            "reference_slot_sha256": action["reference_slot_sha256"],
            "reference_action_kind": str(action["reference_action_kind"]),
            "sanitized_plan_member": True,
            "gold_future_visible": False,
        },
        "status": "REGISTERED_INJECTION_PREFLIGHT_REQUIRED",
        "official_test_used": False,
    }
    row["branch_slot_sha256"] = sha256(row)
    return row


def build_registry(
    *,
    split: Mapping[str, Any],
    config: Mapping[str, Any],
    task_catalog: Mapping[str, Mapping[str, Any]],
    preflight_receipt: Mapping[str, Any],
    preflight_file_sha256: str,
    split_file_sha256: str,
    config_file_sha256: str,
    task_file_sha256: Mapping[str, str] | None = None,
    source: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    closure.validate_constants()
    config_contract = validate_config(config)
    arm_train, arm_by_domain, forbidden = legacy.frozen_arm_train(
        split,
        strict=strict,
    )
    if set(closure.FORMAL_TASK_IDS) - set(arm_train):
        raise V610RegistryError("formal task universe is outside arm-train")
    if set(closure.FORMAL_TASK_IDS) & (
        forbidden["sealed"] | forbidden["validation"]
    ):
        raise V610RegistryError("formal registry overlaps a forbidden split")
    try:
        by_task = reference.verify_preflight_receipt(
            preflight_receipt,
            expected_task_ids=closure.FORMAL_TASK_IDS,
            expected_source_commit=(
                str(source["commit"]) if source is not None else None
            ),
            expected_tau2_commit=base.TAU2_COMMIT,
            expected_config_sha256=config_file_sha256,
            expected_split_manifest_sha256=split_file_sha256,
        )
    except reference.ReferenceContractError as error:
        raise V610RegistryError(str(error)) from error
    if (
        not isinstance(preflight_file_sha256, str)
        or SHA256_RE.fullmatch(preflight_file_sha256) is None
    ):
        raise V610RegistryError("preflight file hash is invalid")

    actions_by_task: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    for task_identity in closure.FORMAL_TASK_IDS:
        row = by_task[task_identity]
        if row.get("status") != "PASS":
            exclusions.append(
                {
                    "task_identity": task_identity,
                    "reason_code": "REFERENCE_PREFLIGHT_REJECTED",
                    "preflight_reason_codes": deepcopy(
                        row.get("reason_codes") or []
                    ),
                    "task_preflight_sha256": row.get(
                        "task_preflight_sha256"
                    ),
                }
            )
            continue
        task = task_catalog.get(task_identity)
        if not isinstance(task, Mapping):
            raise V610RegistryError(
                f"task catalog lacks {task_identity}"
            )
        actions = executable_actions(
            task_identity=task_identity,
            task=task,
            preflight_row=row,
        )
        try:
            _pair_options(task_identity, actions)
        except V610RegistryError:
            exclusions.append(
                {
                    "task_identity": task_identity,
                    "reason_code": (
                        "FEWER_THAN_TWO_DISTINCT_EXECUTABLE_ACTIONS"
                    ),
                    "eligible_reference_action_indices": [
                        int(value["reference_action_index"])
                        for value in actions
                    ],
                    "task_preflight_sha256": row[
                        "task_preflight_sha256"
                    ],
                }
            )
            continue
        actions_by_task[task_identity] = actions

    choices: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for phase, task_ids in PHASE_TASK_IDS.items():
        for task_identity in task_ids:
            if task_identity not in actions_by_task:
                continue
            domain, task_id = task_identity.split(":", 1)
            choice_set_id = f"v610:{phase}:{task_identity}:prefix:initial"
            prefix = {
                "prefix_slot_id": f"{choice_set_id}:prefix-slot",
                "source": "single_user_turn_before_first_assistant_tool_action",
                "reference_action_prefix_count": 0,
                "clean_future_available_to_causal_continuation": False,
                "shared_across_candidate_pairs": True,
                "materialization_seed": _seed(
                    REGISTRY_PROTOCOL,
                    phase,
                    task_identity,
                    "prefix",
                ),
            }
            snapshot = {
                "snapshot_slot_id": f"{choice_set_id}:snapshot-slot",
                "source": "fresh_pinned_tau2_task_initial_state",
                "agent_database_hash_required": True,
                "user_database_hash_requirement": (
                    "explicit_hash_or_null_unavailable"
                ),
                "shared_across_candidate_pairs": True,
            }
            prefix_hash = sha256(prefix)
            snapshot_hash = sha256(snapshot)
            options = _pair_options(
                task_identity,
                actions_by_task[task_identity],
            )
            pair_ids: list[str] = []
            for pair_index in range(PAIRS_PER_TASK):
                selected = options[pair_index % len(options)]
                candidate_pair_id = (
                    f"{choice_set_id}:candidate-pair:{pair_index + 1:02d}"
                )
                branches = [
                    _branch(
                        phase=phase,
                        task_identity=task_identity,
                        choice_set_id=choice_set_id,
                        candidate_pair_id=candidate_pair_id,
                        pair_index=pair_index,
                        branch_index=branch_index,
                        action=action,
                        prefix_sha256=prefix_hash,
                        snapshot_sha256=snapshot_hash,
                        reference_preflight_receipt_sha256=(
                            str(preflight_receipt["receipt_sha256"])
                        ),
                        reference_preflight_file_sha256=(
                            preflight_file_sha256
                        ),
                    )
                    for branch_index, action in enumerate(selected)
                ]
                pair = {
                    "candidate_pair_id": candidate_pair_id,
                    "choice_set_id": choice_set_id,
                    "phase": phase,
                    "partition": "arm_train",
                    "task_identity": task_identity,
                    "domain": domain,
                    "task_id": task_id,
                    "pair_index": pair_index,
                    "selection_unit": True,
                    "prefix_slot": prefix,
                    "snapshot_slot": snapshot,
                    "prefix_sha256": prefix_hash,
                    "environment_snapshot_sha256": snapshot_hash,
                    "choice_seed": _seed(
                        REGISTRY_PROTOCOL,
                        PHASE_SEED_SALTS[phase],
                        candidate_pair_id,
                    ),
                    "branches": branches,
                    "expected_canonical_corrective_families": [
                        branch["canonical_corrective_family"]
                        for branch in branches
                    ],
                    "reference_preflight_receipt_sha256": (
                        str(preflight_receipt["receipt_sha256"])
                    ),
                    "reference_preflight_file_sha256": (
                        preflight_file_sha256
                    ),
                    "reference_task_preflight_sha256": by_task[task_identity][
                        "task_preflight_sha256"
                    ],
                    "sanitized_reference_plan_sha256": by_task[task_identity][
                        "sanitized_successful_plan_sha256"
                    ],
                    "status": "REGISTERED_INJECTION_PREFLIGHT_REQUIRED",
                    "official_test_used": False,
                }
                pair["candidate_pair_sha256"] = sha256(pair)
                pairs.append(pair)
                pair_ids.append(candidate_pair_id)
            choice = {
                "choice_set_id": choice_set_id,
                "phase": phase,
                "task_identity": task_identity,
                "domain": domain,
                "task_id": task_id,
                "prefix_slot_id": prefix["prefix_slot_id"],
                "prefix_spec_sha256": prefix_hash,
                "snapshot_slot_id": snapshot["snapshot_slot_id"],
                "snapshot_spec_sha256": snapshot_hash,
                "candidate_pair_ids": pair_ids,
                "minimum_complete_pairs": PAIRS_PER_TASK,
                "status": "REGISTERED",
                "official_test_used": False,
            }
            choice["choice_set_sha256"] = sha256(choice)
            choices.append(choice)

    phase_registry: dict[str, Any] = {}
    for phase, task_ids_tuple in PHASE_TASK_IDS.items():
        task_ids = list(task_ids_tuple)
        eligible_tasks = [
            task for task in task_ids if task in actions_by_task
        ]
        phase_registry[phase] = {
            "task_ids": task_ids,
            "task_ids_sha256": sha256(task_ids),
            "task_count": len(task_ids),
            "eligible_task_ids": eligible_tasks,
            "eligible_task_ids_sha256": sha256(eligible_tasks),
            "eligible_task_count": len(eligible_tasks),
            "candidate_pair_count": len(eligible_tasks) * PAIRS_PER_TASK,
            "seed_salt": PHASE_SEED_SALTS[phase],
            "all_tasks_require_terminal_receipts": True,
        }

    forbidden_hash = sha256(
        {
            "validation": sorted(forbidden["validation"]),
            "sealed": sorted(forbidden["sealed"]),
        }
    )
    payload: dict[str, Any] = {
        "protocol": REGISTRY_PROTOCOL,
        "design_protocol": closure.PROTOCOL,
        "design_version": closure.DESIGN_VERSION,
        "registry_version": REGISTRY_VERSION,
        "registry_status": (
            "REFERENCE_PREFLIGHT_BOUND_INJECTION_PREFLIGHT_REQUIRED"
        ),
        "generation_authorized_before_runtime_injection_preflight": False,
        "selection_unit": "candidate_pair",
        "grouping_unit": "choice_set",
        "accounting_unit": "sibling_branch",
        "selection_uses_model_outcomes": False,
        "selection_inputs": [
            "frozen_arm_train_identity",
            "pinned_reference_action_structure",
            "model_free_reference_execution_preflight",
            "strict_identifier_allowlist",
            "protocol_seed",
        ],
        "source": {
            "tau2_commit": base.TAU2_COMMIT,
            "partition_sha256": base.PARTITION_SHA256,
            "split_file_sha256": split_file_sha256,
            "config_file_sha256": config_file_sha256,
            "task_file_sha256": dict(
                sorted((task_file_sha256 or {}).items())
            ),
            "source_commit": (
                str(source["commit"]) if source is not None else None
            ),
            "arm_train_task_ids_sha256": sha256(arm_train),
            "forbidden_identity_sets_sha256": forbidden_hash,
            "reference_preflight": {
                "protocol": reference.RECEIPT_PROTOCOL,
                "receipt_sha256": preflight_receipt["receipt_sha256"],
                "file_sha256": preflight_file_sha256,
                "ordered_task_ids_sha256": preflight_receipt[
                    "ordered_task_ids_sha256"
                ],
                "task_source_hashes_sha256": preflight_receipt[
                    "task_source_hashes_sha256"
                ],
            },
            "config_contract": config_contract,
        },
        "official_test_used": False,
        "official_test_sealed": True,
        "official_test_task_content_exported": False,
        "official_test_identity_overlap_count": 0,
        "arm_train_task_ids": arm_train,
        "arm_train_domain_counts": {
            domain: len(arm_by_domain[domain])
            for domain in ("retail", "airline")
        },
        # Retain the original structural digest for legacy verifier continuity.
        "structural_eligibility_sha256": (
            base.STRUCTURAL_ELIGIBILITY_SHA256
        ),
        "structural_registry_sha256": base.STRUCTURAL_ELIGIBILITY_SHA256,
        "reference_preflight_receipt_sha256": preflight_receipt[
            "receipt_sha256"
        ],
        "reference_preflight_file_sha256": preflight_file_sha256,
        "sanitized_plan_hashes_sha256": sha256(
            {
                task: row["sanitized_successful_plan_sha256"]
                for task, row in by_task.items()
            }
        ),
        "reference_preflight_exclusions": exclusions,
        "phase_registry": phase_registry,
        "gates": {
            "compatibility": {
                "minimum_tasks_with_three_accepted_pairs": 22,
                "minimum_accepted_pairs": 66,
            },
            "pilot": {
                "minimum_tasks_with_three_accepted_pairs": (
                    closure.PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": (
                    closure.PROSPECTIVE_MIN_ACCEPTED_PAIRS
                ),
                "minimum_domains": base.MIN_DOMAINS,
                "minimum_error_families": base.MIN_ERROR_FAMILIES,
            },
            "formal": {
                "minimum_tasks_with_three_accepted_pairs": (
                    base.FORMAL_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_accepted_pairs": (
                    base.FORMAL_MIN_ACCEPTED_PAIRS
                ),
                "screen_minimum_tasks": (
                    base.SCREEN_MIN_TASKS_WITH_THREE_PAIRS
                ),
                "minimum_domains": base.MIN_DOMAINS,
                "minimum_error_families": base.MIN_ERROR_FAMILIES,
            },
        },
        "choice_sets": choices,
        "candidate_pairs": pairs,
        "hashes": {
            "choice_sets_sha256": sha256(
                [row["choice_set_sha256"] for row in choices]
            ),
            "candidate_pairs_sha256": sha256(
                [row["candidate_pair_sha256"] for row in pairs]
            ),
            "reference_preflight_task_hashes_sha256": sha256(
                {
                    task: row["task_preflight_sha256"]
                    for task, row in by_task.items()
                }
            ),
        },
    }
    payload["registry_sha256"] = sha256(payload)
    return payload


def verify_registry(payload: Mapping[str, Any]) -> str:
    if (
        payload.get("protocol") != REGISTRY_PROTOCOL
        or payload.get("design_protocol") != closure.PROTOCOL
        or payload.get("official_test_used") is not False
        or payload.get("official_test_sealed") is not True
        or payload.get("official_test_task_content_exported") is not False
        or payload.get("official_test_identity_overlap_count") != 0
    ):
        raise V610RegistryError("registry identity/test seal drift")
    receipt_hash = payload.get("reference_preflight_receipt_sha256")
    file_hash = payload.get("reference_preflight_file_sha256")
    if (
        not isinstance(receipt_hash, str)
        or SHA256_RE.fullmatch(receipt_hash) is None
        or not isinstance(file_hash, str)
        or SHA256_RE.fullmatch(file_hash) is None
    ):
        raise V610RegistryError("registry preflight hashes are invalid")
    declared = payload.get("registry_sha256")
    if not isinstance(declared, str) or SHA256_RE.fullmatch(declared) is None:
        raise V610RegistryError("registry self-hash is absent")
    unhashed = deepcopy(dict(payload))
    unhashed.pop("registry_sha256", None)
    if sha256(unhashed) != declared:
        raise V610RegistryError("registry self-hash drift")
    phase_registry = payload.get("phase_registry")
    if not isinstance(phase_registry, Mapping):
        raise V610RegistryError("registry phases are absent")
    for phase, expected in PHASE_TASK_IDS.items():
        row = phase_registry.get(phase)
        if (
            not isinstance(row, Mapping)
            or row.get("task_ids") != list(expected)
            or row.get("task_ids_sha256") != sha256(list(expected))
        ):
            raise V610RegistryError(f"{phase}: registry task-list drift")
    pair_ids: set[str] = set()
    pairs = payload.get("candidate_pairs")
    if not isinstance(pairs, list):
        raise V610RegistryError("registry candidate pairs are absent")
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise V610RegistryError("registry pair is not an object")
        pair_id = pair.get("candidate_pair_id")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in pair_ids
        ):
            raise V610RegistryError("candidate pair id missing/duplicated")
        pair_ids.add(pair_id)
        copy = deepcopy(dict(pair))
        pair_hash = copy.pop("candidate_pair_sha256", None)
        if pair_hash != sha256(copy):
            raise V610RegistryError(f"{pair_id}: pair self-hash drift")
        task_hash = pair.get("reference_task_preflight_sha256")
        plan_hash = pair.get("sanitized_reference_plan_sha256")
        if (
            pair.get("reference_preflight_receipt_sha256") != receipt_hash
            or pair.get("reference_preflight_file_sha256") != file_hash
            or not isinstance(task_hash, str)
            or SHA256_RE.fullmatch(task_hash) is None
            or not isinstance(plan_hash, str)
            or SHA256_RE.fullmatch(plan_hash) is None
        ):
            raise V610RegistryError(
                f"{pair_id}: canonical preflight binding drift"
            )
        branches = pair.get("branches")
        if not isinstance(branches, list) or len(branches) != 2:
            raise V610RegistryError(f"{pair_id}: pair does not have two branches")
        for branch in branches:
            if not isinstance(branch, Mapping):
                raise V610RegistryError(f"{pair_id}: branch is malformed")
            branch_copy = deepcopy(dict(branch))
            branch_hash = branch_copy.pop("branch_slot_sha256", None)
            if branch_hash != sha256(branch_copy):
                raise V610RegistryError(
                    f"{pair_id}: branch self-hash drift"
                )
            binding = branch.get("reference_preflight_binding")
            corrective = branch.get("corrective_action_spec")
            if (
                not isinstance(binding, Mapping)
                or not isinstance(corrective, Mapping)
                or binding.get("reference_preflight_receipt_sha256")
                != receipt_hash
                or binding.get("reference_preflight_file_sha256")
                != file_hash
                or binding.get("reference_task_preflight_sha256")
                != task_hash
                or binding.get("sanitized_reference_plan_sha256")
                != plan_hash
                or binding.get("reference_action_index")
                != corrective.get("reference_action_index")
                or binding.get("reference_slot_id")
                != corrective.get("reference_slot_id")
            ):
                raise V610RegistryError(
                    f"{pair_id}: exact reference binding drift"
                )
    return declared


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(__file__).resolve().parents[1]
    tau2_root = args.tau2_root.resolve()
    split_path = args.split_manifest.resolve()
    config_path = args.config.resolve()
    preflight_path = args.reference_preflight.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite registry: {output}")
    source = source_provenance(source_root)
    if _git_output(tau2_root, "rev-parse", "HEAD") != base.TAU2_COMMIT:
        raise SystemExit("tau2 commit drift")
    if _git_output(
        tau2_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ):
        raise SystemExit("tau2 worktree contains tracked or untracked drift")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    config = legacy.load_config(config_path)
    receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
    catalog, task_hashes = legacy.load_task_catalog(tau2_root, split)
    registry = build_registry(
        split=split,
        config=config,
        task_catalog=catalog,
        preflight_receipt=receipt,
        preflight_file_sha256=sha256_file(preflight_path),
        split_file_sha256=sha256_file(split_path),
        config_file_sha256=sha256_file(config_path),
        task_file_sha256=task_hashes,
        source=source,
        strict=True,
    )
    verify_registry(registry)
    _write_json(output, registry)
    print(
        canonical(
            {
                "protocol": registry["protocol"],
                "registry_sha256": registry["registry_sha256"],
                "compatibility_tasks": registry["phase_registry"][
                    "compatibility"
                ]["eligible_task_count"],
                "pilot_tasks": registry["phase_registry"]["pilot"][
                    "eligible_task_count"
                ],
                "formal_tasks": registry["phase_registry"]["formal"][
                    "eligible_task_count"
                ],
                "candidate_pairs": len(registry["candidate_pairs"]),
                "status": "PASS",
            }
        )
    )


if __name__ == "__main__":
    main()
