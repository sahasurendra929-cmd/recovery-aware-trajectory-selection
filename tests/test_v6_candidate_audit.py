from __future__ import annotations

from copy import deepcopy

from scripts import audit_v6_candidates as audit
from scripts import measure_v6_candidate_tokens as measurement
from scripts import prepare_v6_candidate_registry as registry
from scripts import v6_selection_protocol as protocol


def frozen_config() -> dict:
    return {
        "protocol": protocol.PROTOCOL,
        "benchmark": {
            "commit": protocol.TAU2_COMMIT,
            "arm_train": {"tasks": 70},
            "official_test": {
                "used_during_generation_scoring_selection_or_model_selection": False,
                "sealed": True,
                "ids_or_content_exported_to_generation_workers": False,
            },
        },
    }


def task(task_id: int, domain: str) -> dict:
    if domain == "retail":
        actions = [
            {
                "action_id": f"{task_id}_0",
                "name": "get_order_details",
                "arguments": {"order_id": f"#W{task_id:04d}"},
            },
            {
                "action_id": f"{task_id}_1",
                "name": "cancel_pending_order",
                "arguments": {"order_id": f"#W{task_id:04d}"},
            },
            {
                "action_id": f"{task_id}_2",
                "name": "get_user_details",
                "arguments": {"user_id": f"user_{task_id:04d}"},
            },
        ]
    else:
        actions = [
            {
                "action_id": f"{task_id}_0",
                "name": "get_reservation_details",
                "arguments": {"reservation_id": f"R{task_id:05d}"},
            },
            {
                "action_id": f"{task_id}_1",
                "name": "update_reservation_flights",
                "arguments": {
                    "reservation_id": f"R{task_id:05d}",
                    "flight_number": f"FL{task_id:04d}",
                },
            },
        ]
    return {
        "id": task_id,
        "evaluation_criteria": {"actions": actions},
    }


def build_registry(task_count: int = 4) -> dict:
    airline_count = min(6, max(1, task_count // 8))
    retail_count = task_count - airline_count
    retail_ids = [str(1000 + index) for index in range(retail_count)]
    airline_ids = [str(2000 + index) for index in range(airline_count)]
    split = {
        "domains": {
            "retail": {
                "inner_train_ids": retail_ids,
                "validation_ids": ["9000"],
                "sealed_test_ids": ["9001"],
            },
            "airline": {
                "inner_train_ids": airline_ids,
                "validation_ids": ["9100"],
                "sealed_test_ids": ["9101"],
            },
        }
    }
    catalog = {
        **{
            f"retail:{task_id}": task(int(task_id), "retail")
            for task_id in retail_ids
        },
        **{
            f"airline:{task_id}": task(int(task_id), "airline")
            for task_id in airline_ids
        },
    }
    return registry.build_registry(
        split=split,
        config=frozen_config(),
        task_catalog=catalog,
        strict=False,
    )


def forced_cells(kappa_bucket: str) -> dict:
    if kappa_bucket == "HIGH":
        cross_12, cross_21 = 0.0, 0.0
    elif kappa_bucket == "LOW":
        cross_12, cross_21 = 1.0, 1.0
    else:
        cross_12, cross_21 = 0.0, 1.0
    seeds = [1, 2, 3]
    common = {
        "forced_first_only": True,
        "gold_suffix_visible": False,
        "continuation_policy_sha256": "policy" * 10,
        "continuation_seed_set_sha256": audit.sha256(seeds),
        "decoding_sha256": "decode" * 10,
        "rollout_budget": 60,
        "independent_replay_pass": True,
    }
    def cell(success: float, action: str) -> dict:
        return {
            **common,
            "task_success": success,
            "trials": [
                {
                    "seed": seed,
                    "task_success": success,
                    "forced_call": {
                        "name": action,
                        "arguments": {"id": action},
                    },
                    "independent_replay": {"pass": True},
                }
                for seed in seeds
            ],
        }
    return {
        "q_e1_a1": cell(1.0, "a1"),
        "q_e1_a2": cell(cross_12, "a2"),
        "q_e2_a1": cell(cross_21, "a1"),
        "q_e2_a2": cell(1.0, "a2"),
    }


def runtime_record(reg: dict, pair: dict, kappa_bucket: str) -> dict:
    system = {
        "role": "system",
        "content": "<instructions>help</instructions><policy>frozen</policy>",
    }
    prefix = [
        {
            "role": "user",
            "content": f"complete {pair['task_identity']}",
        }
    ]
    snapshot_state = {
        "task_identity": pair["task_identity"],
        "database": {"version": 1},
    }
    snapshot = {
        "state": snapshot_state,
        "state_sha256": audit.sha256(snapshot_state),
        "agent_db_hash": f"agent-before-{pair['task_identity']}",
        "user_db_hash": f"user-before-{pair['task_identity']}",
    }
    clean_messages = [
        *prefix,
        {
            "role": "assistant",
            "content": f"clean completion for {pair['candidate_pair_id']}",
        },
    ]
    clean = {
        "messages": clean_messages,
        "trace_sha256": audit.sha256(clean_messages),
        "official_task_success": 1.0,
        "final_state_valid": True,
        "final_agent_db_hash": f"agent-final-{pair['task_identity']}",
        "final_user_db_hash": f"user-final-{pair['task_identity']}",
    }
    actual_prefix_hash = audit.sha256(prefix)
    actual_snapshot_hash = audit.sha256(snapshot_state)
    tool_names = sorted(
        {
            registered_branch["tool_name"]
            for registered_branch in pair["branches"]
        }
        | {
            registered_branch["corrective_action_spec"][
                "forced_first_action_constructor"
            ]["tool_call"]["name"]
            for registered_branch in pair["branches"]
        }
    )
    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in tool_names
    ]
    branches = []
    for registered_branch in pair["branches"]:
        error_call = deepcopy(
            registered_branch["injection_spec"]["error_call"]
        )
        error_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [error_call],
        }
        error_result = {
            "role": "tool",
            "tool_call_id": error_call["id"],
            "content": {
                "error": (
                    f"not found for {registered_branch['branch_id']}"
                )
            },
            "error": True,
        }
        error_event = [error_assistant, error_result]
        reference_call = deepcopy(
            registered_branch["corrective_action_spec"][
                "forced_first_action_constructor"
            ]["tool_call"]
        )
        recovery_suffix = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [reference_call],
            },
            {
                "role": "tool",
                "tool_call_id": reference_call["id"],
                "content": {"status": "ok"},
                "error": False,
            },
            {
                "role": "assistant",
                "content": (
                    f"recovered with "
                    f"{registered_branch['canonical_corrective_family']}"
                ),
            },
        ]
        prompt = [*prefix, *error_event]
        full = [*prompt, *recovery_suffix]
        suffix_mask = [True, False, True]
        producer_full_mask = [False] * len(prompt) + suffix_mask
        training_prompt = [system, *prompt]
        training_full = [system, *full]
        full_mask = [False] * len(training_prompt) + suffix_mask
        compact_error = {
            "name": error_call["name"],
            "arguments": error_call["arguments"],
        }
        branches.append(
            {
                "branch_id": registered_branch["branch_id"],
                "registered_branch_slot_sha256": registered_branch[
                    "branch_slot_sha256"
                ],
                "shared_prefix_sha256": actual_prefix_hash,
                "environment_snapshot_sha256": actual_snapshot_hash,
                "error_event_messages": error_event,
                "error_event_sha256": audit.sha256(error_event),
                "recovery_prompt": prompt,
                "prompt": training_prompt,
                "training_prompt": training_prompt,
                "recovery_suffix": recovery_suffix,
                "full_trace": full,
                "training_full_trace": training_full,
                "full_trace_sha256": audit.sha256(full),
                "supervised_messages": recovery_suffix,
                "label_mask": producer_full_mask,
                "fresh_recovery_label_mask": suffix_mask,
                "full_assistant_label_mask": full_mask,
                "coverage": {
                    "domain": pair["domain"],
                    "failed_tool": registered_branch["tool_name"],
                    "error_family": (
                        f"{registered_branch['tool_name']}::"
                        f"{registered_branch['identifier_key']}"
                    ),
                    "corrective_action": registered_branch[
                        "corrective_family"
                    ],
                    "recovery_length_bin": "short",
                    "recovery_mode": "agent_initiated",
                },
                "tool_schemas": deepcopy(tool_schemas),
                "supervised_target_tokens": 50,
                "nonpadding_tokens": 100,
                "token_contract": {
                    "sequence_tokens": 100,
                    "supervised_tokens": 50,
                    "label_spans": [
                        {
                            "message_index": len(training_prompt),
                            "token_start": 50,
                            "token_end": 100,
                        }
                    ],
                },
                "token_measurement_status": "MEASURED_FROZEN_STUDENT",
                "database_hashes": {
                    "agent_before_error": snapshot["agent_db_hash"],
                    "agent_after_error": snapshot["agent_db_hash"],
                    "user_before_error": snapshot["user_db_hash"],
                    "user_after_error": snapshot["user_db_hash"],
                },
                "tool_execution_evidence": {
                    "executed_in_pinned_environment": True,
                    "tau2_commit": protocol.TAU2_COMMIT,
                    "tool_call_sha256": audit.sha256(compact_error),
                    "tool_result_sha256": audit.sha256(error_result),
                },
                "label_audit": {
                    "future_message_overlap_count": 0,
                    "clean_future_visible": False,
                    "failed_positive_label_count": 0,
                    "error_result_positive_label_count": 0,
                },
                "matched_replay": {
                    "official_task_success": 1.0,
                    "independent_replay_pass": True,
                    "final_state_valid": True,
                    "final_agent_db_hash": clean["final_agent_db_hash"],
                    "final_user_db_hash": clean["final_user_db_hash"],
                    "independent_replay_audit_sha256": (
                        audit.sha256(
                            [
                                pair["candidate_pair_id"],
                                registered_branch["branch_id"],
                            ]
                        )
                    ),
                },
                "producer_evidence": {
                    "matched_recovery": {
                        "attempts": [
                            {
                                "attempt": 1,
                                "seed": 7,
                                "first_action": reference_call,
                                "first_action_matched": True,
                            }
                        ]
                    }
                },
                "official_test_used": False,
            }
        )
    measurement_provenance = {
        "protocol": measurement.PROTOCOL,
        "model_name": audit.FROZEN_STUDENT_MODEL,
        "model_revision": audit.FROZEN_STUDENT_REVISION,
        "tokenizer_name": audit.FROZEN_STUDENT_MODEL,
        "tokenizer_revision": audit.FROZEN_STUDENT_REVISION,
        "load_in_4bit": True,
        "official_test_used": False,
        "validation_used": False,
        "measurement_uses_train_pool_only": True,
        "failed_calls_supervised": False,
        "full_fresh_assistant_suffix_supervised": True,
        "frozen_checkpoint_identity_sha256": "a" * 64,
        "raw_candidate_record_sha256": audit.sha256(
            ["raw-candidate", pair["candidate_pair_id"]]
        ),
        "source_pool_sha256": audit.sha256(
            ["source-pool", reg["registry_sha256"]]
        ),
        "tool_schemas_sha256": measurement.canonical_sha256(
            tool_schemas
        ),
        "training_system_message_sha256": measurement.canonical_sha256(
            system
        ),
    }
    measurement_provenance["measurement_contract_sha256"] = (
        measurement.canonical_sha256(
            {
                key: measurement_provenance.get(key)
                for key in (
                    "protocol",
                    "model_name",
                    "model_revision",
                    "tokenizer_name",
                    "tokenizer_revision",
                    "frozen_checkpoint_identity_sha256",
                    "tool_schemas_sha256",
                    "failed_calls_supervised",
                    "full_fresh_assistant_suffix_supervised",
                )
            }
        )
    )
    return {
        "registry_sha256": reg["registry_sha256"],
        "registered_candidate_pair_sha256": pair[
            "candidate_pair_sha256"
        ],
        "candidate_pair_id": pair["candidate_pair_id"],
        "choice_set_id": pair["choice_set_id"],
        "phase": pair["phase"],
        "task_identity": pair["task_identity"],
        "shared_prefix": prefix,
        "shared_prefix_sha256": actual_prefix_hash,
        "training_system_message": system,
        "training_system_message_sha256": audit.sha256(system),
        "environment_snapshot": snapshot,
        "clean_trace": clean,
        "clean_view": {
            "messages": [system, *clean_messages],
            "raw_messages": clean_messages,
            "label_mask": [False, False, True],
            "c_sup": 10,
            "c_nonpad": 20,
        },
        "tool_schemas": tool_schemas,
        "branches": branches,
        "forced_first_cells": forced_cells(kappa_bucket),
        "token_measurement_provenance": measurement_provenance,
        "token_accounting": {
            "supervised_target_tokens": 100,
            "nonpadding_tokens": 200,
        },
        "official_test_used": False,
    }


def test_complete_pair_audit_accepts_and_preserves_low_kappa():
    reg = build_registry(4)
    pair = next(
        row for row in reg["candidate_pairs"] if row["phase"] == "formal"
    )
    record = runtime_record(reg, pair, "LOW")
    audited = audit.audit_candidate_pair(
        record,
        pair,
        registry_sha256=reg["registry_sha256"],
    )
    assert audited["audit_status"] == "ACCEPTED"
    assert audited["kappa"] == 0.0
    assert audited["kappa_bucket"] == "LOW"
    assert all(
        audited["audit_checks"]["frozen_protocol"].values()
    )

    report, accepted, freeze = audit.audit_candidates(
        registry=reg,
        candidates=[record],
        phase="formal",
    )
    assert report["accepted_candidate_pairs"] == 1
    assert len(accepted) == 1
    assert freeze["status"] == "NOT_FROZEN_FAIL_CLOSED"
    assert freeze["low_kappa_pairs_preserved"] == [
        pair["candidate_pair_id"]
    ]


def test_future_leakage_and_failed_positive_labels_are_rejected():
    reg = build_registry(4)
    pair = next(
        row for row in reg["candidate_pairs"] if row["phase"] == "formal"
    )
    contaminated = runtime_record(reg, pair, "HIGH")
    branch = contaminated["branches"][0]
    clean_future = contaminated["clean_trace"]["messages"][-1]
    branch["recovery_prompt"].append(clean_future)
    branch["full_trace"] = [
        *branch["recovery_prompt"],
        *branch["recovery_suffix"],
    ]
    branch["full_trace_sha256"] = audit.sha256(branch["full_trace"])
    branch["supervised_messages"] = [
        *branch["error_event_messages"],
        *branch["recovery_suffix"],
    ]
    branch["label_audit"]["future_message_overlap_count"] = 1
    branch["label_audit"]["clean_future_visible"] = True
    branch["label_audit"]["failed_positive_label_count"] = 1
    branch["label_audit"]["error_result_positive_label_count"] = 1
    result = audit.audit_candidate_pair(
        contaminated,
        pair,
        registry_sha256=reg["registry_sha256"],
    )
    assert result["audit_status"] == "REJECTED"
    checks = result["audit_checks"]["branches"][branch["branch_id"]]
    assert checks["clean_future_leakage_zero"] is False
    assert checks["failed_positive_labels_zero"] is False


def test_token_cost_tampering_is_rejected_at_every_redundant_layer():
    reg = build_registry(4)
    pair = next(
        row for row in reg["candidate_pairs"] if row["phase"] == "formal"
    )

    top_tampered = runtime_record(reg, pair, "HIGH")
    top_tampered["token_accounting"]["supervised_target_tokens"] += 1
    top_result = audit.audit_candidate_pair(
        top_tampered,
        pair,
        registry_sha256=reg["registry_sha256"],
    )
    assert top_result["audit_status"] == "REJECTED"
    assert (
        top_result["audit_checks"]["top"][
            "token_accounting_recomputed_from_both_branches"
        ]
        is False
    )

    declared_tampered = runtime_record(reg, pair, "HIGH")
    declared_branch = declared_tampered["branches"][0]
    declared_branch["supervised_target_tokens"] += 1
    declared_result = audit.audit_candidate_pair(
        declared_tampered,
        pair,
        registry_sha256=reg["registry_sha256"],
    )
    assert declared_result["audit_status"] == "REJECTED"
    assert (
        declared_result["audit_checks"]["branches"][
            declared_branch["branch_id"]
        ]["frozen_student_token_contract_matches_declared_cost"]
        is False
    )
    assert (
        declared_result["audit_checks"]["top"][
            "token_accounting_recomputed_from_both_branches"
        ]
        is False
    )

    contract_tampered = runtime_record(reg, pair, "HIGH")
    contract_branch = contract_tampered["branches"][0]
    contract_branch["token_contract"]["supervised_tokens"] += 1
    contract_result = audit.audit_candidate_pair(
        contract_tampered,
        pair,
        registry_sha256=reg["registry_sha256"],
    )
    assert contract_result["audit_status"] == "REJECTED"
    assert (
        contract_result["audit_checks"]["branches"][
            contract_branch["branch_id"]
        ]["frozen_student_token_contract_matches_declared_cost"]
        is False
    )
    assert (
        contract_result["audit_checks"]["top"][
            "token_accounting_recomputed_from_both_branches"
        ]
        is False
    )


def test_measurement_contract_sha256_is_recomputed_from_provenance():
    reg = build_registry(4)
    pair = next(
        row for row in reg["candidate_pairs"] if row["phase"] == "formal"
    )
    tampered = runtime_record(reg, pair, "HIGH")
    tampered["token_measurement_provenance"][
        "measurement_contract_sha256"
    ] = "f" * 64
    result = audit.audit_candidate_pair(
        tampered,
        pair,
        registry_sha256=reg["registry_sha256"],
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "measurement_contract_sha256_recomputed"
        ]
        is False
    )


def test_formal_gate_passes_48_tasks_and_keeps_all_kappa_strata():
    reg = build_registry(48)
    formal_pairs = [
        row for row in reg["candidate_pairs"] if row["phase"] == "formal"
    ]
    assert len(formal_pairs) == 144
    buckets = ("HIGH", "LOW", "MIDDLE")
    records = [
        runtime_record(reg, pair, buckets[pair["pair_index"]])
        for pair in formal_pairs
    ]
    report, accepted, freeze = audit.audit_candidates(
        registry=reg,
        candidates=records,
        phase="formal",
    )
    assert len(accepted) == 144
    assert report["gate"]["status"] == "FORMAL_SELECTION_AUTHORIZED"
    assert report["gate"]["pool"]["tasks_with_at_least_three_pairs"] == 48
    assert report["kappa_bucket_counts"] == {
        "HIGH": 48,
        "LOW": 48,
        "MIDDLE": 48,
    }
    assert freeze["status"] == "FROZEN"
    assert freeze["selection_authorized"] is True
    assert len(freeze["low_kappa_pairs_preserved"]) == 48
