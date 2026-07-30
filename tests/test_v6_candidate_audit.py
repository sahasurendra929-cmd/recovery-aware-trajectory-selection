from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import audit_v6_candidates as audit
from scripts import materialize_v6_sft as materialize
from scripts import measure_v6_candidate_tokens as measurement
from scripts import prepare_v6_candidate_registry as registry
from scripts import run_v6_candidate_generation as generation
from scripts import v6_10_selection_protocol as closure
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


def v610_registry_record(
    kappa_bucket: str = "HIGH",
    *,
    phase: str = "formal",
    continuation_seeds: tuple[int, ...] | None = None,
) -> tuple[dict, dict, dict]:
    """Upgrade one legacy fixture to the fail-closed V6.10 evidence schema."""
    if phase not in {"compatibility", "pilot", "formal"}:
        raise ValueError(phase)
    expected_seeds = (
        audit.V6_10_COMPATIBILITY_SEEDS
        if phase == "compatibility"
        else audit.V6_10_CONTINUATION_SEEDS
    )
    seeds = continuation_seeds or expected_seeds
    reg = build_registry(4)
    reg["protocol"] = audit.V6_10_REGISTRY_PROTOCOL
    reg["design_protocol"] = audit.V6_10_PIPELINE_CLOSURE_PROTOCOL
    phase_tasks = {
        "compatibility": list(closure.COMPATIBILITY_TASK_IDS),
        "pilot": list(closure.PROSPECTIVE_PILOT_TASK_IDS),
        "formal": list(closure.FORMAL_TASK_IDS),
    }
    reg["phase_registry"] = {
        name: {
            "task_ids": task_ids,
            "task_ids_sha256": audit.sha256(task_ids),
            "task_count": len(task_ids),
            "all_tasks_require_terminal_receipts": True,
        }
        for name, task_ids in phase_tasks.items()
    }
    reg["gates"] = {
        "compatibility": {
            "minimum_tasks_with_three_accepted_pairs": 22,
            "minimum_accepted_pairs": 66,
        },
        "pilot": {
            "minimum_tasks_with_three_accepted_pairs": 16,
            "minimum_accepted_pairs": 48,
            "minimum_domains": 2,
            "minimum_error_families": 3,
        },
        "formal": {
            "minimum_tasks_with_three_accepted_pairs": 48,
            "minimum_accepted_pairs": 144,
            "screen_minimum_tasks": 40,
            "minimum_domains": 2,
            "minimum_error_families": 3,
        },
    }
    receipt_sha256 = "b" * 64
    receipt_file_sha256 = "a" * 64
    reg["reference_preflight_receipt_sha256"] = receipt_sha256
    reg["reference_preflight_file_sha256"] = receipt_file_sha256
    reg["reference_preflight_ordered_task_ids_sha256"] = audit.sha256(
        sorted(
            {
                row["task_identity"]
                for row in reg["candidate_pairs"]
                if row["phase"] == "formal"
            }
        )
    )
    pair = next(
        row for row in reg["candidate_pairs"] if row["phase"] == "formal"
    )
    pair["phase"] = phase
    for registered_branch in pair["branches"]:
        registered_branch["phase"] = phase
    choice = next(
        row
        for row in reg["choice_sets"]
        if row["choice_set_id"] == pair["choice_set_id"]
    )
    choice["phase"] = phase
    choice["choice_set_sha256"] = audit.sha256(
        {
            key: value
            for key, value in choice.items()
            if key != "choice_set_sha256"
        }
    )
    indices = sorted(
        {
            branch["corrective_action_spec"]["reference_action_index"]
            for branch in pair["branches"]
        }
    )
    plan_hash = audit.sha256(indices)
    task_hash = audit.sha256(
        ["task-preflight", pair["task_identity"], indices]
    )
    slots_by_index = {}
    for registered_branch in pair["branches"]:
        index = registered_branch["corrective_action_spec"][
            "reference_action_index"
        ]
        call = registered_branch["corrective_action_spec"][
            "forced_first_action_constructor"
        ]["tool_call"]
        call_semantics = {
            "requestor": call.get("requestor", "assistant"),
            "name": call["name"],
            "arguments": deepcopy(call["arguments"]),
        }
        slot_id = f"{pair['task_identity']}:reference:{index:03d}"
        slot = {
            "reference_action_index": index,
            "reference_slot_id": slot_id,
            "reference_slot_sha256": audit.sha256(
                [slot_id, call_semantics]
            ),
            "call_semantics_sha256": audit.sha256(call_semantics),
        }
        slots_by_index[str(index)] = slot
        registered_branch["reference_preflight_binding"] = {
            "reference_preflight_receipt_sha256": receipt_sha256,
            "reference_preflight_file_sha256": receipt_file_sha256,
            "reference_task_preflight_sha256": task_hash,
            "sanitized_reference_plan_sha256": plan_hash,
            "reference_action_index": index,
            "reference_slot_id": slot["reference_slot_id"],
            "reference_slot_sha256": slot["reference_slot_sha256"],
            "call_semantics_sha256": slot["call_semantics_sha256"],
        }
        registered_branch["branch_slot_sha256"] = audit.sha256(
            {
                key: value
                for key, value in registered_branch.items()
                if key != "branch_slot_sha256"
            }
        )
    task_row = {
        "task_identity": pair["task_identity"],
        "reference_preflight_receipt_sha256": receipt_sha256,
        "reference_preflight_file_sha256": receipt_file_sha256,
        "task_preflight_sha256": task_hash,
        "reference_slots_by_index": slots_by_index,
        "expected_error_indices": [],
        "expected_error_set_sha256": audit.sha256([]),
        "sanitized_successful_reference_indices": indices,
        "sanitized_successful_plan_sha256": plan_hash,
        "eligible_forced_first_reference_indices": indices,
        "eligible_forced_first_reference_indices_sha256": audit.sha256(
            indices
        ),
        "raw_reference_environment_reward": 1.0,
        "sanitized_environment_reward": 1.0,
        "dynamic_full_official_reward_required": True,
        "official_test_used": False,
    }
    pair["design_protocol"] = audit.V6_10_PIPELINE_CLOSURE_PROTOCOL
    pair["reference_preflight_receipt_sha256"] = receipt_sha256
    pair["reference_preflight_file_sha256"] = receipt_file_sha256
    pair["reference_task_preflight_sha256"] = task_hash
    pair["sanitized_reference_plan_sha256"] = plan_hash
    pair["candidate_pair_sha256"] = audit.sha256(
        {
            key: value
            for key, value in pair.items()
            if key != "candidate_pair_sha256"
        }
    )
    reg["registry_sha256"] = audit.sha256(
        {
            key: value
            for key, value in reg.items()
            if key != "registry_sha256"
        }
    )
    record = runtime_record(reg, pair, kappa_bucket)
    record["reference_plan_binding"] = deepcopy(task_row)
    record["reference_preflight_receipt_sha256"] = receipt_sha256
    record["reference_preflight_file_sha256"] = receipt_file_sha256
    record["reference_task_preflight_sha256"] = task_hash
    record["sanitized_reference_plan_sha256"] = plan_hash
    record["clean_trace"]["sanitized_reference_plan_sha256"] = plan_hash
    record["clean_view"]["sanitized_reference_plan_sha256"] = plan_hash
    record["clean_view"]["official_test_used"] = False
    source_commit = "c" * 40
    generation_contract = {
        "design_protocol": audit.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "teacher_model": "teacher/model",
        "teacher_revision": "teacher-revision",
        "judge_model": "judge/model",
        "judge_revision": "judge-revision",
        "matched_positive_continuation_mode": (
            "deterministic_reference_completion"
        ),
        "causal_cell_continuation_mode": "fresh_teacher",
        "first_action_measurement_mode": "teacher_unforced",
        "teacher_unforced_first_action_generated": True,
        "teacher_unforced_probe_contract": {
            "raw_teacher_response": True,
            "assistant_generations_per_error_context": 1,
            "normalization_applied": False,
            "user_continuation_generated": False,
            "judge_invoked": False,
            "task_success_status": (
                "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE"
            ),
            "retry_count": 0,
        },
        "causal_cell_gold_suffix_visible": False,
        "teacher_unforced_first_action_generated": True,
        "official_test_used": False,
        "runtime_provenance": {
            "expected_source_commit": source_commit,
            "observed_source_commit": source_commit,
            "official_test_used": False,
            "reference_preflight": {
                "protocol": audit.V6_10_REFERENCE_PREFLIGHT_PROTOCOL,
                "design_protocol": audit.V6_10_PIPELINE_CLOSURE_PROTOCOL,
                "receipt_sha256": receipt_sha256,
                "file_sha256": receipt_file_sha256,
                "status": "PASS",
                "tau2_commit": protocol.TAU2_COMMIT,
                "source_commit": source_commit,
                "official_test_used": False,
            },
        },
    }
    record["generation_contract"] = generation_contract
    record["generation_contract_sha256"] = audit.sha256(
        generation_contract
    )
    record["continuation_role_separation"] = {
        "matched_positive_continuation_mode": (
            "deterministic_reference_completion"
        ),
        "causal_cell_continuation_mode": "fresh_teacher",
        "first_action_measurement_mode": "teacher_unforced",
        "causal_cells_gold_free": True,
    }
    record["clean_trace"]["official_reward_info"] = {"reward": 1.0}
    action_slot = {
        "q_e1_a1": 0,
        "q_e1_a2": 1,
        "q_e2_a1": 0,
        "q_e2_a2": 1,
    }
    for name, cell in record["forced_first_cells"].items():
        registered_branch = pair["branches"][action_slot[name]]
        reference_call = deepcopy(
            registered_branch["corrective_action_spec"][
                "forced_first_action_constructor"
            ]["tool_call"]
        )
        cell["continuation_mode"] = "fresh_teacher"
        cell["fresh_recovery_generated"] = True
        cell["gold_access_audit"] = {
            "gold_reference_actions_visible_to_continuation": False,
            "evaluation_criteria_visible_to_continuation": False,
        }
        cell["forced_reference_action_index"] = registered_branch[
            "corrective_action_spec"
        ]["reference_action_index"]
        binding = registered_branch["reference_preflight_binding"]
        cell["forced_reference_slot_id"] = binding["reference_slot_id"]
        cell["reference_task_preflight_sha256"] = task_hash
        cell["sanitized_reference_plan_sha256"] = plan_hash
        cell["reference_preflight_receipt_sha256"] = receipt_sha256
        cell["reference_preflight_file_sha256"] = receipt_file_sha256
        template_trial = deepcopy(cell["trials"][0])
        cell["trials"] = [
            deepcopy(template_trial) for _ in seeds
        ]
        cell["continuation_seed_set_sha256"] = audit.sha256(list(seeds))
        for trial, seed in zip(cell["trials"], seeds):
            trial["seed"] = seed
            trial["forced_call"] = reference_call
            trial["official_reward_info"] = {
                "reward": trial["task_success"]
            }
    for branch, registered_branch in zip(
        record["branches"], pair["branches"]
    ):
        reference_call = deepcopy(
            registered_branch["corrective_action_spec"][
                "forced_first_action_constructor"
            ]["tool_call"]
        )
        branch["sanitized_reference_plan_sha256"] = plan_hash
        branch["reference_preflight_binding"] = deepcopy(
            registered_branch["reference_preflight_binding"]
        )
        branch["reference_action_index"] = registered_branch[
            "corrective_action_spec"
        ]["reference_action_index"]
        branch["reference_slot_id"] = registered_branch[
            "reference_preflight_binding"
        ]["reference_slot_id"]
        branch["producer_evidence"]["matched_recovery"].update(
            {
                "continuation_mode": (
                    "deterministic_reference_completion"
                ),
                "fresh_recovery_generated": False,
                "gold_suffix_used": True,
                "reference_plan_binding": deepcopy(task_row),
                "official_task_success": 1.0,
            }
        )
        branch["producer_evidence"]["matched_recovery"]["attempts"][0].update(
            {
                "official_reward": 1.0,
                "official_reward_info": {"reward": 1.0},
            }
        )
        seed = registered_branch["recovery_seed"]
        semantics = {
            "requestor": reference_call.get("requestor", "assistant"),
            "name": reference_call["name"],
            "arguments": deepcopy(reference_call["arguments"]),
        }
        raw_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [deepcopy(reference_call)],
        }
        branch["teacher_unforced_first_action_measurement"] = {
            "mode": "teacher_unforced_raw_first_response",
            "forced_first": False,
            "gold_suffix_visible": False,
            "fresh_recovery_generated": False,
            "raw_first_response_generated": True,
            "teacher_model": generation_contract["teacher_model"],
            "teacher_revision": generation_contract["teacher_revision"],
            "measurement_seed": seed,
            "measurement_seed_sha256": audit.sha256(seed),
            "trial_count": 1,
            "generation_count": 1,
            "matched_registered_corrective_accuracy": 1.0,
            "normalization_applied": False,
            "normalization_forbidden": True,
            "task_success_status": (
                "DEFERRED_NOT_PART_OF_FIRST_ACTION_PROBE"
            ),
            "trials": [
                {
                    "seed": seed,
                    "generation_count": 1,
                    "task_success": None,
                    "task_success_status": (
                        "NOT_MEASURED_FIRST_RESPONSE_ONLY"
                    ),
                    "official_reward_info": None,
                    "first_action_observed": True,
                    "first_action": reference_call,
                    "first_action_matched_registered_corrective": True,
                    "first_action_semantics_sha256": audit.sha256(
                        (
                            semantics["name"],
                            audit.canonical(semantics["arguments"]),
                            semantics["requestor"],
                        )
                    ),
                    "first_response_status": "VALID_SINGLE_TOOL_CALL",
                    "malformed_reason": None,
                    "raw_response": raw_response,
                    "raw_response_sha256": audit.sha256(raw_response),
                    "normalization_applied": False,
                    "normalization_forbidden": True,
                    "user_continuation_generated": False,
                    "judge_invoked": False,
                    "full_rollout_generated": False,
                    "retry_count": 0,
                    "wall_seconds": 0.1,
                }
            ],
            "official_test_used": False,
        }
    return reg, pair, record


def v610_gate_registry() -> dict:
    task_ids = {
        "compatibility": list(closure.COMPATIBILITY_TASK_IDS),
        "pilot": list(closure.PROSPECTIVE_PILOT_TASK_IDS),
        "formal": list(closure.FORMAL_TASK_IDS),
    }
    return {
        "design_protocol": audit.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "phase_registry": {
            phase: {
                "task_ids": values,
                "task_ids_sha256": audit.sha256(values),
                "task_count": len(values),
                "all_tasks_require_terminal_receipts": True,
            }
            for phase, values in task_ids.items()
        },
        "gates": {
            "compatibility": {
                "minimum_tasks_with_three_accepted_pairs": 22,
                "minimum_accepted_pairs": 66,
            },
            "pilot": {
                "minimum_tasks_with_three_accepted_pairs": 16,
                "minimum_accepted_pairs": 48,
                "minimum_domains": 2,
                "minimum_error_families": 3,
            },
            "formal": {
                "minimum_tasks_with_three_accepted_pairs": 48,
                "minimum_accepted_pairs": 144,
                "screen_minimum_tasks": 40,
                "minimum_domains": 2,
                "minimum_error_families": 3,
            },
        },
    }


def v610_gate_choice_sets(
    phase: str,
    task_ids: tuple[str, ...],
) -> list[dict]:
    base_registry = build_registry(4)
    base_by_domain: dict[str, list[dict]] = {}
    for domain in ("retail", "airline"):
        base_pair = next(
            row
            for row in base_registry["candidate_pairs"]
            if row["phase"] == "formal" and row["domain"] == domain
        )
        base_by_domain[domain] = [
            audit.audit_candidate_pair(
                runtime_record(base_registry, base_pair, bucket),
                base_pair,
                registry_sha256=base_registry["registry_sha256"],
            )
            for bucket in ("HIGH", "LOW", "MIDDLE")
        ]
        assert all(
            row["audit_status"] == "ACCEPTED"
            for row in base_by_domain[domain]
        )
    result = []
    for task_identity in task_ids:
        domain, task_id = task_identity.split(":", 1)
        choice_id = f"{phase}:{task_identity}:choice"
        pairs = []
        for index, base_pair in enumerate(base_by_domain[domain]):
            pair = deepcopy(base_pair)
            pair_id = f"{phase}:{task_identity}:pair:{index}"
            pair["candidate_pair_id"] = pair_id
            pair["choice_set_id"] = choice_id
            pair["phase"] = phase
            pair["task_identity"] = task_identity
            pair["task_id"] = task_id
            pair["domain"] = domain
            for branch_index, branch in enumerate(pair["branches"]):
                branch["candidate_pair_id"] = pair_id
                branch["choice_set_id"] = choice_id
                branch["phase"] = phase
                branch["task_identity"] = task_identity
                branch["branch_id"] = (
                    f"{pair_id}:branch:{branch_index}"
                )
            pairs.append(pair)
        choice = {
            "choice_set_id": choice_id,
            "task_identity": task_identity,
            "domain": domain,
            "prefix_sha256": pairs[0]["prefix_sha256"],
            "environment_snapshot_sha256": pairs[0][
                "environment_snapshot_sha256"
            ],
            "candidate_pairs": pairs,
        }
        choice["audit_checks"] = protocol.audit_choice_set(choice)
        assert all(choice["audit_checks"].values())
        result.append(choice)
    return result


def v610_generation_closure_fixture(
    phase: str = "compatibility",
) -> tuple[dict, list[dict], dict]:
    expected_tasks = list(audit._v610_phase_spec(phase)["task_ids"])
    passing_task = expected_tasks[0]
    pair_id = f"{phase}:{passing_task}:pair:0"
    registry_sha256 = "1" * 64
    registry_payload = {
        "registry_sha256": registry_sha256,
        "candidate_pairs": [
            {
                "candidate_pair_id": pair_id,
                "task_identity": passing_task,
                "phase": phase,
            }
        ],
    }
    candidates = [
        {
            "candidate_pair_id": pair_id,
            "task_identity": passing_task,
            "phase": phase,
        }
    ]
    attempt_id = "v610-terminal-fixture"
    semantic_contract = {
        "design_protocol": audit.V6_10_PIPELINE_CLOSURE_PROTOCOL,
        "runtime_provenance": {"attempt_id": attempt_id},
    }
    semantic_hash = audit.sha256(semantic_contract)
    run_hash = "2" * 64
    receipts = []
    ledger = []
    for task_identity in expected_tasks:
        common = {
            "protocol": audit.GENERATION_PROTOCOL,
            "status": "PASS",
            "execution_status": "PASS",
            "task_identity": task_identity,
            "registry_sha256": registry_sha256,
            "run_contract_sha256": run_hash,
            "semantic_generation_contract": deepcopy(
                semantic_contract
            ),
            "semantic_generation_contract_sha256": semantic_hash,
            "runtime_provenance": {"attempt_id": attempt_id},
            "attempt_id": attempt_id,
            "official_test_used": False,
        }
        if task_identity == passing_task:
            raw_pair = {
                "protocol": audit.GENERATION_PROTOCOL,
                "candidate_pair_id": pair_id,
                "task_identity": task_identity,
                "registry_sha256": registry_sha256,
                "generation_contract": deepcopy(semantic_contract),
                "generation_contract_sha256": semantic_hash,
                "official_test_used": False,
            }
            raw_pair["generated_candidate_pair_sha256"] = (
                audit.sha256(raw_pair)
            )
            candidates[0].update(
                {
                    "registry_sha256": registry_sha256,
                    "generated_candidate_pair_sha256": raw_pair[
                        "generated_candidate_pair_sha256"
                    ],
                    "generation_contract": deepcopy(
                        semantic_contract
                    ),
                    "generation_contract_sha256": semantic_hash,
                    "token_measurement_provenance": {
                        "raw_candidate_record_sha256": audit.sha256(
                            raw_pair
                        )
                    },
                }
            )
            receipt = {
                **common,
                "scientific_outcome": "ACCEPTED",
                "candidate_pairs": [raw_pair],
                "candidate_pair_count": 1,
            }
        else:
            receipt = {
                **common,
                "scientific_outcome": "REJECTED",
                "reason_code": "REFERENCE_PREFLIGHT_TASK_REJECTED",
                "error_type": "V6GenerationError",
                "error": "no verified PASS reference plan",
                "expected_candidate_pair_ids": [],
                "candidate_pairs": [],
                "candidate_pair_count": 0,
                "training_started": False,
            }
        receipt["task_receipt_sha256"] = audit.sha256(receipt)
        receipts.append(receipt)
        if receipt["scientific_outcome"] == "REJECTED":
            ledger.append(
                {
                    "task_identity": task_identity,
                    "reason_code": receipt["reason_code"],
                    "error_type": receipt["error_type"],
                    "error": receipt["error"],
                    "task_receipt_sha256": receipt[
                        "task_receipt_sha256"
                    ],
                    "attempt_id": attempt_id,
                    "official_test_used": False,
                }
            )
    receipt_hashes = {
        row["task_identity"]: row["task_receipt_sha256"]
        for row in receipts
    }
    generation_receipt = {
        "protocol": audit.GENERATION_PROTOCOL,
        "status": "PASS",
        "execution_status": "PASS",
        "phase": phase,
        "expected_tasks": len(expected_tasks),
        "completed_tasks": len(expected_tasks),
        "missing_tasks": [],
        "passing_tasks": 1,
        "rejected_tasks": len(expected_tasks) - 1,
        "accepted_candidate_pairs": 1,
        "expected_candidate_pairs": 1,
        "task_receipt_sha256_by_task": dict(
            sorted(receipt_hashes.items())
        ),
        "task_receipts_sha256": audit.sha256(
            dict(sorted(receipt_hashes.items()))
        ),
        "registry_sha256": registry_sha256,
        "run_contract_sha256": run_hash,
        "semantic_generation_contract_sha256": semantic_hash,
        "attempt_id": attempt_id,
        "training_started": False,
        "official_test_used": False,
    }
    closure_bundle = {
        "generation_receipt": generation_receipt,
        "generation_receipt_sha256": audit.sha256(
            generation_receipt
        ),
        "failure_ledger": ledger,
        "failure_ledger_sha256": audit.sha256(ledger),
        "task_receipts": receipts,
        "task_receipts_sha256": audit.sha256(
            dict(sorted(receipt_hashes.items()))
        ),
    }
    return registry_payload, candidates, closure_bundle


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


def test_v610_complete_evidence_is_accepted_and_uses_unforced_probe():
    reg, pair, record = v610_registry_record("HIGH")
    branch = record["branches"][0]
    sibling_call = deepcopy(
        pair["branches"][1]["corrective_action_spec"][
            "forced_first_action_constructor"
        ]["tool_call"]
    )
    trial = branch["teacher_unforced_first_action_measurement"]["trials"][0]
    trial["first_action"] = sibling_call
    trial["raw_response"]["tool_calls"] = [deepcopy(sibling_call)]
    trial["raw_response_sha256"] = audit.sha256(trial["raw_response"])
    trial["first_action_matched_registered_corrective"] = False
    trial["first_action_semantics_sha256"] = audit.sha256(
        (
            sibling_call["name"],
            audit.canonical(sibling_call["arguments"]),
            sibling_call.get("requestor", "assistant"),
        )
    )
    branch["teacher_unforced_first_action_measurement"][
        "matched_registered_corrective_accuracy"
    ] = 0.0
    # The matched-positive constructor remains forced and reports a match.
    assert branch["producer_evidence"]["matched_recovery"]["attempts"][0][
        "first_action_matched"
    ] is True

    result = audit.audit_candidate_pair(
        record,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "ACCEPTED"
    audited_branch = next(
        row
        for row in result["branches"]
        if row["branch_id"] == branch["branch_id"]
    )
    assert audited_branch["teacher_first_attempt_action_matched"] is False
    assert (
        audited_branch["teacher_first_attempt_measurement_source"]
        == "teacher_unforced_first_action_measurement"
    )

    report, accepted, _ = audit.audit_candidates(
        registry=reg,
        candidates=[record],
        phase="formal",
    )
    assert len(accepted) == 1
    assert (
        report["action_switch_diagnostic"][
            "teacher_action_switch_accuracy"
        ]
        == 0.5
    )


def test_v610_explicitly_unavailable_user_database_is_accepted_but_missing_key_is_not():
    reg, pair, record = v610_registry_record("HIGH")
    record["environment_snapshot"]["user_db_hash"] = None
    record["clean_trace"]["final_user_db_hash"] = None
    for branch in record["branches"]:
        branch["database_hashes"]["user_before_error"] = None
        branch["database_hashes"]["user_after_error"] = None
        branch["matched_replay"]["final_user_db_hash"] = None

    accepted = audit.audit_candidate_pair(
        record,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert accepted["audit_status"] == "ACCEPTED"
    assert accepted["user_db_hash_available"] is False
    assert all(
        branch["user_db_hash_available"] is False
        for branch in accepted["branches"]
    )

    missing = deepcopy(record)
    missing["branches"][0]["database_hashes"].pop("user_before_error")
    rejected = audit.audit_candidate_pair(
        missing,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert rejected["audit_status"] == "REJECTED"
    missing_branch_id = missing["branches"][0]["branch_id"]
    assert (
        rejected["audit_checks"]["branches"][missing_branch_id][
            "failed_injection_user_db_unchanged"
        ]
        is False
    )


def test_v610_missing_or_forced_first_action_probe_is_rejected():
    reg, pair, missing = v610_registry_record()
    missing["branches"][0].pop(
        "teacher_unforced_first_action_measurement"
    )
    result = audit.audit_candidate_pair(
        missing,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    checks = result["audit_checks"]["branches"][
        missing["branches"][0]["branch_id"]
    ]
    assert result["audit_status"] == "REJECTED"
    assert checks["measurement_mode_is_unforced_unretried"] is False

    reg, pair, forced = v610_registry_record()
    measurement = forced["branches"][0][
        "teacher_unforced_first_action_measurement"
    ]
    measurement["forced_first"] = True
    result = audit.audit_candidate_pair(
        forced,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    checks = result["audit_checks"]["branches"][
        forced["branches"][0]["branch_id"]
    ]
    assert result["audit_status"] == "REJECTED"
    assert checks["measurement_mode_is_unforced_unretried"] is False


def test_v610_causal_cells_reject_oracle_wrong_index_and_seed_drift():
    reg, pair, oracle = v610_registry_record()
    oracle["forced_first_cells"]["q_e1_a1"][
        "continuation_provenance"
    ] = "oracle_reference_completion"
    result = audit.audit_candidate_pair(
        oracle,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "v610_causal_cells_have_no_reference_or_oracle_helper"
        ]
        is False
    )

    reg, pair, wrong_index = v610_registry_record()
    wrong_index["forced_first_cells"]["q_e1_a1"][
        "forced_reference_action_index"
    ] = pair["branches"][1]["corrective_action_spec"][
        "reference_action_index"
    ]
    result = audit.audit_candidate_pair(
        wrong_index,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "v610_causal_cells_exact_registered_reference_indices"
        ]
        is False
    )

    reg, pair, seed_drift = v610_registry_record()
    cell = seed_drift["forced_first_cells"]["q_e2_a2"]
    drifted = [20260806, 20260807, 999]
    for trial, seed in zip(cell["trials"], drifted):
        trial["seed"] = seed
    cell["continuation_seed_set_sha256"] = audit.sha256(drifted)
    result = audit.audit_candidate_pair(
        seed_drift,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "v610_scientific_cells_use_three_registered_seeds"
        ]
        is False
    )


def test_v610_sanitized_plan_and_receipt_hash_drift_are_rejected():
    reg, pair, plan_drift = v610_registry_record()
    task_row = plan_drift["reference_plan_binding"]
    task_row["sanitized_successful_plan_sha256"] = "e" * 64
    task_row["task_preflight_sha256"] = audit.sha256(
        {
            key: value
            for key, value in task_row.items()
            if key != "task_preflight_sha256"
        }
    )
    result = audit.audit_candidate_pair(
        plan_drift,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"]["sanitized_plan_hash_bound"]
        is False
    )

    reg, pair, receipt_drift = v610_registry_record()
    binding = receipt_drift["generation_contract"][
        "runtime_provenance"
    ]["reference_preflight"]
    binding["receipt_sha256"] = "f" * 64
    receipt_drift["generation_contract_sha256"] = audit.sha256(
        receipt_drift["generation_contract"]
    )
    result = audit.audit_candidate_pair(
        receipt_drift,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "preflight_receipt_identity_bound"
        ]
        is False
    )


def test_v610_positive_tool_error_target_is_rejected():
    reg, pair, record = v610_registry_record()
    branch = record["branches"][0]
    branch["recovery_suffix"][1]["error"] = True
    result = audit.audit_candidate_pair(
        record,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    checks = result["audit_checks"]["branches"][branch["branch_id"]]
    assert result["audit_status"] == "REJECTED"
    assert (
        checks["v610_positive_recovery_suffix_has_no_tool_errors"]
        is False
    )
    assert (
        checks["v610_positive_recovery_has_zero_tool_error_targets"]
        is False
    )


def test_v610_static_environment_reward_cannot_replace_dynamic_judge_reward():
    reg, pair, missing_clean_reward = v610_registry_record()
    missing_clean_reward["clean_trace"].pop("official_reward_info")
    result = audit.audit_candidate_pair(
        missing_clean_reward,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "v610_clean_positive_has_full_official_success"
        ]
        is False
    )

    reg, pair, missing_cell_reward = v610_registry_record()
    missing_cell_reward["forced_first_cells"]["q_e1_a1"]["trials"][0].pop(
        "official_reward_info"
    )
    result = audit.audit_candidate_pair(
        missing_cell_reward,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert (
        result["audit_checks"]["top"][
            "v610_causal_cells_use_full_official_task_success"
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


def test_v610_exact_compatibility_pilot_and_formal_gates_can_pass():
    registry_payload = v610_gate_registry()
    cases = (
        (
            "compatibility",
            closure.COMPATIBILITY_TASK_IDS[:22],
            "COMPATIBILITY_RELEASE_AUTHORIZED",
            66,
        ),
        (
            "pilot",
            closure.PROSPECTIVE_PILOT_TASK_IDS[:16],
            "GO_FORMAL_POOL",
            48,
        ),
        (
            "formal",
            closure.FORMAL_TASK_IDS[:48],
            "FORMAL_SELECTION_AUTHORIZED",
            144,
        ),
    )
    for phase, task_ids, expected_status, expected_pairs in cases:
        result = audit._gate(
            registry=registry_payload,
            choice_sets=v610_gate_choice_sets(phase, task_ids),
            phase=phase,
            teacher_action_switch_accuracy=1.0,
            error_blind_action_switch_accuracy=0.5,
        )
        assert result["status"] == expected_status
        assert result["pool"]["accepted_candidate_pairs"] == expected_pairs
        assert result["checks"][
            f"registered_population_is_frozen_"
            f"{len(audit._v610_phase_spec(phase)['task_ids'])}"
        ] is True


def test_v610_gate_rejects_wrong_population_order_or_hash():
    choices = v610_gate_choice_sets(
        "pilot",
        closure.PROSPECTIVE_PILOT_TASK_IDS[:16],
    )
    wrong_order = v610_gate_registry()
    row = wrong_order["phase_registry"]["pilot"]
    row["task_ids"] = list(reversed(row["task_ids"]))
    row["task_ids_sha256"] = audit.sha256(row["task_ids"])
    result = audit._gate(
        registry=wrong_order,
        choice_sets=choices,
        phase="pilot",
        teacher_action_switch_accuracy=1.0,
        error_blind_action_switch_accuracy=0.5,
    )
    assert result["status"] == "STOP_NO_GO"
    assert result["checks"]["registered_population_is_frozen_26"] is False

    wrong_hash = v610_gate_registry()
    wrong_hash["phase_registry"]["formal"][
        "task_ids_sha256"
    ] = "f" * 64
    result = audit._gate(
        registry=wrong_hash,
        choice_sets=v610_gate_choice_sets(
            "formal", closure.FORMAL_TASK_IDS[:48]
        ),
        phase="formal",
        teacher_action_switch_accuracy=1.0,
        error_blind_action_switch_accuracy=0.5,
    )
    assert result["status"] == "STOP_INSUFFICIENT_POOL"
    assert result["checks"]["registered_population_is_frozen_50"] is False


def test_v610_phase_specific_seed_contract_accepts_and_rejects_exactly():
    reg, pair, compatibility = v610_registry_record(
        phase="compatibility"
    )
    result = audit.audit_candidate_pair(
        compatibility,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "ACCEPTED"

    reg, pair, wrong_compatibility = v610_registry_record(
        phase="compatibility",
        continuation_seeds=audit.V6_10_CONTINUATION_SEEDS,
    )
    result = audit.audit_candidate_pair(
        wrong_compatibility,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "REJECTED"
    assert result["audit_checks"]["top"][
        "v610_compatibility_cells_use_one_registered_seed"
    ] is False

    for phase in ("pilot", "formal"):
        reg, pair, correct = v610_registry_record(phase=phase)
        result = audit.audit_candidate_pair(
            correct,
            pair,
            registry_sha256=reg["registry_sha256"],
            registry_root=reg,
        )
        assert result["audit_status"] == "ACCEPTED"

        reg, pair, wrong = v610_registry_record(
            phase=phase,
            continuation_seeds=audit.V6_10_COMPATIBILITY_SEEDS,
        )
        result = audit.audit_candidate_pair(
            wrong,
            pair,
            registry_sha256=reg["registry_sha256"],
            registry_root=reg,
        )
        assert result["audit_status"] == "REJECTED"
        assert result["audit_checks"]["top"][
            "v610_scientific_cells_use_three_registered_seeds"
        ] is False


def test_v610_compatibility_phase_is_supported_by_public_audit_api():
    reg, _, record = v610_registry_record(phase="compatibility")
    report, accepted, freeze = audit.audit_candidates(
        registry=reg,
        candidates=[record],
        phase="compatibility",
    )
    assert len(accepted) == 1
    assert report["phase"] == "compatibility"
    assert report["gate"]["status"] == "STOP_COMPATIBILITY_NO_GO"
    assert report["global_checks"][
        "v610_generation_closure_terminal_and_bound"
    ] is False
    assert freeze["status"] == "NOT_FROZEN_FAIL_CLOSED"


def test_v610_actual_generator_raw_probe_schema_is_accepted_end_to_end():
    reg, pair, record = v610_registry_record()
    branch = record["branches"][0]
    registered_branch = pair["branches"][0]
    expected = deepcopy(
        registered_branch["corrective_action_spec"][
            "forced_first_action_constructor"
        ]["tool_call"]
    )
    raw_response = {
        "role": "assistant",
        "content": None,
        "tool_calls": [deepcopy(expected)],
        "generation_time_seconds": 0.25,
    }
    with (
        patch.object(
            generation,
            "raw_teacher_first_response",
            return_value=(raw_response, 0.25),
        ),
        patch.object(generation, "write_json"),
    ):
        generated = (
            generation.teacher_unforced_first_action_measurement(
                SimpleNamespace(
                    teacher_model="teacher/model",
                    teacher_revision="teacher-revision",
                ),
                task=object(),
                domain=pair["domain"],
                error_prompt=branch["recovery_prompt"],
                expected_corrective_call=expected,
                measurement_seed=registered_branch["recovery_seed"],
                log_root=Path("/tmp/v610-audit-raw-probe"),
            )
        )
    branch["teacher_unforced_first_action_measurement"] = generated
    result = audit.audit_candidate_pair(
        record,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert result["audit_status"] == "ACCEPTED"
    checks = result["audit_checks"]["branches"][branch["branch_id"]]
    assert checks["raw_first_response_shape_and_hash_exact"] is True
    assert checks[
        "unforced_probe_no_normalization_or_continuation"
    ] is True


def test_v610_raw_text_multi_call_and_malformed_probe_count_incorrect():
    malformed_responses = (
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [],
        },
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [{"name": "lookup", "arguments": {}}],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"name": "lookup", "arguments": {}},
                {"name": "lookup", "arguments": {"id": "second"}},
            ],
        },
    )
    for raw_response in malformed_responses:
        reg, pair, record = v610_registry_record()
        branch = record["branches"][0]
        registered_branch = pair["branches"][0]
        expected = deepcopy(
            registered_branch["corrective_action_spec"][
                "forced_first_action_constructor"
            ]["tool_call"]
        )
        with (
            patch.object(
                generation,
                "raw_teacher_first_response",
                return_value=(deepcopy(raw_response), 0.1),
            ),
            patch.object(generation, "write_json"),
        ):
            generated = (
                generation.teacher_unforced_first_action_measurement(
                    SimpleNamespace(
                        teacher_model="teacher/model",
                        teacher_revision="teacher-revision",
                    ),
                    task=object(),
                    domain=pair["domain"],
                    error_prompt=branch["recovery_prompt"],
                    expected_corrective_call=expected,
                    measurement_seed=registered_branch[
                        "recovery_seed"
                    ],
                    log_root=Path("/tmp/v610-audit-malformed-probe"),
                )
            )
        assert generated["matched_registered_corrective_accuracy"] == 0.0
        branch["teacher_unforced_first_action_measurement"] = generated
        result = audit.audit_candidate_pair(
            record,
            pair,
            registry_sha256=reg["registry_sha256"],
            registry_root=reg,
        )
        assert result["audit_status"] == "ACCEPTED"
        audited_branch = next(
            row
            for row in result["branches"]
            if row["branch_id"] == branch["branch_id"]
        )
        assert (
            audited_branch["teacher_first_attempt_action_matched"]
            is False
        )


def test_v610_generation_closure_accepts_exact_typed_rejections():
    registry_payload, candidates, closure_bundle = (
        v610_generation_closure_fixture()
    )
    normalized, checks = audit._v610_generation_closure_audit(
        registry=registry_payload,
        candidates=candidates,
        phase="compatibility",
        generation_closure=closure_bundle,
    )
    assert all(checks.values())
    assert normalized["status"] == "PASS"
    assert normalized["rejected_tasks"] == 23


def test_v610_generation_closure_rejects_missing_or_unrun_task():
    registry_payload, candidates, closure_bundle = (
        v610_generation_closure_fixture()
    )
    closure_bundle["task_receipts"].pop()
    receipt_hashes = {
        row["task_identity"]: row["task_receipt_sha256"]
        for row in closure_bundle["task_receipts"]
    }
    closure_bundle["task_receipts_sha256"] = audit.sha256(
        dict(sorted(receipt_hashes.items()))
    )
    _, checks = audit._v610_generation_closure_audit(
        registry=registry_payload,
        candidates=candidates,
        phase="compatibility",
        generation_closure=closure_bundle,
    )
    assert checks["closure_semantic_hashes_recomputed"] is True
    assert checks["exact_phase_task_receipt_population"] is False
    assert checks["generation_receipt_recomputes_terminal_closure"] is False


def test_v610_generation_closure_rejects_corrupt_failure_ledger():
    registry_payload, candidates, closure_bundle = (
        v610_generation_closure_fixture()
    )
    closure_bundle["failure_ledger"][0][
        "reason_code"
    ] = "POSITIVE_SUFFIX_INVALID"
    closure_bundle["failure_ledger_sha256"] = audit.sha256(
        closure_bundle["failure_ledger"]
    )
    _, checks = audit._v610_generation_closure_audit(
        registry=registry_payload,
        candidates=candidates,
        phase="compatibility",
        generation_closure=closure_bundle,
    )
    assert checks["closure_semantic_hashes_recomputed"] is True
    assert checks[
        "failure_ledger_exactly_matches_typed_rejections"
    ] is False


def test_v610_generation_closure_rejects_same_id_other_attempt_splice():
    registry_payload, candidates, closure_bundle = (
        v610_generation_closure_fixture()
    )
    candidates[0]["generation_contract"]["runtime_provenance"][
        "attempt_id"
    ] = "stale-other-attempt"
    candidates[0]["generation_contract_sha256"] = audit.sha256(
        candidates[0]["generation_contract"]
    )
    _, checks = audit._v610_generation_closure_audit(
        registry=registry_payload,
        candidates=candidates,
        phase="compatibility",
        generation_closure=closure_bundle,
    )
    assert checks[
        "submitted_pairs_exactly_partition_to_passing_tasks"
    ] is True
    assert checks[
        "submitted_candidate_content_bound_to_pass_receipts"
    ] is False


def test_v610_audit_output_materializes_with_truthful_oracle_provenance():
    reg, pair, record = v610_registry_record()
    for branch in record["branches"]:
        selected_indices = [
            index
            for index, selected in enumerate(
                branch["full_assistant_label_mask"]
            )
            if selected
        ]
        branch["token_contract"] = {
            "sequence_tokens": 100,
            "supervised_tokens": 50,
            "label_spans": [
                {
                    "message_index": index,
                    "token_start": span_index * 25,
                    "token_end": (span_index + 1) * 25,
                }
                for span_index, index in enumerate(selected_indices)
            ],
        }
    audited = audit.audit_candidate_pair(
        record,
        pair,
        registry_sha256=reg["registry_sha256"],
        registry_root=reg,
    )
    assert audited["audit_status"] == "ACCEPTED"
    assert all(
        branch["recovery_suffix_provenance"]["origin"]
        == "sanitized_deterministic_reference_plan"
        for branch in audited["branches"]
    )

    selected = {
        "candidate_pair_id": audited["candidate_pair_id"],
        "task_identity": audited["task_identity"],
        "domain": audited["domain"],
        "branch_ids": sorted(
            branch["branch_id"] for branch in audited["branches"]
        ),
        "c_sup": audited["c_sup"],
        "c_nonpad": audited["c_nonpad"],
        "candidate_pair": audited,
    }
    manifest = {
        "protocol": materialize.MANIFEST_PROTOCOL,
        "selector": "full_proposed",
        "selection_seed": None,
        "candidate_pool_sha256": "3" * 64,
        "matched_task_ids": [audited["task_identity"]],
        "selected": [selected],
        "budget": {
            "target_c_sup": audited["c_sup"],
            "actual_c_sup": audited["c_sup"],
            "remaining_c_sup": 0,
            "actual_c_nonpad": audited["c_nonpad"],
        },
        "official_test_used": False,
    }

    def frozen_contract(messages, mask, schemas):
        del messages, schemas
        indices = [
            index for index, selected_value in enumerate(mask)
            if selected_value
        ]
        return {
            "sequence_tokens": 100,
            "supervised_tokens": 50,
            "label_spans": [
                {
                    "message_index": index,
                    "token_start": span_index * 25,
                    "token_end": (span_index + 1) * 25,
                }
                for span_index, index in enumerate(indices)
            ],
        }

    rows, materialization_audit = materialize.materialize_manifest(
        manifest,
        tokenizer_name="fake-qwen",
        tokenizer_revision="4" * 64,
        contract_builder=frozen_contract,
        contract_builder_name="tests.v610_frozen_contract",
    )
    assert len(rows) == 2
    assert materialization_audit["status"] == "PASS"
    assert all(
        row["metadata"]["recovery_suffix_origin"]
        == "sanitized_deterministic_reference_plan"
        for row in rows
    )

    tampered = deepcopy(manifest)
    tampered["selected"][0]["candidate_pair"]["branches"][0][
        "recovery_suffix_provenance"
    ]["sanitized_reference_plan_sha256"] = "f" * 64
    try:
        materialize.materialize_manifest(
            tampered,
            tokenizer_name="fake-qwen",
            tokenizer_revision="4" * 64,
            contract_builder=frozen_contract,
            contract_builder_name="tests.v610_frozen_contract",
        )
    except materialize.V6MaterializationError:
        pass
    else:
        raise AssertionError("tampered provenance hash was accepted")
