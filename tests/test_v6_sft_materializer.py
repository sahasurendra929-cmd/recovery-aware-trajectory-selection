from __future__ import annotations

from copy import deepcopy
import hashlib

import pytest

from scripts import materialize_v6_sft as materialize


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up one order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]


def call(call_id: str, order_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "arguments": {"order_id": order_id},
                },
            }
        ],
    }


def result(call_id: str, *, error: bool) -> dict:
    return {
        "role": "tool",
        "name": "lookup_order",
        "tool_call_id": call_id,
        "content": "not found" if error else {"status": "paid"},
        "error": error,
    }


def recovery_branch(pair_id: str, suffix: str) -> dict:
    branch_id = f"{pair_id}:{suffix}"
    prompt = [
        {"role": "system", "content": "Frozen retail policy."},
        {"role": "user", "content": "Please check my order."},
        call(f"provider-failed-{suffix}", f"bad-{suffix}"),
        result(f"provider-failed-{suffix}", error=True),
    ]
    fresh = [
        call(f"provider-repair-{suffix}", f"good-{suffix}"),
        result(f"provider-repair-{suffix}", error=False),
        {"role": "assistant", "content": "Your order is paid."},
    ]
    return {
        "branch_id": branch_id,
        "error_family": f"lookup_order::order_id::{suffix}",
        "error_event_sha256": digest(f"{branch_id}:error"),
        "first_recovery_action_key": f"lookup_order::order_id::{suffix}",
        "supervised_target_tokens": 10,
        "nonpadding_tokens": 70,
        "coverage": {
            "failed_tool": "lookup_order",
            "corrective_action": f"lookup_order::order_id::{suffix}",
            "recovery_length_bin": "short",
            "recovery_mode": "agent_initiated",
        },
        "prompt": prompt,
        "fresh_recovery_suffix": fresh,
        # Combined prompt + fresh suffix scope.
        "loss_mask": [False, False, False, False, True, False, True],
        "token_contract": {
            "sequence_tokens": 70,
            "supervised_tokens": 10,
            "label_spans": [
                {"message_index": 4, "token_start": 40, "token_end": 45},
                {"message_index": 6, "token_start": 60, "token_end": 65},
            ],
        },
    }


def candidate_pair(task: str = "retail:1") -> dict:
    pair_id = f"{task}:pair-0"
    return {
        "candidate_pair_id": pair_id,
        "choice_set_id": f"{task}:prefix-0",
        "task_identity": task,
        "domain": "retail",
        "prefix_sha256": digest(f"{task}:prefix"),
        "environment_snapshot_sha256": digest(f"{task}:snapshot"),
        "quality": {
            "real_error_executed": True,
            "matched_recovery_replay_success": True,
            "cross_replay_complete": True,
            "no_future_leakage": True,
            "independent_replay_audited": True,
            "official_test_used": False,
            "failed_positive_labels": 0,
        },
        "tool_schemas": tool_schemas(),
        "branches": [
            recovery_branch(pair_id, "e1"),
            recovery_branch(pair_id, "e2"),
        ],
        "cost": {
            "c_sup": 20,
            "c_nonpad": 140,
            "selector_atom": "complete_candidate_pair_two_sibling_branches",
        },
    }


def recovery_manifest() -> dict:
    pair = candidate_pair()
    return {
        "protocol": materialize.MANIFEST_PROTOCOL,
        "selector": "full_proposed",
        "selection_seed": None,
        "candidate_pool_sha256": digest("pool"),
        "matched_task_ids": ["retail:1"],
        "selected": [
            {
                "candidate_pair_id": pair["candidate_pair_id"],
                "task_identity": pair["task_identity"],
                "domain": pair["domain"],
                "branch_ids": sorted(
                    branch["branch_id"] for branch in pair["branches"]
                ),
                "c_sup": 20,
                "c_nonpad": 140,
                "candidate_pair": pair,
            }
        ],
        "budget": {
            "target_c_sup": 20,
            "actual_c_sup": 20,
            "remaining_c_sup": 0,
            "actual_c_nonpad": 140,
        },
        "official_test_used": False,
    }


def clean_view(task: str = "retail:1") -> dict:
    messages = [
        {"role": "system", "content": "Frozen retail policy."},
        {"role": "user", "content": "Please check and confirm my order."},
        call("clean-1", "order-1"),
        result("clean-1", error=False),
        call("clean-2", "order-2"),
        result("clean-2", error=False),
        {"role": "assistant", "content": "Both orders are paid."},
    ]
    return {
        "clean_id": f"{task}:clean",
        "messages": messages,
        "label_mask": [False, False, True, False, True, False, True],
        "tool_schemas": tool_schemas(),
        "full_trajectory_reward": 1.0,
        "c_sup": 15,
        "c_nonpad": 70,
        "source_candidate_pair_ids": [
            f"{task}:pair-0",
            f"{task}:pair-1",
            f"{task}:pair-2",
        ],
        "token_contract": {
            "sequence_tokens": 70,
            "supervised_tokens": 15,
            "label_spans": [
                {"message_index": 2, "token_start": 20, "token_end": 25},
                {"message_index": 4, "token_start": 40, "token_end": 45},
                {"message_index": 6, "token_start": 60, "token_end": 65},
            ],
        },
    }


def flawless_manifest() -> dict:
    view = clean_view()
    return {
        "protocol": materialize.MANIFEST_PROTOCOL,
        "selector": "flawless_only",
        "selection_seed": None,
        "candidate_pool_sha256": digest("pool"),
        "matched_task_ids": ["retail:1"],
        "selected": [
            {
                "clean_id": view["clean_id"],
                "task_identity": "retail:1",
                "domain": "retail",
                "c_sup": 15,
                "c_nonpad": 70,
                "training_view": "flawless_only",
                "clean_view": view,
            }
        ],
        "budget": {
            "target_c_sup": 15,
            "actual_c_sup": 15,
            "remaining_c_sup": 0,
            "actual_c_nonpad": 70,
            "budget_comparable_primary": False,
        },
        "official_test_used": False,
    }


def fake_contract(
    messages: list[dict], mask: list[bool], schemas: list[dict]
) -> dict:
    del schemas
    spans = [
        {
            "message_index": index,
            "token_start": index * 10,
            "token_end": index * 10 + 5,
        }
        for index, selected in enumerate(mask)
        if selected
    ]
    return {
        "sequence_tokens": len(messages) * 10,
        "supervised_tokens": 5 * len(spans),
        "label_spans": spans,
    }


def run(manifest: dict) -> tuple[list[dict], dict]:
    return materialize.materialize_manifest(
        manifest,
        tokenizer_name="fake-qwen",
        tokenizer_revision=digest("revision"),
        contract_builder=fake_contract,
        contract_builder_name="tests.fake_contract_v1",
    )


def test_recovery_manifest_materializes_both_sibling_branches():
    rows, audit = run(recovery_manifest())
    assert len(rows) == 2
    assert audit["rows"] == audit["expected_rows"] == 2
    assert audit["candidate_pair_atom_split"] is False
    assert audit["failed_action_positive_labels"] == 0
    assert {row["metadata"]["branch_id"].rsplit(":", 1)[-1] for row in rows} == {
        "e1",
        "e2",
    }
    for row in rows:
        assert row["metadata"]["arm"] == "v6_recovery_selected"
        assert row["metadata"]["source"] == "failure_rich"
        assert row["metadata"]["official_test_used"] is False
        assert len(row["messages"]) == 7
        assert row["label_mask"] == [
            False,
            False,
            False,
            False,
            True,
            False,
            True,
        ]
        failed = row["metadata"]["injected_failed_assistant_message_index"]
        assert failed == 2
        assert row["label_mask"][failed] is False
        assert row["messages"][failed + 1]["error"] is True
        assert row["messages"][5]["error"] is False
        assert row["messages"][6]["role"] == "assistant"
        assert row["messages"][6]["content"] == "Your order is paid."
        assert row["label_mask"][6] is True
        assert row["metadata"]["recovery_suffix_origin"] == (
            "fresh_teacher_generation"
        )
        assert row["metadata"]["fresh_recovery_suffix"] is True
        assert row["metadata"]["gold_reference_suffix_used"] is False


def test_v610_sanitized_oracle_suffix_is_not_laundered_as_fresh():
    manifest = recovery_manifest()
    for branch in manifest["selected"][0]["candidate_pair"]["branches"]:
        binding = {
            "reference_preflight_receipt_sha256": digest("receipt"),
            "reference_task_preflight_sha256": digest("task"),
            "sanitized_reference_plan_sha256": digest("plan"),
        }
        branch["reference_preflight_binding"] = binding
        branch["recovery_suffix_provenance"] = {
            "protocol": "v6_10_audited_recovery_suffix_provenance_v1",
            "origin": "sanitized_deterministic_reference_plan",
            "fresh_recovery_generated": False,
            "gold_suffix_used": True,
            **binding,
            "matched_recovery_evidence_sha256": digest("matched"),
        }
    rows, _ = run(manifest)
    assert rows
    for row in rows:
        metadata = row["metadata"]
        assert metadata["recovery_suffix_origin"] == (
            "sanitized_deterministic_reference_plan"
        )
        assert metadata["fresh_recovery_suffix"] is False
        assert metadata["gold_reference_suffix_used"] is True


def test_v610_oracle_suffix_requires_truthful_producer_evidence():
    manifest = recovery_manifest()
    branch = manifest["selected"][0]["candidate_pair"]["branches"][0]
    branch["reference_preflight_binding"] = {"protocol": "fixture-v610"}
    with pytest.raises(
        materialize.V6MaterializationError,
        match="canonical audited sanitized-oracle provenance",
    ):
        run(manifest)


def test_flawless_manifest_materializes_complete_matched_task_control():
    rows, audit = run(flawless_manifest())
    assert len(rows) == 1
    row = rows[0]
    assert audit["selector"] == "flawless_only"
    assert audit["supervised_tokens"] == 15
    assert row["metadata"]["arm"] == "v6_flawless_control"
    assert row["metadata"]["source"] == "perfect_success"
    assert row["metadata"]["failed_assistant_message_indices"] == []
    assert row["metadata"]["source_candidate_pair_ids"] == [
        "retail:1:pair-0",
        "retail:1:pair-1",
        "retail:1:pair-2",
    ]
    assert len(row["messages"]) == 7
    assert row["label_mask"] == [
        False,
        False,
        True,
        False,
        True,
        False,
        True,
    ]


def test_failed_injected_call_cannot_enter_positive_label():
    manifest = recovery_manifest()
    branch = manifest["selected"][0]["candidate_pair"]["branches"][0]
    branch["loss_mask"][2] = True
    with pytest.raises(
        materialize.V6MaterializationError, match="prompt context"
    ):
        run(manifest)


def test_every_successful_fresh_tool_call_must_be_labeled():
    manifest = recovery_manifest()
    branch = manifest["selected"][0]["candidate_pair"]["branches"][0]
    branch["loss_mask"][4] = False
    with pytest.raises(
        materialize.V6MaterializationError, match="exactly all non-failed"
    ):
        run(manifest)


def test_prompt_must_end_at_the_controlled_error_result():
    manifest = recovery_manifest()
    branch = manifest["selected"][0]["candidate_pair"]["branches"][0]
    branch["prompt"].append({"role": "user", "content": "What happened?"})
    branch["loss_mask"].insert(4, False)
    branch["nonpadding_tokens"] = 70
    manifest["selected"][0]["c_nonpad"] = 130
    manifest["selected"][0]["candidate_pair"]["cost"]["c_nonpad"] = 130
    manifest["budget"]["actual_c_nonpad"] = 130
    with pytest.raises(
        materialize.V6MaterializationError, match="prompt must end"
    ):
        run(manifest)


def test_additional_failure_in_fresh_suffix_is_context_only():
    manifest = recovery_manifest()
    branch = manifest["selected"][0]["candidate_pair"]["branches"][0]
    branch["fresh_recovery_suffix"] = [
        call("second-failure", "still-bad"),
        result("second-failure", error=True),
        call("eventual-success", "good"),
        result("eventual-success", error=False),
    ]
    branch["loss_mask"] = [False] * 6 + [True, False]
    branch["supervised_target_tokens"] = 5
    branch["nonpadding_tokens"] = 80
    branch["token_contract"] = {
        "sequence_tokens": 80,
        "supervised_tokens": 5,
        "label_spans": [
            {"message_index": 6, "token_start": 60, "token_end": 65}
        ],
    }
    pair = manifest["selected"][0]["candidate_pair"]
    pair["cost"]["c_sup"] = 15
    pair["cost"]["c_nonpad"] = 150
    manifest["selected"][0]["c_sup"] = 15
    manifest["selected"][0]["c_nonpad"] = 150
    manifest["budget"]["target_c_sup"] = 15
    manifest["budget"]["actual_c_sup"] = 15
    manifest["budget"]["actual_c_nonpad"] = 150
    rows, _ = run(manifest)
    changed = next(row for row in rows if row["metadata"]["branch_id"].endswith("e1"))
    assert changed["label_mask"][4] is False
    assert changed["messages"][5]["error"] is True
    assert changed["label_mask"][6] is True


def test_selector_atom_with_one_branch_fails_closed():
    manifest = recovery_manifest()
    pair = manifest["selected"][0]["candidate_pair"]
    pair["branches"].pop()
    manifest["selected"][0]["branch_ids"].pop()
    with pytest.raises(
        materialize.V6MaterializationError, match="exactly two sibling branches"
    ):
        run(manifest)


def test_registered_token_cost_must_match_materialized_contract():
    manifest = recovery_manifest()
    manifest["selected"][0]["candidate_pair"]["branches"][0][
        "supervised_target_tokens"
    ] = 6
    with pytest.raises(
        materialize.V6MaterializationError, match="registered c_sup"
    ):
        run(manifest)


def test_flawless_id_and_cost_without_complete_view_is_not_training_data():
    manifest = flawless_manifest()
    del manifest["selected"][0]["clean_view"]
    with pytest.raises(
        materialize.V6MaterializationError, match="lacks complete clean_view"
    ):
        run(manifest)


def test_flawless_view_with_error_is_not_flawless():
    manifest = flawless_manifest()
    manifest["selected"][0]["clean_view"]["messages"][3]["error"] = True
    with pytest.raises(
        materialize.V6MaterializationError, match="contains a tool error"
    ):
        run(manifest)


def test_official_test_must_remain_sealed():
    manifest = recovery_manifest()
    manifest["official_test_used"] = True
    with pytest.raises(
        materialize.V6MaterializationError, match="opened the official test"
    ):
        run(manifest)


def test_injected_contract_builder_requires_auditable_identity():
    with pytest.raises(
        materialize.V6MaterializationError, match="contract_builder_name"
    ):
        materialize.materialize_manifest(
            recovery_manifest(),
            tokenizer_name="fake-qwen",
            tokenizer_revision=digest("revision"),
            contract_builder=fake_contract,
        )


def test_tool_schema_drift_is_rejected():
    manifest = recovery_manifest()
    manifest["selected"][0]["candidate_pair"]["branches"][0][
        "tool_schemas"
    ] = [
        {
            "type": "function",
            "function": {"name": "different_tool", "parameters": {}},
        }
    ]
    with pytest.raises(
        materialize.V6MaterializationError, match="tool_schemas differ"
    ):
        run(manifest)


def test_candidate_quality_future_leakage_seal_is_required():
    manifest = recovery_manifest()
    manifest["selected"][0]["candidate_pair"]["quality"][
        "no_future_leakage"
    ] = False
    with pytest.raises(
        materialize.V6MaterializationError, match="no_future_leakage"
    ):
        run(manifest)
