from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import prepare_v6_candidate_registry as legacy
from scripts import prepare_v6_10_registry as registry
from scripts import v6_10_selection_protocol as closure
from scripts import v6_reference_contract as reference


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "v6_10_closure.yaml"
    return legacy.load_config(path)


def test_source_provenance_rejects_untracked_drift(monkeypatch):
    calls = []

    def fake_git_output(root, *arguments):
        del root
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return "b" * 40
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return "?? untracked-evidence"
        raise AssertionError(arguments)

    monkeypatch.setattr(registry, "_git_output", fake_git_output)
    with pytest.raises(
        registry.V610RegistryError,
        match="tracked or untracked drift",
    ):
        registry.source_provenance(Path("/frozen/source"))
    assert (
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ) in calls


def _task(identity: str) -> dict:
    domain, task_id = identity.split(":", 1)
    if domain == "retail":
        actions = [
            {
                "action_id": f"{task_id}-0",
                "requestor": "assistant",
                "name": "get_order_details",
                "arguments": {"order_id": f"#W{int(task_id):07d}"},
            },
            {
                "action_id": f"{task_id}-1",
                "requestor": "assistant",
                "name": "get_user_details",
                "arguments": {"user_id": f"user_{int(task_id):04d}"},
            },
            {
                "action_id": f"{task_id}-2",
                "requestor": "assistant",
                "name": "cancel_pending_order",
                "arguments": {"order_id": f"#W{int(task_id):07d}"},
            },
        ]
    else:
        actions = [
            {
                "action_id": f"{task_id}-0",
                "requestor": "assistant",
                "name": "get_reservation_details",
                "arguments": {"reservation_id": f"R{int(task_id):05d}"},
            },
            {
                "action_id": f"{task_id}-1",
                "requestor": "assistant",
                "name": "get_user_details",
                "arguments": {"user_id": f"user_{int(task_id):04d}"},
            },
            {
                "action_id": f"{task_id}-2",
                "requestor": "assistant",
                "name": "cancel_reservation",
                "arguments": {"reservation_id": f"R{int(task_id):05d}"},
            },
        ]
    return {
        "id": int(task_id),
        "evaluation_criteria": {"actions": actions},
    }


def _split() -> dict:
    by_domain = {"retail": [], "airline": []}
    for identity in closure.FORMAL_TASK_IDS:
        domain, task_id = identity.split(":", 1)
        by_domain[domain].append(task_id)
    return {
        "domains": {
            domain: {
                "inner_train_ids": values,
                "validation_ids": [],
                "sealed_test_ids": [],
            }
            for domain, values in by_domain.items()
        }
    }


def _row(identity: str, task: dict, *, status: str = "PASS") -> dict:
    slots = reference.build_reference_slots(
        identity,
        task["evaluation_criteria"]["actions"],
    )
    indices = [0, 1, 2] if status == "PASS" else []
    row = {
        "protocol": reference.RECEIPT_PROTOCOL,
        "design_protocol": reference.DESIGN_PROTOCOL,
        "environment_reward_scope": reference.ENVIRONMENT_REWARD_SCOPE,
        "task_identity": identity,
        "status": status,
        "reason_codes": ["PASS"] if status == "PASS" else ["FIXTURE_REJECTED"],
        "reference_action_count": len(slots),
        "reference_slots": slots,
        "reference_slots_sha256": reference.sha256(
            [slot["reference_slot_sha256"] for slot in slots]
        ),
        "renderer_lint": {"status": "PASS"},
        "reference_outcome_contract": {
            "status": "PASS",
            "environment_reward_scope": reference.ENVIRONMENT_REWARD_SCOPE,
            "raw_reference_environment_reward": 1.0,
            "raw_reference_repeat_environment_reward": 1.0,
            "sanitized_environment_reward": 1.0,
            "sanitized_plan_all_actions_succeeded": True,
            "sanitized_final_state_equivalent": True,
            "sanitized_tool_error_indices": [],
            "full_official_reward_status": (
                reference.FULL_OFFICIAL_REWARD_STATUS
            ),
        },
        "expected_error_indices": [],
        "expected_error_set_sha256": reference.sha256([]),
        "sanitized_successful_reference_indices": indices,
        "sanitized_successful_plan_sha256": reference.sha256(indices),
        "sanitized_reference_indices": indices,
        "eligible_forced_first_reference_indices": indices,
        "eligible_forced_first_reference_indices_sha256": reference.sha256(
            indices
        ),
        "ineligible_forced_first_reference_indices": [],
        "forced_first_slot_screen": {
            "status": "PASS",
            "eligible_forced_first_reference_indices": indices,
            "rows": [],
        },
        "raw_reference_environment_reward": (
            1.0 if status == "PASS" else None
        ),
        "raw_reference_repeat_environment_reward": (
            1.0 if status == "PASS" else None
        ),
        "sanitized_environment_reward": (
            1.0 if status == "PASS" else None
        ),
        "full_official_reward_status": (
            reference.FULL_OFFICIAL_REWARD_STATUS
        ),
        "official_test_used": False,
    }
    row["task_preflight_sha256"] = reference.sha256(row)
    return row


def _receipt(
    catalog: dict[str, dict],
    *,
    rejected: str | None = None,
) -> dict:
    digest = "a" * 64
    rows = [
        _row(
            identity,
            catalog[identity],
            status="REJECTED" if identity == rejected else "PASS",
        )
        for identity in closure.FORMAL_TASK_IDS
    ]
    return reference.build_preflight_receipt(
        expected_task_ids=closure.FORMAL_TASK_IDS,
        task_results=rows,
        provenance={
            "source": {
                "commit": "b" * 40,
                "tree": digest,
                "raw_git_tree_object": "c" * 40,
                "tracked_worktree_clean": True,
            },
            "tau2": {
                "commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
                "tree": digest,
                "raw_git_tree_object": "d" * 40,
                "tracked_worktree_clean": True,
            },
            "files": {
                "preflight_script_sha256": digest,
                "contract_module_sha256": digest,
                "config_sha256": digest,
                "split_manifest_sha256": digest,
            },
        },
        task_source_hashes={
            identity: reference.sha256(catalog[identity])
            for identity in closure.FORMAL_TASK_IDS
        },
    )


def _build(*, rejected: str | None = None) -> dict:
    catalog = {
        identity: _task(identity) for identity in closure.FORMAL_TASK_IDS
    }
    receipt = _receipt(catalog, rejected=rejected)
    return registry.build_registry(
        split=_split(),
        config=_config(),
        task_catalog=catalog,
        preflight_receipt=receipt,
        preflight_file_sha256="e" * 64,
        split_file_sha256="a" * 64,
        config_file_sha256="a" * 64,
        source={"commit": "b" * 40},
        strict=False,
    )


def test_registry_v2_freezes_24_26_50_and_exact_preflight_slots():
    payload = _build()
    assert registry.verify_registry(payload) == payload["registry_sha256"]
    assert payload["protocol"] == closure.REGISTRY_PROTOCOL
    assert payload["reference_preflight_receipt_sha256"] != (
        payload["reference_preflight_file_sha256"]
    )
    assert {
        phase: payload["phase_registry"][phase]["task_count"]
        for phase in ("compatibility", "pilot", "formal")
    } == {"compatibility": 24, "pilot": 26, "formal": 50}
    assert len(payload["candidate_pairs"]) == (24 + 26 + 50) * 3

    pair = payload["candidate_pairs"][0]
    assert len(pair["branches"]) == 2
    indices = []
    for branch in pair["branches"]:
        binding = branch["reference_preflight_binding"]
        corrective = branch["corrective_action_spec"]
        assert binding["reference_action_index"] == corrective[
            "reference_action_index"
        ]
        assert binding["reference_slot_id"] == corrective["reference_slot_id"]
        assert binding["sanitized_reference_plan_sha256"]
        assert binding["reference_preflight_receipt_sha256"] == payload[
            "reference_preflight_receipt_sha256"
        ]
        assert binding["reference_preflight_file_sha256"] == payload[
            "reference_preflight_file_sha256"
        ]
        assert corrective["sanitized_plan_member"] is True
        indices.append(binding["reference_action_index"])
    assert len(set(indices)) == 2


def test_reference_rejection_is_typed_and_excluded_from_every_phase():
    rejected = "retail:35"
    payload = _build(rejected=rejected)
    assert registry.verify_registry(payload)
    assert any(
        row["task_identity"] == rejected
        and row["reason_code"] == "REFERENCE_PREFLIGHT_REJECTED"
        for row in payload["reference_preflight_exclusions"]
    )
    assert not any(
        pair["task_identity"] == rejected
        for pair in payload["candidate_pairs"]
    )
    assert payload["phase_registry"]["formal"]["eligible_task_count"] == 49


def test_registry_verifier_rejects_pair_or_slot_tampering():
    payload = _build()
    damaged = deepcopy(payload)
    damaged["candidate_pairs"][0]["branches"][0][
        "reference_preflight_binding"
    ]["reference_action_index"] = 99
    # Rehashing only the registry cannot hide the nested pair/branch drift.
    damaged.pop("registry_sha256")
    damaged["registry_sha256"] = registry.sha256(damaged)
    try:
        registry.verify_registry(damaged)
    except registry.V610RegistryError as error:
        assert "pair self-hash drift" in str(error)
    else:
        raise AssertionError("tampered exact reference slot was accepted")
