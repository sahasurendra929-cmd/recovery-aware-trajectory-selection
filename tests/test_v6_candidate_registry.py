from __future__ import annotations

import json
import re
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from scripts import prepare_v6_candidate_registry as registry
from scripts import v6_selection_protocol as protocol


@contextmanager
def expect_raises(error_type: type[Exception], match: str):
    try:
        yield
    except error_type as error:
        assert re.search(match, str(error))
    else:
        raise AssertionError(f"{error_type.__name__} was not raised")


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


def test_config_contract_accepts_only_explicitly_frozen_designs():
    v6 = frozen_config()
    assert registry._config_contract(v6)["protocol"] == protocol.PROTOCOL

    v6_1 = frozen_config()
    v6_1["protocol"] = registry.V6_1_72B_TEACHER_PROTOCOL
    assert (
        registry._config_contract(v6_1)["protocol"]
        == registry.V6_1_72B_TEACHER_PROTOCOL
    )

    unknown = frozen_config()
    unknown["protocol"] = "v6_1_unregistered"
    with expect_raises(registry.V6RegistryError, "explicitly frozen"):
        registry._config_contract(unknown)


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


def synthetic_inputs() -> tuple[dict, dict[str, dict]]:
    split = {
        "domains": {
            "retail": {
                "inner_train_ids": ["1000", "1001"],
                "validation_ids": ["1990"],
                "sealed_test_ids": ["1991"],
            },
            "airline": {
                "inner_train_ids": ["2000", "2001"],
                "validation_ids": ["2990"],
                "sealed_test_ids": ["2991"],
            },
        }
    }
    catalog = {
        "retail:1000": task(1000, "retail"),
        "retail:1001": task(1001, "retail"),
        "airline:2000": task(2000, "airline"),
        "airline:2001": task(2001, "airline"),
    }
    return split, catalog


def test_structural_registry_has_pair_atomic_units_and_executable_specs():
    split, catalog = synthetic_inputs()
    result = registry.build_registry(
        split=split,
        config=frozen_config(),
        task_catalog=catalog,
        strict=False,
    )
    assert result["selection_uses_outcomes"] is False
    assert result["official_test_used"] is False
    assert result["official_test_sealed"] is True
    assert result["selection_unit"] == "candidate_pair"
    assert result["grouping_unit"] == "choice_set"
    assert result["structural_eligibility_sha256"] == registry.sha256(
        result["structural_eligibility_payload"]
    )
    assert len(result["candidate_pairs"]) == 24
    pair = result["candidate_pairs"][0]
    assert len(pair["branches"]) == 2
    assert len(set(pair["expected_canonical_corrective_families"])) == 2
    assert pair["candidate_pair_sha256"] == registry.sha256(
        {key: value for key, value in pair.items() if key != "candidate_pair_sha256"}
    )
    for branch in pair["branches"]:
        injection = branch["injection_spec"]
        corrective = branch["corrective_action_spec"]
        assert injection["expected_state_mutating"] is False
        assert injection["expected_error_predicate"] == {
            "json_path": "$.error",
            "operator": "is",
            "value": True,
        }
        assert injection["error_call"]["name"] in injection["site_tool_allowlist"]
        assert (
            corrective["canonical_family"]
            == f"{branch['tool_name']}::{branch['identifier_key']}"
        )
        assert corrective["forced_first_action_constructor"]["kind"] == (
            "exact_registered_reference_tool_call"
        )
        assert branch["original_identifier"] != branch["mutated_identifier"]
        assert type(branch["original_identifier"]) is type(
            branch["mutated_identifier"]
        )
        assert branch["branch_slot_sha256"] == registry.sha256(
            {
                key: value
                for key, value in branch.items()
                if key != "branch_slot_sha256"
            }
        )


def test_official_test_overlap_and_unsealed_config_fail_closed():
    split, catalog = synthetic_inputs()
    leaked = deepcopy(split)
    leaked["domains"]["retail"]["sealed_test_ids"].append("1000")
    with expect_raises(registry.V6RegistryError, "disjoint"):
        registry.build_registry(
            split=leaked,
            config=frozen_config(),
            task_catalog=catalog,
            strict=False,
        )

    unsealed = frozen_config()
    unsealed["benchmark"]["official_test"]["sealed"] = False
    with expect_raises(registry.V6RegistryError, "fail-closed"):
        registry.build_registry(
            split=split,
            config=unsealed,
            task_catalog=catalog,
            strict=False,
        )


def test_real_pinned_structural_payload_reproduces_frozen_hash():
    repo = Path(__file__).resolve().parents[1]
    candidates = (
        repo / "data" / "raw" / "tau2-bench",
        repo.parent
        / "recovery-trajectory-selection"
        / "data"
        / "raw"
        / "tau2-bench",
    )
    tau2_root = next((path for path in candidates if path.is_dir()), None)
    if tau2_root is None:
        return
    split_path = repo / "artifacts" / "v5_stage0" / "manifests" / "split_manifest.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    catalog, _ = registry.load_task_catalog(tau2_root, split)
    arm_train, _, _ = registry.frozen_arm_train(split, strict=True)
    payload, actions, exclusions = registry.structural_payload(
        arm_train_task_ids=arm_train,
        task_catalog=catalog,
        strict=True,
    )
    assert registry.sha256(payload) == registry.STRUCTURAL_PAYLOAD_SHA256
    assert registry.STRUCTURAL_PAYLOAD_SHA256 == (
        "002b5a2c5d83a4dab6dc8ce398d81bb8541bf7bfc6b25c04e62ce9ed179587f7"
    )
    assert {domain: len(payload["domains"][domain]["eligible_task_ids"]) for domain in ("retail", "airline")} == {
        "retail": 48,
        "airline": 12,
    }
    assert len(actions) == 60
    assert len(exclusions) == 10
