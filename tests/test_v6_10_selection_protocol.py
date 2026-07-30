from pathlib import Path

import yaml

from scripts import v6_10_selection_protocol as protocol


def test_v6_10_task_sets_are_disjoint_and_complete():
    compatibility = set(protocol.COMPATIBILITY_TASK_IDS)
    prospective = set(protocol.PROSPECTIVE_PILOT_TASK_IDS)
    assert len(compatibility) == 24
    assert len(prospective) == 26
    assert compatibility.isdisjoint(prospective)
    assert set(protocol.FORMAL_TASK_IDS) == compatibility | prospective
    assert protocol.sha256(list(protocol.COMPATIBILITY_TASK_IDS)) == (
        protocol.COMPATIBILITY_TASK_IDS_SHA256
    )
    assert protocol.sha256(list(protocol.PROSPECTIVE_PILOT_TASK_IDS)) == (
        protocol.PROSPECTIVE_PILOT_TASK_IDS_SHA256
    )
    assert protocol.sha256(list(protocol.FORMAL_TASK_IDS)) == (
        protocol.FORMAL_TASK_IDS_SHA256
    )


def test_v6_10_pilot_gate_is_arithmetically_possible():
    assert protocol.PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS == 16
    assert protocol.PROSPECTIVE_MIN_ACCEPTED_PAIRS == 48
    assert (
        protocol.PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS
        * protocol.PAIRS_PER_TASK
        >= protocol.PROSPECTIVE_MIN_ACCEPTED_PAIRS
    )


def test_v6_10_separates_positive_data_and_causal_measurement():
    assert protocol.REFERENCE_PREFLIGHT_PROTOCOL == (
        "v6_reference_execution_preflight_v1"
    )
    assert protocol.POSITIVE_DATA_CONSTRUCTOR == (
        "sanitized_deterministic_reference_plan"
    )
    assert protocol.CAUSAL_CONTINUATION_POLICY == "fresh_teacher_gold_free"
    assert protocol.ACTION_SWITCH_PROBE == (
        "unforced_unretried_teacher_first_action"
    )


def test_v6_10_required_release_artifacts_match_runtime_names():
    required = set(protocol.REQUIRED_RELEASE_ARTIFACTS)
    assert {
        "release_manifest.json",
        "source_container_provenance.json",
        "model_server_receipts.json",
        "runtime_receipt_hashes.json",
        "reference_preflight_receipt.json",
        "registry.runtime.json",
        "run_contract.json",
        "selected_phase_pre_model_preflight.json",
        "tasks/",
        "failure_ledger.jsonl",
        "generation_receipt.json",
        "candidate_pairs.unscored.jsonl",
        "audit_report.json",
        "accepted_candidate_pairs.jsonl",
        "freeze_manifest.json",
        "files.sha256",
    } <= required
    assert "source_and_container_provenance.json" not in required
    assert "static_reference_audit.json" not in required
    assert "sanitized_reference_plans.jsonl" not in required
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "v6_10_closure.yaml"
        ).read_text(encoding="utf-8")
    )
    assert {value.rstrip("/") for value in required} == set(
        config["required_artifacts"]
    )
