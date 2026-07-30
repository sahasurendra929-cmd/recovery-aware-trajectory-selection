#!/usr/bin/env python3
"""Frozen closure contract for the V6.10 recovery-selection experiment.

V6.10 is one scientific revision, not another task-specific patch.  It
separates three operations that V6.9 accidentally conflated:

* deterministic, sanitized reference plans construct positive SFT examples;
* a gold-free frozen policy supplies the continuation in the four causal
  ``kappa`` cells; and
* an unforced, unretried policy probe measures first-action switching.

The 24 tasks inspected while repairing V6.0--V6.9 are a compatibility suite.
They are useful engineering regressions but are no longer prospective
scientific evidence.  The remaining 26 action-identifiable arm-train tasks
form the only prospective Pilot gate.  The official test remains sealed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

if __package__:
    from scripts import v6_selection_protocol as base
else:  # pragma: no cover - direct script execution
    import v6_selection_protocol as base


PROTOCOL = "v6_10_pipeline_closure_v1"
DESIGN_VERSION = "6.10"
REGISTRY_PROTOCOL = "v6_executable_candidate_pair_registry_v2"
REFERENCE_PREFLIGHT_PROTOCOL = "v6_reference_execution_preflight_v1"

# These tasks were repeatedly observed while fixing V6.0--V6.9.  They may be
# used to verify engineering compatibility, never to claim prospective yield.
COMPATIBILITY_TASK_IDS = tuple(base.PILOT_TASK_IDS)

# Frozen before any V6.10 generation outcome.  This is exactly the complement
# of the compatibility suite within the 50 V6 action-identifiable tasks.
PROSPECTIVE_PILOT_TASK_IDS = (
    "airline:1",
    "airline:17",
    "airline:23",
    "airline:38",
    "airline:43",
    "retail:2",
    "retail:3",
    "retail:4",
    "retail:8",
    "retail:10",
    "retail:13",
    "retail:14",
    "retail:15",
    "retail:25",
    "retail:28",
    "retail:29",
    "retail:43",
    "retail:46",
    "retail:54",
    "retail:59",
    "retail:66",
    "retail:67",
    "retail:69",
    "retail:78",
    "retail:110",
    "retail:112",
)


def _task_sort_key(identity: str) -> tuple[str, int]:
    domain, task_id = identity.split(":", 1)
    return domain, int(task_id)


FORMAL_TASK_IDS = tuple(
    sorted(
        {*COMPATIBILITY_TASK_IDS, *PROSPECTIVE_PILOT_TASK_IDS},
        key=_task_sort_key,
    )
)

COMPATIBILITY_TASKS = 24
PROSPECTIVE_PILOT_TASKS = 26
FORMAL_TASKS = 50
PAIRS_PER_TASK = base.MIN_PAIRS_PER_TASK
COMPATIBILITY_TASK_IDS_SHA256 = (
    "5170b139c6307a3f46e29ab9db2573b864181c7b8ff85e27452e14f591bf27c4"
)
PROSPECTIVE_PILOT_TASK_IDS_SHA256 = (
    "930e0a0f9b6b7285a14d17d98b0b8f789f70e26b4c4625f8ac5f6cdd9eae0edb"
)
FORMAL_TASK_IDS_SHA256 = (
    "e2433544a160bdbde7c64318309c2f541833a45b3d907c1de3f02503164e0801"
)

# 48 pairs and three pairs per task imply at least 16 complete tasks.
PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS = 16
PROSPECTIVE_MIN_ACCEPTED_PAIRS = 48

POSITIVE_DATA_CONSTRUCTOR = "sanitized_deterministic_reference_plan"
CAUSAL_CONTINUATION_POLICY = "fresh_teacher_gold_free"
ACTION_SWITCH_PROBE = "unforced_unretried_teacher_first_action"

REQUIRED_PROVENANCE_FIELDS = (
    "attempt_id",
    "source_commit",
    "source_tree_clean",
    "generation_script_sha256",
    "tau2_commit",
    "tau2_tree_clean",
    "reference_preflight_sha256",
    "container_image_digest",
    "model_server_receipts_sha256",
)

REQUIRED_RELEASE_ARTIFACTS = (
    "release_manifest.json",
    "service_specs.json",
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
    "candidate_pool.unscored.sha256",
    "candidate_pairs.measured.jsonl",
    "audit_report.json",
    "accepted_candidate_pairs.jsonl",
    "freeze_manifest.json",
    "scored_candidate_pairs.jsonl",
    "scoring_audit.json",
    "files.sha256",
    "selector_manifests/",
    "materialization_audits/",
    "training_run_manifests/",
    "checkpoint_registry.json",
    "official_test_unseal_receipt.json",
    "per_task_evaluation.jsonl",
    "statistical_summary.json",
    "console_logs/",
)


class V610ProtocolError(RuntimeError):
    """The V6.10 closure contract was violated."""


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def validate_constants() -> None:
    compatibility = set(COMPATIBILITY_TASK_IDS)
    prospective = set(PROSPECTIVE_PILOT_TASK_IDS)
    formal = set(FORMAL_TASK_IDS)
    if len(compatibility) != COMPATIBILITY_TASKS:
        raise V610ProtocolError("compatibility task count drift")
    if len(prospective) != PROSPECTIVE_PILOT_TASKS:
        raise V610ProtocolError("prospective Pilot task count drift")
    if compatibility & prospective:
        raise V610ProtocolError("compatibility and prospective tasks overlap")
    if formal != compatibility | prospective or len(formal) != FORMAL_TASKS:
        raise V610ProtocolError("formal task universe drift")
    observed_hashes = {
        "compatibility": sha256(list(COMPATIBILITY_TASK_IDS)),
        "pilot": sha256(list(PROSPECTIVE_PILOT_TASK_IDS)),
        "formal": sha256(list(FORMAL_TASK_IDS)),
    }
    expected_hashes = {
        "compatibility": COMPATIBILITY_TASK_IDS_SHA256,
        "pilot": PROSPECTIVE_PILOT_TASK_IDS_SHA256,
        "formal": FORMAL_TASK_IDS_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise V610ProtocolError(
            f"ordered task population hash drift: {observed_hashes}"
        )
    if PROSPECTIVE_MIN_TASKS_WITH_THREE_PAIRS * PAIRS_PER_TASK < (
        PROSPECTIVE_MIN_ACCEPTED_PAIRS
    ):
        raise V610ProtocolError("prospective Pilot pair gate is impossible")
    if base.TAU2_COMMIT != "fc0055dc4e0a316c3f83133267fbd6faaa770992":
        raise V610ProtocolError("tau2 commit drift")


validate_constants()
