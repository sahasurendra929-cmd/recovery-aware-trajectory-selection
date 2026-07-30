"""Truthful local V6.10 ``retail:35`` regression through SFT materialization.

This is deliberately a one-task engineering projection, not a counterfeit
formal 50-task gate.  It executes the pinned tau2 environment, reference
preflight, injected failures, deterministic sanitized positives, replay
audits, selector, and materializer.  Only the external teacher/user generation
service boundary and the frozen-student token/log-prob service boundary are
deterministic CPU fakes, so the test needs neither network nor a GPU.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import audit_v6_candidates as audit
from scripts import build_v6_selector_manifests as selectors
from scripts import materialize_v6_sft as materialize
from scripts import measure_v6_candidate_tokens as measurement
from scripts import preflight_v6_reference_traces as static_preflight
from scripts import prepare_v6_10_registry as registry
from scripts import prepare_v6_candidate_registry as legacy_registry
from scripts import run_v6_candidate_generation as generation
from scripts import score_v6_candidates as scoring
from scripts import v6_10_selection_protocol as closure
from scripts import v6_reference_contract as reference
from scripts import v6_selection_protocol as selection


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = REPO_ROOT.parent / "recovery-trajectory-selection"
TAU2_ROOT = LEGACY_ROOT / "data" / "raw" / "tau2-bench"
SPLIT_PATH = (
    LEGACY_ROOT / "artifacts" / "v5_stage0" / "manifests" / "split_manifest.json"
)
CONFIG_PATH = REPO_ROOT / "configs" / "v6_10_closure.yaml"

pytestmark = pytest.mark.skipif(
    not (TAU2_ROOT / "src" / "tau2").is_dir(),
    reason="pinned local tau2 checkout is unavailable",
)


class PrefixStableFakeTokenizer:
    """Deterministic chat tokenizer used only at the tokenizer boundary."""

    @staticmethod
    def _message_tokens(message):
        role = message["role"]
        if role == "assistant":
            body = str(message.get("content") or "")
            calls = message.get("tool_calls") or []
            width = max(1, len(body.split())) + 2 * len(calls)
            return [900, *range(1000, 1000 + width), 901]
        tag = {"system": 100, "user": 200, "tool": 300}[role]
        return [tag, 1, tag + 1]

    def apply_chat_template(
        self, messages, *, tools, tokenize, add_generation_prompt
    ):
        assert tokenize is True
        output = [10, len(tools), 11]
        for message in messages:
            output.extend(self._message_tokens(message))
        if add_generation_prompt:
            output.append(900)
        return output


def _git(root: Path, *arguments: str, env=None) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _clean_source_snapshot(tmp_path: Path) -> dict:
    """Commit the exact tested bytes so receipt cleanliness is not fabricated."""

    snapshot = tmp_path / "tested-source-snapshot"
    copied = {
        relative: REPO_ROOT / relative
        for relative in generation.V610_REQUIRED_RELEASE_SCRIPTS
    }
    copied.update(
        {
        "configs/v6_10_closure.yaml": CONFIG_PATH,
        "V6_10_CLOSURE_PREREGISTRATION.md": (
            REPO_ROOT / "V6_10_CLOSURE_PREREGISTRATION.md"
        ),
        "artifacts/v5_stage0/manifests/split_manifest.json": SPLIT_PATH,
        }
    )
    assert len(generation.V610_REQUIRED_RELEASE_SCRIPTS) == 14
    assert all(source.is_file() for source in copied.values())
    for relative, source in copied.items():
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _git(snapshot, "init", "-q")
    _git(snapshot, "add", ".")
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_DATE": "2026-07-30T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-07-30T00:00:00Z",
        }
    )
    _git(
        snapshot,
        "-c",
        "user.name=V6.10 regression",
        "-c",
        "user.email=v610-regression@example.invalid",
        "commit",
        "-q",
        "-m",
        "freeze exact tested bytes",
        env=environment,
    )
    provenance = static_preflight.repository_provenance(snapshot)
    assert provenance["tracked_worktree_clean"] is True
    return provenance


def _write_canonical_fixture(path: Path, payload) -> str:
    """Write deterministic fixture bytes and return their actual file hash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generation.canonical(payload), encoding="utf-8")
    return generation.sha256_file(path)


def _real_reference_receipt(tmp_path: Path):
    generation.configure_tau2(TAU2_ROOT)
    generation.register_agent()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    catalog, catalog_hashes = legacy_registry.load_task_catalog(TAU2_ROOT, split)
    task_payload = catalog["retail:35"]
    domain, task = generation.select_task("retail:35")
    criteria = task.evaluation_criteria
    renderer_inputs = reference.build_renderer_lint_inputs(
        task_identity="retail:35",
        communicate_info=criteria.communicate_info,
        nl_assertions=criteria.nl_assertions,
        fallback="The requested changes have been completed.",
    )
    rendered = generation.deterministic_completion_message(
        criteria,
        renderer="explicit_user_direct_v3",
        fallback="The requested changes have been completed.",
    )
    row = reference.preflight_task(
        task_identity="retail:35",
        actions=list(criteria.actions),
        allowed_identifier_keys=selection.SAFE_ON_ERROR_IDENTIFIER_KEYS["retail"],
        prefix=[],
        environment_factory=lambda: generation.initial_environment(
            "retail", task, []
        ),
        execute_call=generation.execute_call,
        state_hashes=generation.database_hashes,
        environment_evaluator=static_preflight._environment_evaluator(
            domain="retail", task=task, generation=generation
        ),
        renderer_lint_inputs=renderer_inputs,
        rendered_message=rendered,
    )
    assert row["status"] == "PASS"
    assert row["expected_error_indices"] == [0]
    assert row["sanitized_successful_reference_indices"] == [1, 2, 3, 4, 5, 6]
    assert row["eligible_forced_first_reference_indices"] == [1, 2, 3, 4, 5, 6]
    assert row["raw_reference_environment_reward"] == pytest.approx(1.0)
    assert row["sanitized_environment_reward"] == pytest.approx(1.0)

    source_provenance = _clean_source_snapshot(tmp_path)
    tau2_provenance = static_preflight.tau2_provenance(
        TAU2_ROOT, selection.TAU2_COMMIT
    )
    assert tau2_provenance["tracked_worktree_clean"] is True
    provenance = {
        "source": source_provenance,
        "tau2": tau2_provenance,
        "files": {
            "preflight_script_sha256": reference.sha256_file(
                Path(static_preflight.__file__)
            ),
            "contract_module_sha256": reference.sha256_file(
                Path(reference.__file__)
            ),
            "generation_adapter_sha256": reference.sha256_file(
                Path(generation.__file__)
            ),
            "selection_protocol_sha256": reference.sha256_file(
                Path(selection.__file__)
            ),
            "config_sha256": reference.sha256_file(CONFIG_PATH),
            "split_manifest_sha256": reference.sha256_file(SPLIT_PATH),
            "task_catalog_file_sha256": catalog_hashes,
        },
        "completion_renderer": "explicit_user_direct_v3",
        "source_scope": "ephemeral_clean_snapshot_of_exact_tested_bytes",
        "official_test_sealed": True,
        "official_test_identity_overlap_count": 0,
    }
    receipt = reference.build_preflight_receipt(
        expected_task_ids=["retail:35"],
        task_results=[row],
        provenance=provenance,
        task_source_hashes={
            "retail:35": static_preflight._task_source_hash(task_payload)
        },
    )
    receipt_path = tmp_path / "retail35-reference-preflight.json"
    reference.atomic_write_receipt(receipt_path, receipt)
    receipt_file_sha256 = generation.sha256_file(receipt_path)
    by_task = reference.verify_preflight_receipt(
        receipt,
        expected_task_ids=["retail:35"],
        expected_source_commit=source_provenance["commit"],
        expected_tau2_commit=selection.TAU2_COMMIT,
        expected_config_sha256=reference.sha256_file(CONFIG_PATH),
        expected_split_manifest_sha256=reference.sha256_file(SPLIT_PATH),
        require_pass=True,
    )
    assert by_task["retail:35"]["task_preflight_sha256"] == row[
        "task_preflight_sha256"
    ]
    return (
        task_payload,
        domain,
        task,
        row,
        receipt,
        receipt_file_sha256,
        source_provenance,
        tau2_provenance,
    )


def _task_level_registry(
    *,
    task_payload,
    preflight_row,
    receipt,
    receipt_file_sha256,
):
    """Project the real registry-v2 constructor onto one compatibility task."""

    actions = registry.executable_actions(
        task_identity="retail:35",
        task=task_payload,
        preflight_row=preflight_row,
    )
    options = registry._pair_options("retail:35", actions)
    phase = "compatibility"
    choice_set_id = "v610:compatibility:retail:35:prefix:initial"
    prefix = {
        "prefix_slot_id": f"{choice_set_id}:prefix-slot",
        "source": "single_user_turn_before_first_assistant_tool_action",
        "reference_action_prefix_count": 0,
        "clean_future_available_to_causal_continuation": False,
        "shared_across_candidate_pairs": True,
        "materialization_seed": registry._seed(
            registry.REGISTRY_PROTOCOL, phase, "retail:35", "prefix"
        ),
    }
    snapshot = {
        "snapshot_slot_id": f"{choice_set_id}:snapshot-slot",
        "source": "fresh_pinned_tau2_task_initial_state",
        "agent_and_user_database_hashes_required": True,
        "shared_across_candidate_pairs": True,
    }
    prefix_hash = registry.sha256(prefix)
    snapshot_hash = registry.sha256(snapshot)
    pairs = []
    for pair_index in range(closure.PAIRS_PER_TASK):
        selected = options[pair_index % len(options)]
        candidate_pair_id = (
            f"{choice_set_id}:candidate-pair:{pair_index + 1:02d}"
        )
        branches = [
            registry._branch(
                phase=phase,
                task_identity="retail:35",
                choice_set_id=choice_set_id,
                candidate_pair_id=candidate_pair_id,
                pair_index=pair_index,
                branch_index=branch_index,
                action=action,
                prefix_sha256=prefix_hash,
                snapshot_sha256=snapshot_hash,
                reference_preflight_receipt_sha256=receipt["receipt_sha256"],
                reference_preflight_file_sha256=receipt_file_sha256,
            )
            for branch_index, action in enumerate(selected)
        ]
        pair = {
            "candidate_pair_id": candidate_pair_id,
            "choice_set_id": choice_set_id,
            "phase": phase,
            "partition": "arm_train",
            "task_identity": "retail:35",
            "domain": "retail",
            "task_id": "35",
            "pair_index": pair_index,
            "selection_unit": True,
            "prefix_slot": deepcopy(prefix),
            "snapshot_slot": deepcopy(snapshot),
            "prefix_sha256": prefix_hash,
            "environment_snapshot_sha256": snapshot_hash,
            "choice_seed": registry._seed(
                registry.REGISTRY_PROTOCOL,
                registry.PHASE_SEED_SALTS[phase],
                candidate_pair_id,
            ),
            "branches": branches,
            "expected_canonical_corrective_families": [
                branch["canonical_corrective_family"] for branch in branches
            ],
            "reference_preflight_receipt_sha256": receipt["receipt_sha256"],
            "reference_preflight_file_sha256": receipt_file_sha256,
            "reference_task_preflight_sha256": preflight_row[
                "task_preflight_sha256"
            ],
            "sanitized_reference_plan_sha256": preflight_row[
                "sanitized_successful_plan_sha256"
            ],
            "status": "REGISTERED_INJECTION_PREFLIGHT_REQUIRED",
            "official_test_used": False,
        }
        pair["candidate_pair_sha256"] = registry.sha256(pair)
        pairs.append(pair)
    root = {
        "protocol": closure.REGISTRY_PROTOCOL,
        "design_protocol": closure.PROTOCOL,
        "scope": "single_task_engineering_regression_not_formal_gate",
        "reference_preflight_receipt_sha256": receipt["receipt_sha256"],
        "reference_preflight_file_sha256": receipt_file_sha256,
        "candidate_pairs": pairs,
        "official_test_used": False,
    }
    root["registry_sha256"] = generation.sha256(root)
    return root, pairs


def _generation_args():
    return SimpleNamespace(
        phase="compatibility",
        shard_index=0,
        num_shards=1,
        smoke_task=None,
        teacher_model=generation.V610_TEACHER_MODEL,
        teacher_revision=generation.V610_TEACHER_REVISION,
        teacher_api_base="http://teacher.invalid/v1",
        user_model=generation.V610_USER_JUDGE_MODEL,
        user_revision=generation.V610_USER_JUDGE_REVISION,
        user_api_base="http://user.invalid/v1",
        judge_model=generation.V610_USER_JUDGE_MODEL,
        judge_revision=generation.V610_USER_JUDGE_REVISION,
        judge_api_base="http://user.invalid/v1",
        max_tokens=512,
        max_steps=60,
        timeout=30.0,
        clean_attempts=1,
        recovery_attempts=1,
        clean_agent_mode="single_turn_user_reference_replay",
        matched_positive_continuation_mode="deterministic_reference_completion",
        causal_cell_continuation_mode="fresh_teacher",
        first_action_measurement_mode="teacher_unforced",
        completion_renderer="explicit_user_direct_v3",
        recovery_continuation_mode="fresh_teacher",
    )


def _failed_checks(audited):
    result = {}
    for section, checks in audited.get("audit_checks", {}).items():
        if isinstance(checks, dict):
            if checks and all(isinstance(value, bool) for value in checks.values()):
                failed = [key for key, value in checks.items() if value is not True]
                if failed:
                    result[section] = failed
            else:
                nested = {
                    key: [name for name, value in value.items() if value is not True]
                    for key, value in checks.items()
                    if isinstance(value, dict)
                    and any(item is not True for item in value.values())
                }
                if nested:
                    result[section] = nested
    return result


def test_retail35_real_environment_regression_reaches_two_sft_rows(tmp_path):
    (
        task_payload,
        domain,
        task,
        preflight_row,
        receipt,
        receipt_file_sha256,
        source_provenance,
        tau2_provenance,
    ) = _real_reference_receipt(tmp_path)
    registry_root, registered_pairs = _task_level_registry(
        task_payload=task_payload,
        preflight_row=preflight_row,
        receipt=receipt,
        receipt_file_sha256=receipt_file_sha256,
    )
    plan_hashes = {
        "retail:35": preflight_row["sanitized_successful_plan_sha256"]
    }
    task_hashes = {"retail:35": preflight_row["task_preflight_sha256"]}
    preflight_binding = {
        "protocol": receipt["protocol"],
        "artifact_type": receipt["artifact_type"],
        "file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "status": "PASS",
        "design_protocol": closure.PROTOCOL,
        "tau2_commit": selection.TAU2_COMMIT,
        "source_commit": source_provenance["commit"],
        "ordered_task_ids_sha256": receipt["ordered_task_ids_sha256"],
        "sanitized_plan_hashes_sha256": generation.sha256(plan_hashes),
        "task_preflight_hashes_sha256": generation.sha256(task_hashes),
        "official_test_used": False,
    }
    reference_plan = generation.verified_reference_plan_for_task(
        receipt, "retail:35", preflight_binding=preflight_binding
    )

    # This gate executes all six registered mutations in the real tau2 env.
    outcomes, injected_report = generation.preflight_selected_phase_tasks(
        registered_pairs_by_task={"retail:35": registered_pairs},
        reference_preflight=receipt,
        preflight_binding=preflight_binding,
        phase="compatibility",
    )
    assert injected_report["passing_task_count"] == 1
    assert injected_report["rejected_task_count"] == 0
    assert outcomes["retail:35"]["status"] == "PASS"
    injected = outcomes["retail:35"]["branch_evidence"]
    assert len(injected) == 6
    assert all(row["actual_tool_error"] is True for row in injected)
    assert all(row["state_unchanged_after_error"] is True for row in injected)

    args = _generation_args()
    runtime_reference = {
        "protocol": reference.RECEIPT_PROTOCOL,
        "design_protocol": closure.PROTOCOL,
        "receipt_sha256": receipt["receipt_sha256"],
        "file_sha256": receipt_file_sha256,
        "status": "PASS",
        "tau2_commit": selection.TAU2_COMMIT,
        "source_commit": source_provenance["commit"],
        "official_test_used": False,
    }
    created_at_utc = "2026-07-30T00:00:00Z"
    container_fixture = {
        "scope": "deterministic_cpu_service_boundary_test_fixture",
        "actual_container_launched": False,
    }
    container_image_digest = "sha256:" + generation.sha256(
        container_fixture
    )
    registry_file_sha256 = _write_canonical_fixture(
        tmp_path / "candidate-registry.json",
        registry_root,
    )
    source_container_fixture = {
        "artifact_type": "deterministic_test_fixture",
        "source_commit": source_provenance["commit"],
        "source_tree": source_provenance["git_tree_object"],
        "tau2_commit": tau2_provenance["commit"],
        "tau2_tree": tau2_provenance["git_tree_object"],
        "container_image_digest": container_image_digest,
        "actual_container_launched": False,
        "official_test_used": False,
    }
    source_container_fixture_sha256 = _write_canonical_fixture(
        tmp_path / "source_container_provenance.json",
        source_container_fixture,
    )
    model_server_fixture = {
        "artifact_type": "deterministic_test_fixture",
        "container_image_digest": container_image_digest,
        "teacher_user_judge_boundary_faked": True,
        "roles": {
            "teacher": {
                "model": args.teacher_model,
                "revision": args.teacher_revision,
            },
            "user": {
                "model": args.user_model,
                "revision": args.user_revision,
            },
            "judge": {
                "model": args.judge_model,
                "revision": args.judge_revision,
            },
        },
        "official_test_used": False,
    }
    model_server_fixture_sha256 = _write_canonical_fixture(
        tmp_path / "model_server_receipts.json",
        model_server_fixture,
    )
    runtime_receipt_hashes = {
        "protocol": "v6_10_runtime_receipt_hashes_v1",
        "design_protocol": closure.PROTOCOL,
        "status": "TEST_FIXTURE",
        "source_container_provenance": {
            "file_sha256": source_container_fixture_sha256,
            "semantic_sha256": generation.sha256(source_container_fixture),
        },
        "model_server_receipts": {
            "file_sha256": model_server_fixture_sha256,
            "semantic_sha256": generation.sha256(model_server_fixture),
        },
        "created_at_utc": created_at_utc,
        "official_test_used": False,
    }
    runtime_receipt_hashes_sha256 = _write_canonical_fixture(
        tmp_path / "runtime_receipt_hashes.json",
        runtime_receipt_hashes,
    )
    parent_artifact_hashes = {
        "config_sha256": generation.sha256_file(CONFIG_PATH),
        "preregistration_sha256": generation.sha256_file(
            REPO_ROOT / "V6_10_CLOSURE_PREREGISTRATION.md"
        ),
        "split_manifest_sha256": generation.sha256_file(SPLIT_PATH),
        "reference_preflight_receipt_sha256": receipt_file_sha256,
        "registry_file_sha256": registry_file_sha256,
        "source_container_provenance_sha256": (
            source_container_fixture_sha256
        ),
        "model_server_receipts_sha256": model_server_fixture_sha256,
        "runtime_receipt_hashes_sha256": runtime_receipt_hashes_sha256,
    }
    assert set(parent_artifact_hashes) == (
        generation.V610_PARENT_ARTIFACT_HASH_KEYS
        - {"release_manifest_sha256"}
    )
    release_identities = {
        "source_commit": source_provenance["commit"],
        "source_tree": source_provenance["git_tree_object"],
        "container_image_digest": container_image_digest,
        "tau2_commit": tau2_provenance["commit"],
        **parent_artifact_hashes,
    }
    assert len(release_identities) == 12
    relevant_scripts = {
        relative: generation.sha256_file(REPO_ROOT / relative)
        for relative in sorted(generation.V610_REQUIRED_RELEASE_SCRIPTS)
    }
    assert len(relevant_scripts) == 14
    release_payload = {
        "protocol": generation.V610_RELEASE_MANIFEST_PROTOCOL,
        "design_protocol": closure.PROTOCOL,
        "status": "PASS",
        **release_identities,
        "relevant_scripts": relevant_scripts,
        "created_at_utc": created_at_utc,
        "official_test_used": False,
    }
    release_manifest_file_sha256 = _write_canonical_fixture(
        tmp_path / "release_manifest.json",
        release_payload,
    )
    parent_artifact_hashes["release_manifest_sha256"] = (
        release_manifest_file_sha256
    )
    release_manifest_binding = {
        "protocol": generation.V610_RELEASE_MANIFEST_PROTOCOL,
        "file_sha256": release_manifest_file_sha256,
        "semantic_sha256": generation.sha256(release_payload),
        "created_at_utc": created_at_utc,
        "identities": release_identities,
        "relevant_scripts": relevant_scripts,
        "relevant_scripts_sha256": generation.sha256(relevant_scripts),
        "official_test_used": False,
    }
    runtime_provenance = {
        "scope": "single_task_engineering_regression_not_formal_release",
        "attempt_id": "v6_10-compatibility-test-fixture",
        "source_commit": source_provenance["commit"],
        "source_tree": source_provenance["git_tree_object"],
        "source_tree_clean": True,
        "expected_source_commit": source_provenance["commit"],
        "observed_source_commit": source_provenance["commit"],
        "source_tracked_worktree_clean": True,
        "container_image_digest": container_image_digest,
        "tau2_commit": tau2_provenance["commit"],
        "tau2_tree": tau2_provenance["git_tree_object"],
        "tau2_tree_clean": True,
        "expected_tau2_commit": tau2_provenance["commit"],
        "observed_tau2_commit": tau2_provenance["commit"],
        "tau2_tracked_worktree_clean": True,
        "generation_script_sha256": relevant_scripts[
            "scripts/run_v6_candidate_generation.py"
        ],
        "reference_preflight": runtime_reference,
        "parent_artifact_hashes": parent_artifact_hashes,
        "task_file_sha256": {
            domain_name: reference.sha256_file(
                TAU2_ROOT
                / "data"
                / "tau2"
                / "domains"
                / domain_name
                / "tasks.json"
            )
            for domain_name in ("airline", "retail")
        },
        "release_manifest": release_manifest_binding,
        "release_manifest_sha256": release_manifest_file_sha256,
        "official_test_used": False,
    }
    execution_provenance = generation.build_execution_provenance(
        run_started_at_utc="2026-07-30T00:00:00Z",
        exact_argv=(
            str(Path(os.sys.executable).resolve()),
            "-m",
            "pytest",
            "tests/test_v6_10_retail35_end_to_end.py",
        ),
    )
    semantic_contract = generation.semantic_generation_contract(
        args,
        continuation_seeds=generation.V610_COMPATIBILITY_SEEDS,
        design_protocol=closure.PROTOCOL,
        runtime_provenance=runtime_provenance,
        execution_provenance=execution_provenance,
    )

    from tau2.data_model.message import AssistantMessage

    def fake_user_generate(*args, **kwargs):
        assert kwargs.get("call_name") in (None, "user_simulator_response")
        messages = kwargs.get("messages") or []
        last = messages[-1] if messages else {}
        last_content = (
            last.get("content")
            if isinstance(last, dict)
            else getattr(last, "content", None)
        )
        return AssistantMessage(
            role="assistant",
            content=(
                "###STOP###"
                if isinstance(last_content, str)
                and "###STOP###" in last_content
                else "Please complete both requested changes."
            ),
        )

    def fake_teacher_generate(*args, **kwargs):
        assert kwargs.get("call_name") in (None, "agent_response")
        return AssistantMessage(role="assistant", content="###STOP###")

    with (
        patch(
            "tau2.user.user_simulator.generate",
            side_effect=fake_user_generate,
        ),
        patch(
            "tau2.agent.llm_agent.generate",
            side_effect=fake_teacher_generate,
        ),
    ):
        clean = generation.clean_rollout(
            args,
            task=task,
            domain=domain,
            task_identity="retail:35",
            seed=int(registered_pairs[0]["choice_seed"]),
            log_root=tmp_path / "clean",
            reference_plan=reference_plan,
        )
        assert clean["official_reward"] == pytest.approx(1.0)
        assert clean["clean_replay"]["pass"] is True
        assert not any(
            message.get("role") == "tool" and message.get("error") is True
            for message in clean["clean_messages"]
        )
        raw_pairs = [
            generation.materialize_pair(
                args,
                registered_pair=registered_pair,
                clean=clean,
                task=task,
                domain=domain,
                continuation_seeds=generation.V610_COMPATIBILITY_SEEDS,
                log_root=tmp_path
                / registered_pair["candidate_pair_id"].replace(":", "_"),
                registry_sha256=registry_root["registry_sha256"],
                semantic_contract=semantic_contract,
                reference_plan=reference_plan,
            )
            for registered_pair in registered_pairs
        ]

    tokenizer = PrefixStableFakeTokenizer()

    def deterministic_scorer(prompt, action, tools):
        digest = int(
            measurement.canonical_sha256([prompt, action, tools])[:8], 16
        )
        count = 1 + digest % 4
        return {
            "sum_logprob": -float(count) * (1.0 + (digest % 7) / 10.0),
            "token_count": count,
        }

    model_provenance = {
        "model_name": audit.FROZEN_STUDENT_MODEL,
        "model_revision": audit.FROZEN_STUDENT_REVISION,
        "tokenizer_name": audit.FROZEN_STUDENT_MODEL,
        "tokenizer_revision": audit.FROZEN_STUDENT_REVISION,
        "load_in_4bit": True,
        "frozen_checkpoint_identity_sha256": measurement.canonical_sha256(
            ["deterministic-cpu-service-boundary", audit.FROZEN_STUDENT_REVISION]
        ),
    }
    source_pool_sha256 = measurement.canonical_sha256(raw_pairs)
    measured = [
        measurement.enrich_candidate(
            raw,
            tokenizer=tokenizer,
            scorer=deterministic_scorer,
            model_provenance=model_provenance,
            tau2_provenance=tau2_provenance,
            source_pool_sha256=source_pool_sha256,
        )
        for raw in raw_pairs
    ]
    assert all(
        row["token_accounting"]["status"] == "MEASURED_FROZEN_STUDENT"
        for row in measured
    )

    audited = [
        audit.audit_candidate_pair(
            row,
            registered_pair,
            registry_sha256=registry_root["registry_sha256"],
            registry_root=registry_root,
        )
        for row, registered_pair in zip(measured, registered_pairs)
    ]
    assert all(row["audit_status"] == "ACCEPTED" for row in audited), [
        {
            "candidate_pair_id": row["candidate_pair_id"],
            "failed_checks": _failed_checks(row),
        }
        for row in audited
    ]
    for pair in audited:
        for branch in pair["branches"]:
            provenance = branch["recovery_suffix_provenance"]
            assert provenance["origin"] == "sanitized_deterministic_reference_plan"
            assert provenance["fresh_recovery_generated"] is False
            assert provenance["gold_suffix_used"] is True
            assert (
                provenance["reference_preflight_receipt_sha256"]
                == receipt["receipt_sha256"]
            )
            assert (
                provenance["reference_task_preflight_sha256"]
                == preflight_row["task_preflight_sha256"]
            )
            assert (
                provenance["sanitized_reference_plan_sha256"]
                == preflight_row["sanitized_successful_plan_sha256"]
            )

    scored, scoring_audit = scoring.score_candidates(audited)
    assert scoring_audit["status"] == "PASS"
    manifests, selection_audit = selectors.build_manifests(
        scored, selectors=["full_proposed"]
    )
    assert selection_audit["status"] == "PASS"
    manifest = manifests["full_proposed"]
    assert manifest["matched_task_ids"] == ["retail:35"]
    assert len(manifest["selected"]) == 1

    def deterministic_contract_builder(messages, mask, schemas):
        contract = measurement.tokenize_labeled_messages(
            tokenizer,
            messages,
            mask,
            schemas,
            label="retail35.materialization",
        )
        return {
            key: contract[key]
            for key in ("sequence_tokens", "supervised_tokens", "label_spans")
        }

    rows, materialization_audit = materialize.materialize_manifest(
        manifest,
        tokenizer_name=audit.FROZEN_STUDENT_MODEL,
        tokenizer_revision=audit.FROZEN_STUDENT_REVISION,
        contract_builder=deterministic_contract_builder,
        contract_builder_name=(
            "tests.retail35_prefix_stable_deterministic_token_contract_v1"
        ),
    )
    assert materialization_audit["status"] == "PASS"
    assert len(rows) == materialization_audit["expected_rows"] == 2
    assert {row["metadata"]["task_identity"] for row in rows} == {"retail:35"}
    assert len({row["metadata"]["candidate_pair_id"] for row in rows}) == 1
    for row in rows:
        messages = row["messages"]
        mask = row["label_mask"]
        failed = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
            and message.get("tool_calls")
            and index + 1 < len(messages)
            and messages[index + 1].get("role") == "tool"
            and messages[index + 1].get("error") is True
        ]
        assert len(failed) == 1
        assert mask[failed[0]] is False
        assert mask[failed[0] + 1] is False
        assert row["metadata"]["failed_action_label_messages"] == 0
        assert (
            row["metadata"]["recovery_suffix_origin"]
            == "sanitized_deterministic_reference_plan"
        )
        assert row["metadata"]["fresh_recovery_suffix"] is False
        assert row["metadata"]["gold_reference_suffix_used"] is True
