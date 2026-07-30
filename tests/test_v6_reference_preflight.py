from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import preflight_v6_reference_traces as preflight
from scripts import v6_reference_contract as contract


class FakeEnvironment:
    def __init__(self, *, mutate_on_error: bool = False):
        self.data = {"xx": 0, "yy": 0}
        self.mutate_on_error = mutate_on_error


def state_hashes(environment):
    return {"agent_db_hash": contract.sha256(environment.data), "user_db_hash": None}


def execute_call(environment, call):
    name = call["name"]
    arguments = call["arguments"]
    error = False
    content = "ok"
    if name == "lookup":
        key = arguments["id"]
        if key == "missing" or key not in environment.data:
            error = True
            content = "not found"
            if environment.mutate_on_error:
                environment.data["corruption"] = 1
        else:
            content = str(environment.data[key])
    elif name == "set_value":
        environment.data[arguments["id"]] = arguments["value"]
    elif name == "require_value":
        if environment.data.get(arguments["id"]) != arguments["value"]:
            error = True
            content = "dependency missing"
    else:
        raise AssertionError(name)
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [deepcopy(dict(call))],
    }
    result = {
        "role": "tool",
        "tool_call_id": call["id"],
        "content": content,
        "error": error,
    }
    return assistant, result


def environment_reward_one(messages):
    return {
        "reward": 1.0,
        "reward_info": {"basis": "environment", "n": len(messages)},
    }


def actions():
    return [
        {
            "action_id": "a0",
            "requestor": "assistant",
            "name": "lookup",
            "arguments": {"id": "missing"},
        },
        {
            "action_id": "a1",
            "requestor": "assistant",
            "name": "set_value",
            "arguments": {"id": "xx", "value": 1},
        },
        {
            "action_id": "a2",
            "requestor": "assistant",
            "name": "set_value",
            "arguments": {"id": "yy", "value": 1},
        },
        {
            "action_id": "a3",
            "requestor": "assistant",
            "name": "require_value",
            "arguments": {"id": "xx", "value": 1},
        },
    ]


def renderer():
    lint = contract.build_renderer_lint_inputs(
        task_identity="retail:1",
        communicate_info=["the refund is $54.04"],
        nl_assertions=["Agent should tell the user that order #A123 is complete"],
        fallback="done",
    )
    message = {
        "role": "assistant",
        "content": (
            "the refund is $54.04\n"
            "I am telling you directly: order #A123 is complete"
        ),
    }
    return lint, message


def pass_task_row(*, reward_evaluator=environment_reward_one):
    lint, message = renderer()
    return contract.preflight_task(
        task_identity="retail:1",
        actions=actions(),
        allowed_identifier_keys={"id"},
        prefix=[],
        environment_factory=FakeEnvironment,
        execute_call=execute_call,
        state_hashes=state_hashes,
        environment_evaluator=reward_evaluator,
        renderer_lint_inputs=lint,
        rendered_message=message,
    )


def provenance():
    commit = "a" * 40
    digest = "b" * 64
    return {
        "source": {
            "commit": commit,
            "tree": digest,
            "tracked_worktree_clean": True,
        },
        "tau2": {
            "commit": commit,
            "tree": digest,
            "tracked_worktree_clean": True,
        },
        "files": {
            "preflight_script_sha256": digest,
            "contract_module_sha256": digest,
            "config_sha256": digest,
            "split_manifest_sha256": digest,
        },
    }


def test_preflight_provenance_does_not_ignore_untracked_files(monkeypatch):
    calls = []

    def fake_git(root, *arguments, allow_failure=False):
        del root, allow_failure
        calls.append(arguments)
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if arguments[:3] == ("status", "--porcelain=v1", "--untracked-files=all"):
            return "?? untracked-evidence"
        if arguments == ("ls-tree", "-r", "--full-tree", "HEAD"):
            return "100644 blob deadbeef\ttracked"
        raise AssertionError(arguments)

    monkeypatch.setattr(preflight, "_git", fake_git)
    observed = preflight.repository_provenance(Path("/frozen/source"))

    assert observed["tracked_worktree_clean"] is False
    assert observed["worktree_scope"] == "tracked_and_untracked_files"
    assert (
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ) in calls


def test_exact_index_identity_survives_semantic_duplicates():
    repeated = actions()
    repeated.append(deepcopy(repeated[2]))
    slots = contract.build_reference_slots("retail:1", repeated)
    assert slots[2]["call_semantics_sha256"] == slots[4]["call_semantics_sha256"]
    assert slots[2]["reference_slot_id"] != slots[4]["reference_slot_id"]
    assert slots[2]["semantically_identical_reference_indices"] == [2, 4]
    assert slots[4]["semantic_occurrence_ordinal"] == 1


def test_raw_expected_error_is_removed_and_environment_rewards_are_one():
    row = pass_task_row()
    assert row["status"] == "PASS"
    assert row["expected_error_indices"] == [0]
    assert row["sanitized_reference_indices"] == [1, 2, 3]
    assert row["raw_reference_environment_reward"] == 1.0
    assert row["raw_reference_repeat_environment_reward"] == 1.0
    assert row["sanitized_environment_reward"] == 1.0
    assert (
        row["full_official_reward_status"]
        == "DEFERRED_TO_DYNAMIC_JUDGE"
    )
    assert "raw_reference_official_reward" not in row
    assert "sanitized_official_reward" not in row
    assert row["reference_outcome_contract"][
        "sanitized_plan_all_actions_succeeded"
    ]


def test_environment_reward_gate_rejects_sanitized_plan():
    def evaluator(messages):
        tool_calls = sum(
            message.get("role") == "assistant" and bool(message.get("tool_calls"))
            for message in messages
        )
        # Raw has four calls; sanitized and forced-first plans have three.
        return {"reward": 1.0 if tool_calls == 4 else 0.0, "reward_info": {}}

    row = pass_task_row(reward_evaluator=evaluator)
    assert row["status"] == "REJECTED"
    assert row["raw_reference_environment_reward"] == 1.0
    assert row["sanitized_environment_reward"] == 0.0
    assert "RAW_OR_SANITIZED_REFERENCE_REJECTED" in row["reason_codes"]
    assert row["eligible_forced_first_reference_indices"] == []


def test_failed_action_that_mutates_state_fails_closed():
    slots = contract.build_reference_slots("retail:1", actions())
    _, message = renderer()
    with pytest.raises(contract.ReferenceContractError, match="changed state"):
        contract.derive_reference_outcome_contract(
            task_identity="retail:1",
            slots=slots,
            prefix=[],
            rendered_completion=message,
            environment_factory=lambda: FakeEnvironment(mutate_on_error=True),
            execute_call=execute_call,
            state_hashes=state_hashes,
            environment_evaluator=environment_reward_one,
        )


def test_forced_first_screen_excludes_error_and_dependency_slots():
    row = pass_task_row()
    assert row["eligible_forced_first_reference_indices"] == [1, 2]
    reasons = {
        value["reference_action_index"]: value["reason_code"]
        for value in row["forced_first_slot_screen"]["rows"]
    }
    assert reasons[0] == "EXPECTED_ERROR_REFERENCE_ACTION"
    assert reasons[3] == "FORCED_FIRST_PLAN_TOOL_ERROR"
    assert all(
        value["forced_first_environment_reward"] == 1.0
        for value in row["forced_first_slot_screen"]["rows"]
        if value["eligible"]
    )


def test_renderer_lint_preserves_literals_and_rejects_meta_artifacts():
    lint, good = renderer()
    assert contract.lint_rendered_completion(lint, good)["status"] == "PASS"
    bad = {"role": "assistant", "content": "Agent should directly: that it passed"}
    assert contract.lint_rendered_completion(lint, bad)["status"] == "REJECTED"


def test_receipt_verifier_rejects_stale_bindings_and_tampering(tmp_path):
    row = pass_task_row()
    receipt = contract.build_preflight_receipt(
        expected_task_ids=["retail:1"],
        task_results=[row],
        provenance=provenance(),
        task_source_hashes={"retail:1": "b" * 64},
    )
    mapping = contract.verify_preflight_receipt(
        receipt,
        expected_task_ids=["retail:1"],
        expected_config_sha256="b" * 64,
    )
    assert mapping["retail:1"]["sanitized_reference_indices"] == [1, 2, 3]
    with pytest.raises(contract.ReferenceContractError, match="stale"):
        contract.verify_preflight_receipt(
            receipt,
            expected_config_sha256="c" * 64,
        )
    damaged = deepcopy(receipt)
    damaged["task_results"][0]["status"] = "REJECTED"
    with pytest.raises(contract.ReferenceContractError, match="hash drift"):
        contract.verify_preflight_receipt(damaged)

    output = tmp_path / "reference_preflight_receipt.json"
    contract.atomic_write_receipt(output, receipt)
    loaded = json.loads(output.read_text())
    contract.verify_preflight_receipt(loaded)
    with pytest.raises(contract.ReferenceContractError, match="overwrite"):
        contract.atomic_write_receipt(output, receipt)


def test_unclassified_task_error_makes_receipt_no_go():
    receipt = contract.build_preflight_receipt(
        expected_task_ids=["retail:1"],
        task_results=[
            contract.task_error_row("retail:1", RuntimeError("adapter broke"))
        ],
        provenance=provenance(),
        task_source_hashes={"retail:1": "b" * 64},
    )
    assert receipt["status"] == "NO_GO"
    with pytest.raises(contract.ReferenceContractError, match="not PASS"):
        contract.verify_preflight_receipt(receipt)
