from __future__ import annotations

from copy import deepcopy

from scripts import v5_5_protocol as protocol


def test_constants_and_mutations_are_frozen():
    protocol.validate_constants()
    assert protocol.mutate_identifier("#W123", 0) == "#W124"
    assert protocol.mutate_identifier("#W123", 1) == "#W143"
    assert protocol.mutate_identifier("ABCZ", 0) == "ABCA"
    assert protocol.semantic_sha256(
        [{"role": "tool", "content": "x", "timestamp": "first"}]
    ) == protocol.semantic_sha256(
        [{"role": "tool", "content": "x", "timestamp": "second"}]
    )


def test_reference_action_filter_rejects_names_and_mutations():
    actions = [
        {
            "requestor": "assistant",
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "Ada", "last_name": "Lovelace", "zip": "1"},
        },
        {
            "requestor": "assistant",
            "name": "get_order_details",
            "arguments": {"order_id": "#W123"},
        },
        {
            "requestor": "assistant",
            "name": "cancel_pending_order",
            "arguments": {"order_id": "#W123"},
        },
    ]
    assert protocol.eligible_reference_actions("retail", actions) == [
        {
            "action_index": 1,
            "tool_name": "get_order_details",
            "identifier_key": "order_id",
            "correct_identifier": "#W123",
        }
    ]


def valid_pair(task: int, variant: int, domain: str = "retail"):
    correct = {
        "id": "correct",
        "name": "get_order_details",
        "arguments": {"order_id": f"#W{task:03d}"},
        "requestor": "assistant",
    }
    injected = deepcopy(correct)
    injected["id"] = "error"
    injected["arguments"]["order_id"] += str(variant)
    failed = [
        {"role": "assistant", "content": None, "tool_calls": [injected]},
        {"role": "tool", "id": "error", "content": "Error: not found", "error": True},
    ]
    prefix = [{"role": "user", "content": f"task {task}"}]
    supervised = [
        {"role": "assistant", "content": None, "tool_calls": [correct]},
        {"role": "tool", "id": "correct", "content": "{}", "error": False},
    ]
    return {
        "pair_id": f"{domain}:{task}:counterfactual:{variant}",
        "task_identity": f"{domain}:{task}",
        "domain": domain,
        "identifier_key": "order_id",
        "clean_call": correct,
        "injected_call": injected,
        "injected_result": failed[1],
        "correction_call": correct,
        "correction_result": supervised[1],
        "failed_event": failed,
        "supervised_messages": supervised,
        "supervision_starts_after_error": True,
        "clean_future_present_in_recovery_prompt": False,
        "clean_prefix_sha256": protocol.sha256(prefix),
        "recovery_prefix_sha256": protocol.sha256(prefix),
        "error_event_sha256": protocol.sha256(failed),
        "clean_end_state_matches_reference": True,
        "recovery_end_state_matches_reference": True,
        "clean_agent_db_hash": f"agent-{task}",
        "clean_user_db_hash": None,
        "recovery_agent_db_hash": f"agent-{task}",
        "recovery_user_db_hash": None,
        "independent_environment_replay_pass": True,
        "official_test_used": False,
    }


def test_independent_pair_audit_detects_failed_call_supervision():
    pair = valid_pair(1, 1)
    assert all(protocol.audit_pair(pair).values())
    pair["supervised_messages"] = pair["failed_event"] + pair["supervised_messages"]
    assert protocol.audit_pair(pair)["failed_event_not_supervised"] is False


def test_full_gate_requires_coverage_and_distinct_events():
    pairs = []
    registered = []
    for task in range(24):
        domain = "airline" if task < 6 else "retail"
        identity = f"{domain}:{task}"
        registered.append(identity)
        for variant in (1, 2):
            pairs.append(valid_pair(task, variant, domain))
    result = protocol.gate(pairs, registered)
    assert result["status"] == "PASS_TRAINING_AUTHORIZED"
    pairs[0]["error_event_sha256"] = pairs[1]["error_event_sha256"]
    result = protocol.gate(pairs, registered)
    assert result["status"] == "FAIL_CLOSED"
    assert result["checks"]["distinct_error_events_within_task"] is False
