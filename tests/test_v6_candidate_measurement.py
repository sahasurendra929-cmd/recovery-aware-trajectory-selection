from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import measure_v6_candidate_tokens as measurement


class FakeChatTokenizer:
    """Small prefix-stable chat-template tokenizer for pure contract tests."""

    @staticmethod
    def _message_tokens(message):
        role = message["role"]
        if role == "assistant":
            body = str(message.get("content", ""))
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


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "repair",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
        },
    },
]


def call(name, identifier):
    return {
        "id": f"{name}-{identifier}",
        "name": name,
        "arguments": {"id": identifier},
        "requestor": "assistant",
    }


def test_failure_context_is_masked_and_full_text_suffix_is_labeled():
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "help"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("lookup", "bad")],
        },
        {
            "role": "tool",
            "content": "not found",
            "tool_call_id": "lookup-bad",
            "error": True,
        },
        {"role": "assistant", "content": "I will recover."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("repair", "good")],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "repair-good",
            "error": False,
        },
        {"role": "assistant", "content": "The issue is fixed."},
    ]
    mask = measurement.assistant_label_mask(messages, assistant_start=4)
    assert mask == [False, False, False, False, True, True, False, True]
    contract = measurement.tokenize_labeled_messages(
        FakeChatTokenizer(), messages, mask, TOOLS, label="recovery"
    )
    assert [span["message_index"] for span in contract["label_spans"]] == [
        4,
        5,
        7,
    ]
    assert contract["supervised_tokens"] > 0
    assert contract["sequence_tokens"] > contract["supervised_tokens"]


def test_failed_tool_call_inside_suffix_is_also_never_a_label():
    messages = [
        {"role": "system", "content": "policy"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("lookup", "bad")],
        },
        {
            "role": "tool",
            "content": "error",
            "tool_call_id": "lookup-bad",
            "error": True,
        },
        {"role": "assistant", "content": "try again"},
    ]
    assert measurement.assistant_label_mask(
        messages, assistant_start=1
    ) == [False, False, False, True]


def test_length_normalization_uses_mean_token_log_probability():
    first = measurement.make_logprob_cell(-6.0, 3)
    second = measurement.make_logprob_cell(-4.0, 2)
    assert first["mean_logprob"] == pytest.approx(-2.0)
    assert second["mean_logprob"] == pytest.approx(-2.0)
    assert first["length_normalized"] is True


def raw_branch(index):
    action = call("lookup" if index == 0 else "repair", f"right-{index}")
    return {
        "branch_id": f"branch-{index}",
        "recovery_prompt": [
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call("lookup", f"bad-{index}")],
            },
            {
                "role": "tool",
                "content": "error",
                "tool_call_id": f"lookup-bad-{index}",
                "error": True,
            },
        ],
        "producer_evidence": {
            "registered_branch": {
                "corrective_action_spec": {
                    "forced_first_action_constructor": {
                        "tool_call": action
                    }
                }
            }
        },
    }


def test_first_action_grid_has_exactly_four_matched_and_crossed_cells():
    seen = []

    def scorer(prompt, action, tools):
        seen.append((prompt[-1]["content"], action["name"], len(tools)))
        count = 2 if action["name"] == "lookup" else 4
        return {"sum_logprob": -float(count), "token_count": count}

    cells = measurement.four_cell_first_action_logprobs(
        [raw_branch(0), raw_branch(1)], TOOLS, scorer=scorer
    )
    assert tuple(cells) == measurement.Q_KEYS
    assert len(seen) == 4
    assert all(cell["mean_logprob"] == pytest.approx(-1.0) for cell in cells.values())
    assert cells["q_e1_a1"]["corrective_action_sha256"] != cells[
        "q_e1_a2"
    ]["corrective_action_sha256"]
    assert cells["q_e1_a1"]["error_prompt_sha256"] == cells[
        "q_e1_a2"
    ]["error_prompt_sha256"]


def test_revision_arguments_are_mandatory():
    with pytest.raises(SystemExit):
        measurement.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
    with pytest.raises(
        measurement.V6CandidateMeasurementError, match="model_revision"
    ):
        measurement.require_revision(" ", "model_revision")


def test_enrichment_preserves_raw_evidence_and_sums_branch_costs():
    branches = []
    for index in range(2):
        branch = raw_branch(index)
        suffix = [
            {"role": "assistant", "content": f"recover {index}"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    call("lookup" if index == 0 else "repair", f"right-{index}")
                ],
            },
            {
                "role": "tool",
                "content": "ok",
                "tool_call_id": (
                    f"{'lookup' if index == 0 else 'repair'}-right-{index}"
                ),
                "error": False,
            },
        ]
        branch["recovery_suffix"] = suffix
        branch["full_trace"] = [*deepcopy(branch["recovery_prompt"]), *suffix]
        branch["producer_only_note"] = f"keep-{index}"
        branches.append(branch)
    clean_messages = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "help"},
        {"role": "assistant", "content": "checking"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("lookup", "clean")],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "lookup-clean",
            "error": False,
        },
    ]
    raw = {
        "candidate_pair_id": "retail:1:pair:0",
        "partition": "arm_train",
        "domain": "retail",
        "official_test_used": False,
        "tool_schemas": TOOLS,
        "training_system_message": {
            "role": "system",
            "content": "<policy>frozen</policy>",
        },
        "branches": branches,
        "shared_prefix": clean_messages[:2],
        "clean_view": {
            "messages": clean_messages,
            "label_mask": [False, False, False, True, False],
            "producer_note": "keep-clean",
        },
        "raw_note": "keep-pair",
    }
    raw["training_system_message_sha256"] = measurement.canonical_sha256(
        raw["training_system_message"]
    )

    def scorer(prompt, action, tools):
        return {"sum_logprob": -2.0, "token_count": 2}

    enriched = measurement.enrich_candidate(
        raw,
        tokenizer=FakeChatTokenizer(),
        scorer=scorer,
        model_provenance={
            "model_name": measurement.DEFAULT_MODEL,
            "model_revision": "model-commit",
            "tokenizer_name": measurement.DEFAULT_MODEL,
            "tokenizer_revision": "tokenizer-commit",
            "frozen_checkpoint_identity_sha256": "a" * 64,
        },
    )
    assert raw["branches"][0].get("supervised_target_tokens") is None
    assert enriched["raw_note"] == "keep-pair"
    assert enriched["branches"][0]["producer_only_note"] == "keep-0"
    assert enriched["clean_view"]["producer_note"] == "keep-clean"
    assert enriched["clean_view"]["label_mask"] == [
        False,
        False,
        False,
        True,
        True,
        False,
    ]
    assert enriched["clean_view"]["producer_label_mask"] == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert enriched["token_accounting"]["supervised_target_tokens"] == sum(
        branch["supervised_target_tokens"] for branch in enriched["branches"]
    )
    assert enriched["token_accounting"]["nonpadding_tokens"] == sum(
        branch["nonpadding_tokens"] for branch in enriched["branches"]
    )
    assert set(enriched["first_action_logprobs"]) == set(measurement.Q_KEYS)
    assert (
        enriched["token_measurement_provenance"]["official_test_used"]
        is False
    )
