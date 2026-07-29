from __future__ import annotations

from scripts import prepare_v5_5_full_sft as v55
from scripts import prepare_v5_6_context_mechanism as data
from scripts import score_v5_6_context as score
from scripts import train_v5_sft_causal as trainer
from tests.test_v5_5_full_sft_data import PrefixStableTokenizer, contexts, valid_pair


def test_v5_6_uses_the_validated_reference_context_capacity():
    assert trainer.V5_6_MAX_SEQUENCE_TOKENS == 10_240
    assert data.protocol.MAX_SEQUENCE_TOKENS == trainer.V5_6_MAX_SEQUENCE_TOKENS


def test_true_and_shuffled_schedules_are_exact_and_cross_task():
    token = PrefixStableTokenizer()
    source = v55.materialize_source_pairs(
        [valid_pair(1), valid_pair(2)], pair_mode="reference", contexts=contexts.__wrapped__(),
        tokenizer=token, independent_replay_authorized=True,
    )
    schedules, audit = data.build_schedules(source, token, 20260805)
    assert set(schedules) == {"perfect_success", "repair_25_true", "repair_25_shuffled"}
    assert {len(rows) for rows in schedules.values()} == {512}
    assert audit["arms"]["repair_25_true"]["realized_recovery_supervised_token_ratio"] == 0.25
    assert audit["arms"]["repair_25_shuffled"]["realized_recovery_supervised_token_ratio"] == 0.25
    assert all(
        arm_audit["max_sequence_tokens"] <= data.protocol.MAX_SEQUENCE_TOKENS
        for arm_audit in audit["arms"].values()
    )
    shuffled = next(row for row in schedules["repair_25_shuffled"] if row["metadata"]["source"] == "failure_rich")
    assert shuffled["metadata"]["error_context_matches_target"] is False
    assert shuffled["metadata"]["error_donor_task_identity"] != f"{shuffled['metadata']['domain']}:{shuffled['metadata']['task_id']}"
    assert trainer.validate_row(shuffled, arm="repair_25_shuffled", split="train")


def test_context_summary_is_task_clustered():
    rows = [
        {"protocol": score.SCORE_PROTOCOL, "official_test_used": False, "task_identity": "retail:1", "delta_context": 2.0},
        {"protocol": score.SCORE_PROTOCOL, "official_test_used": False, "task_identity": "retail:1", "delta_context": 0.0},
        {"protocol": score.SCORE_PROTOCOL, "official_test_used": False, "task_identity": "retail:2", "delta_context": -1.0},
    ]
    summary = score.summarize(rows, replicates=1000)
    assert summary["independent_unit"] == "task_identity"
    assert summary["tasks"] == 2
    assert summary["mean_delta_context"] == 0.0


def test_score_extracts_the_repair_span_with_training_canonicalization():
    token = PrefixStableTokenizer()
    source = v55.materialize_source_pairs(
        [valid_pair(1), valid_pair(2)],
        pair_mode="reference",
        contexts=contexts.__wrapped__(),
        tokenizer=token,
        independent_replay_authorized=True,
    )
    token_ids, start, end = score.first_target_span(
        token, source[0]["recovery"]
    )
    assert 0 < start < end <= len(token_ids)
