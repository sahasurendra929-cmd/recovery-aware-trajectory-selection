import hashlib
import json

from scripts.audit_v5_5_screen_terminal import audit, audit_result_grid


def _summary(positive_screen: bool) -> dict:
    return {
        "status": "PASS",
        "official_test_used": False,
        "grid_audit": {
            "status": "PASS",
            "observed_rows": 378,
            "missing_rows": 0,
            "extra_rows": 0,
            "duplicate_rows": 0,
        },
        "statistics": {
            "selection": {
                "positive_screen": positive_screen,
                "selected_arm": "repair_50",
                "selected_paper_arm": "r50",
            }
        },
    }


def test_positive_screen_authorizes_stage_c():
    result = audit(_summary(True))
    assert result["status"] == "GO_STAGE_C"
    assert result["integrity_status"] == "PASS"


def test_negative_screen_is_audited_no_go_not_integrity_failure():
    result = audit(_summary(False))
    assert result["status"] == "NO_GO_REFERENCE_SCREEN"
    assert result["integrity_status"] == "PASS"


def test_missing_statistics_fails_closed():
    summary = _summary(True)
    del summary["statistics"]
    result = audit(summary)
    assert result["status"] == "FAIL_CLOSED"
    assert result["checks"]["selection_present"] is False


def test_result_grid_verifies_all_27_shards_and_summary_hashes(tmp_path):
    summary = _summary(False)
    hashes = {}
    for arm in ("perfect_success", "repair_50", "repair_100"):
        for seed in (20260815, 20260816, 20260817):
            for shard_index in range(3):
                relative = (
                    f"{arm}/20260805/{seed}/shard-{shard_index}"
                )
                shard = tmp_path / "evaluation" / relative
                shard.mkdir(parents=True)
                rows = [
                    {"official_test_used": False, "row": index}
                    for index in range(14)
                ]
                rows_bytes = (
                    "\n".join(json.dumps(row) for row in rows) + "\n"
                ).encode()
                (shard / "rows.jsonl").write_bytes(rows_bytes)
                (shard / "metrics.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS",
                            "rows": 14,
                            "official_test_used": False,
                        }
                    )
                )
                (shard / "run_contract.json").write_text(
                    json.dumps(
                        {
                            "status": "COMPLETE",
                            "official_test_used": False,
                        }
                    )
                )
                hashes[f"{relative}/rows.jsonl"] = hashlib.sha256(
                    rows_bytes
                ).hexdigest()
    summary["result_file_sha256"] = hashes
    result = audit_result_grid(tmp_path, summary)
    assert result["status"] == "PASS"
    assert result["shards"] == 27
    assert result["rows"] == 378
    assert result["all_summary_row_hashes_verified"] is True
