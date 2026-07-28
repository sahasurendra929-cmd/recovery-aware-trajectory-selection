from scripts.audit_v5_5_screen_terminal import audit


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
