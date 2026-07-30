from pathlib import Path
import unittest

import yaml


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs" / "v6_10_closure.yaml"
HANDOFF = REPOSITORY / "V6_10_RUN_HANDOFF.md"
PREREGISTRATION = REPOSITORY / "V6_10_CLOSURE_PREREGISTRATION.md"


class RuntimeServiceLifecycleTests(unittest.TestCase):
    def test_runtime_service_restart_contract_is_explicit(self):
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        session = config["provenance_contract"]["model_servers"][
            "runtime_service_session"
        ]
        self.assertEqual(
            session["id_format"],
            "v6_10-runtime-a{two_digit_attempt}-{utc_timestamp}",
        )
        for key in (
            "one_process_set_per_session_directory",
            "service_restart_requires_new_session_id",
            "service_restart_requires_new_runtime_receipts",
            "service_restart_requires_new_release_manifest",
            "immutable_reference_preflight_reusable_across_sessions",
            "immutable_registry_reusable_across_sessions",
            "runtime_session_id_is_operational_namespace_not_identity_evidence",
            "identity_evidence_is_pid_snapshot_socket_and_receipt_hashes",
        ):
            self.assertIs(session[key], True)
        resume = config["resume_and_fresh_run"]
        self.assertIn(
            "same_runtime_service_session",
            resume["resumable_only_when"],
        )
        self.assertIs(
            resume[
                "model_service_restart_requires_new_runtime_session_and_generation_attempt"
            ],
            True,
        )

    def test_handoff_uses_session_scoped_release_paths_and_full_clean_checks(
        self,
    ):
        handoff = HANDOFF.read_text(encoding="utf-8")
        normalized_handoff = " ".join(handoff.split())
        self.assertNotIn("--untracked-files=no", handoff)
        self.assertGreaterEqual(
            handoff.count("--untracked-files=all"),
            2,
        )
        self.assertNotIn(
            '$ROOT/releases/$SOURCE_COMMIT/runtime_receipts',
            handoff,
        )
        self.assertNotIn(
            '$ROOT/releases/$SOURCE_COMMIT/release_manifest.json',
            handoff,
        )
        for required in (
            "RUNTIME_SESSION_ID=",
            'RUNTIME_SESSION_DIR="$ROOT/releases/$SOURCE_COMMIT/runtime-sessions/$RUNTIME_SESSION_ID"',
            'RUNTIME_RECEIPTS="$RUNTIME_SESSION_DIR/runtime_receipts"',
            'RELEASE_MANIFEST="$RUNTIME_SESSION_DIR/release_manifest.json"',
        ):
            self.assertIn(required, handoff)
        for required in (
            "A model-service restart always requires a new runtime session ID",
            "reuse the immutable reference preflight and registry",
        ):
            self.assertIn(required, normalized_handoff)

    def test_preregistration_separates_static_and_ephemeral_identity(self):
        preregistration = PREREGISTRATION.read_text(encoding="utf-8")
        normalized_preregistration = " ".join(preregistration.split())
        for required in (
            "runtime service session",
            "new runtime receipts and release manifest",
            "reference preflight and executable registry remain reusable",
            "operational namespace",
        ):
            self.assertIn(required, normalized_preregistration)


if __name__ == "__main__":
    unittest.main()
