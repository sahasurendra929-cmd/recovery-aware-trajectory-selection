#!/usr/bin/env python3
"""Fail-closed natural-language assertion judging for V5 generation.

The upstream tau2 judge accepts ``{"results": []}`` because ``all([])`` is
true.  It also does not require the returned rows to cover the registered
assertions.  This module keeps the provider call auditable while enforcing an
exact, one-to-one result schema before tau2 is allowed to compute a reward.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


STRICT_NL_JUDGE_PROTOCOL = "v5_strict_nl_judge_v1"
DEFAULT_CONTENT_ATTEMPTS = 2
_FENCED_JSON_RE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n?(?P<body>.*?)\r?\n?```[ \t]*\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


class StrictNLJudgeResponseError(RuntimeError):
    """The judge exhausted bounded retries without a valid complete response."""

    def __init__(self, message: str, *, audit: dict[str, Any] | None = None):
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class StrictNLResult:
    expected_outcome: str
    met_expectation: bool
    reasoning: str


def parse_json_object(content: Any) -> dict[str, Any]:
    """Parse one bare JSON object or one complete Markdown JSON fence.

    Prose before/after a fence and embedded/salvaged JSON are deliberately
    rejected.  Accepting only the complete response keeps malformed judge
    output from being silently reinterpreted.
    """

    if not isinstance(content, str) or not content.strip():
        raise StrictNLJudgeResponseError("judge content must be a nonempty string")
    candidate = content.strip().lstrip("\ufeff")
    fence = _FENCED_JSON_RE.fullmatch(candidate)
    if fence is not None:
        candidate = fence.group("body").strip()
        if not candidate:
            raise StrictNLJudgeResponseError("fenced judge JSON is empty")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise StrictNLJudgeResponseError(
            f"judge response is not valid JSON: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise StrictNLJudgeResponseError("judge response must be a JSON object")
    return payload


def validate_complete_results(
    payload: dict[str, Any],
    nl_assertions: Sequence[str],
) -> list[StrictNLResult]:
    """Validate exact assertion coverage and return rows in registered order."""

    assertions = list(nl_assertions)
    if not assertions or any(
        not isinstance(assertion, str) or not assertion
        for assertion in assertions
    ):
        raise StrictNLJudgeResponseError(
            "registered nl_assertions must be nonempty strings"
        )
    if len(set(assertions)) != len(assertions):
        raise StrictNLJudgeResponseError(
            "registered nl_assertions contain duplicate expected outcomes"
        )

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise StrictNLJudgeResponseError(
            "judge results must be a nonempty JSON array"
        )
    if len(results) != len(assertions):
        raise StrictNLJudgeResponseError(
            "judge results count does not match registered nl_assertions: "
            f"expected {len(assertions)}, observed {len(results)}"
        )

    expected = set(assertions)
    by_outcome: dict[str, StrictNLResult] = {}
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise StrictNLJudgeResponseError(
                f"judge result {index} must be a JSON object"
            )
        outcome = row.get("expectedOutcome")
        if not isinstance(outcome, str):
            raise StrictNLJudgeResponseError(
                f"judge result {index}.expectedOutcome must be a string"
            )
        if outcome not in expected:
            raise StrictNLJudgeResponseError(
                f"judge result {index} contains an unknown expectedOutcome"
            )
        if outcome in by_outcome:
            raise StrictNLJudgeResponseError(
                f"judge results duplicate expectedOutcome {outcome!r}"
            )
        met = row.get("metExpectation")
        if type(met) is not bool:
            raise StrictNLJudgeResponseError(
                f"judge result {index}.metExpectation must be a JSON boolean"
            )
        reasoning = row.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise StrictNLJudgeResponseError(
                f"judge result {index}.reasoning must be a nonempty string"
            )
        by_outcome[outcome] = StrictNLResult(
            expected_outcome=outcome,
            met_expectation=met,
            reasoning=reasoning,
        )

    missing = expected - set(by_outcome)
    if missing:
        raise StrictNLJudgeResponseError(
            f"judge results omit {len(missing)} registered expected outcome(s)"
        )
    return [by_outcome[assertion] for assertion in assertions]


def parse_and_validate_response(
    content: Any,
    nl_assertions: Sequence[str],
) -> list[StrictNLResult]:
    return validate_complete_results(parse_json_object(content), nl_assertions)


def _message_audit(message: Any, *, attempt: int) -> dict[str, Any]:
    content = getattr(message, "content", None)
    raw_content = content if isinstance(content, str) else None
    encoded = (raw_content or "").encode("utf-8")
    return {
        "attempt": attempt,
        "call_name": f"nl_assertions_eval_strict_attempt_{attempt}",
        "raw_content": raw_content,
        "raw_content_utf8_bytes": len(encoded),
        "raw_content_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_provider_response": getattr(message, "raw_data", None),
        "schema_status": "UNVALIDATED",
        "schema_error": None,
    }


def _write_current_tau2_audit(payload: dict[str, Any]) -> Path | None:
    """Write alongside tau2's per-simulation LLM logs when logging is enabled."""

    try:
        from tau2.utils.llm_utils import llm_log_dir
    except (ImportError, ModuleNotFoundError):
        return None
    directory = llm_log_dir.get()
    if directory is None:
        return None
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    output = path / f"strict_nl_judge_audit_{uuid.uuid4().hex[:12]}.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def run_strict_content_retry(
    *,
    nl_assertions: Sequence[str],
    request: Callable[[int, str | None], Any],
    max_content_attempts: int = DEFAULT_CONTENT_ATTEMPTS,
    audit_writer: Callable[[dict[str, Any]], Any] | None = None,
) -> list[StrictNLResult]:
    """Retry only malformed content, then raise rather than award a reward."""

    if (
        type(max_content_attempts) is not int
        or max_content_attempts < 1
        or max_content_attempts > 3
    ):
        raise ValueError("max_content_attempts must be an integer in [1, 3]")
    writer = audit_writer or _write_current_tau2_audit
    attempts: list[dict[str, Any]] = []
    last_error: str | None = None

    for attempt in range(1, max_content_attempts + 1):
        # Provider/network exceptions intentionally propagate.  They are not
        # schema failures and must not consume the content-retry policy.
        message = request(attempt, last_error)
        record = _message_audit(message, attempt=attempt)
        try:
            results = parse_and_validate_response(
                getattr(message, "content", None),
                nl_assertions,
            )
        except StrictNLJudgeResponseError as error:
            last_error = str(error)
            record["schema_status"] = "REJECTED"
            record["schema_error"] = last_error
            attempts.append(record)
            continue
        record["schema_status"] = "PASS"
        attempts.append(record)
        writer(
            {
                "protocol": STRICT_NL_JUDGE_PROTOCOL,
                "status": "PASS",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expected_outcomes": list(nl_assertions),
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
        )
        return results

    audit = {
        "protocol": STRICT_NL_JUDGE_PROTOCOL,
        "status": "FAIL_CLOSED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_outcomes": list(nl_assertions),
        "attempt_count": len(attempts),
        "attempts": attempts,
    }
    writer(audit)
    raise StrictNLJudgeResponseError(
        "strict NL judge exhausted content retries without an exact result set: "
        f"{last_error}",
        audit=audit,
    )


def _judge_messages(
    trajectory: Sequence[Any],
    nl_assertions: Sequence[str],
    *,
    retry_error: str | None,
) -> list[Any]:
    from tau2.data_model.message import SystemMessage, UserMessage

    trajectory_str = "\n".join(
        f"{message.role}: {message.content}" for message in trajectory
    )
    system_prompt = """
TASK
- Grade every expected outcome against the supplied conversation.
- Return exactly one result for every expected outcome.

FORMAT
- Return only a JSON object, optionally enclosed in one ```json fence.
- `results` must be a nonempty array with exactly one row per expected outcome.
- Copy each `expectedOutcome` exactly; do not omit, add, paraphrase, or repeat it.
- `metExpectation` must be the JSON boolean true or false.
- `reasoning` must be a nonempty string.

Example:
{"results":[{"expectedOutcome":"<exact input string>","reasoning":"<why>",
"metExpectation":false}]}
""".strip()
    retry_instruction = ""
    if retry_error is not None:
        retry_instruction = (
            "\n\nYour previous response was rejected by the strict schema: "
            f"{retry_error}. Return a corrected complete JSON object."
        )
    user_prompt = (
        "conversation:\n"
        f"{trajectory_str}\n\n"
        "expectedOutcomes (copy each string exactly):\n"
        f"{json.dumps(list(nl_assertions), ensure_ascii=False)}"
        f"{retry_instruction}"
    )
    return [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt),
    ]


def strict_evaluate_nl_assertions(
    *,
    trajectory: Sequence[Any],
    nl_assertions: Sequence[str],
    model: str,
    llm_args: dict[str, Any],
    max_content_attempts: int = DEFAULT_CONTENT_ATTEMPTS,
) -> list[Any]:
    """Tau2-compatible strict evaluator entry point."""

    from tau2.data_model.simulation import NLAssertionCheck
    from tau2.utils.llm_utils import generate

    def request(attempt: int, last_error: str | None) -> Any:
        return generate(
            model=model,
            messages=_judge_messages(
                trajectory,
                nl_assertions,
                retry_error=last_error,
            ),
            call_name=f"nl_assertions_eval_strict_attempt_{attempt}",
            **dict(llm_args),
        )

    rows = run_strict_content_retry(
        nl_assertions=nl_assertions,
        request=request,
        max_content_attempts=max_content_attempts,
    )
    return [
        NLAssertionCheck(
            nl_assertion=row.expected_outcome,
            met=row.met_expectation,
            justification=row.reasoning,
        )
        for row in rows
    ]


def install_strict_nl_judge(
    model: str,
    llm_args: dict[str, Any],
    max_content_attempts: int = DEFAULT_CONTENT_ATTEMPTS,
) -> None:
    """Install the strict entry point for both text and full-duplex tau2 judges."""

    import tau2.evaluator.evaluator_nl_assertions as nl_module

    def evaluate(
        cls: type[Any],
        trajectory: list[Any],
        nl_assertions: list[str],
    ) -> list[Any]:
        del cls
        return strict_evaluate_nl_assertions(
            trajectory=trajectory,
            nl_assertions=nl_assertions,
            model=model,
            llm_args=llm_args,
            max_content_attempts=max_content_attempts,
        )

    nl_module.DEFAULT_LLM_NL_ASSERTIONS = model
    nl_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(llm_args)
    nl_module.NLAssertionsEvaluator.evaluate_nl_assertions = classmethod(evaluate)
