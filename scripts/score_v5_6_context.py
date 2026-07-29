#!/usr/bin/env python3
"""Score V5.6's correct repair action under true and shuffled error context.

This is deliberately an offline derived-validation diagnostic.  It does not
invoke the tau2 environment and cannot open the sealed official test.  For
each pair it scores exactly the first successful post-error assistant tool call
under two prompts with identical target tokens, then writes one paired row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import prepare_v5_5_full_sft as v55
    import prepare_v5_sft_causal as v5
    import prepare_v5_6_context_mechanism as data
    import v5_6_context_protocol as protocol
except ModuleNotFoundError:
    from scripts import prepare_v5_5_full_sft as v55
    from scripts import prepare_v5_sft_causal as v5
    from scripts import prepare_v5_6_context_mechanism as data
    from scripts import v5_6_context_protocol as protocol


SCORE_PROTOCOL = "v5_6_context_logprob_score_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_target_span(tokenizer: Any, row: dict[str, Any]) -> tuple[list[int], int, int]:
    """Render through the first repair action and return its exact token span."""
    index = data._first_label_index(row)
    messages = row["messages"]
    tools = row["metadata"]["tool_schemas"]
    before = v5._token_ids(tokenizer, messages[:index], tools=tools, generation=True)
    through = v5._token_ids(tokenizer, messages[: index + 1], tools=tools, generation=False)
    if not before or len(before) >= len(through) or through[: len(before)] != before:
        raise RuntimeError(f"{row['id']}: non-prefix-stable repair target")
    return through, len(before), len(through)


def score_span(model: Any, torch: Any, token_ids: list[int], start: int, end: int) -> float:
    if start <= 0 or end <= start:
        raise RuntimeError("invalid target span")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
        log_probs = torch.log_softmax(logits[:-1], dim=-1)
    return float(sum(log_probs[position - 1, token_ids[position]].item() for position in range(start, end)))


def task_deltas(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("protocol") != SCORE_PROTOCOL or row.get("official_test_used") is not False:
            raise RuntimeError("non-V5.6 or official-test score row")
        grouped[str(row["task_identity"])].append(float(row["delta_context"]))
    return {task: sum(values) / len(values) for task, values in grouped.items()}


def bootstrap_ci(values: list[float], *, seed: int, replicates: int) -> list[float]:
    if not values:
        raise RuntimeError("no task-level values")
    rng = random.Random(seed)
    estimates = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(replicates))
    return [estimates[int(0.025 * replicates)], estimates[min(replicates - 1, int(0.975 * replicates))]]


def sign_flip_pvalue(values: list[float], *, seed: int, replicates: int) -> float:
    observed = abs(sum(values) / len(values))
    rng = random.Random(seed)
    extreme = sum(abs(sum(value * rng.choice((-1, 1)) for value in values) / len(values)) >= observed - 1e-15 for _ in range(replicates))
    return (extreme + 1) / (replicates + 1)


def summarize(rows: list[dict[str, Any]], *, replicates: int) -> dict[str, Any]:
    values = task_deltas(rows)
    ordered = [values[task] for task in sorted(values)]
    return {"protocol": SCORE_PROTOCOL, "status": "PASS", "independent_unit": "task_identity", "tasks": len(ordered), "mean_delta_context": sum(ordered) / len(ordered), "bootstrap_95_ci": bootstrap_ci(ordered, seed=20260815, replicates=replicates), "sign_flip_pvalue": sign_flip_pvalue(ordered, seed=20260816, replicates=replicates), "interpretation": "positive values mean the checkpoint assigns more probability to the identical correct repair action with the matched error context than with a cross-task shuffled error context", "official_test_used": False}


def score(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.arm not in protocol.ARMS:
        raise RuntimeError("arm is outside frozen V5.6 grid")
    pairs = v55.read_jsonl(args.pairs)
    if any(pair.get("official_test_used") is not False for pair in pairs):
        raise RuntimeError("scoring pairs must seal the official test")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, trust_remote_code=True, local_files_only=args.local_files_only)
    contexts = v5.load_tau2_contexts(args.tau2_root)
    sources = v55.materialize_source_pairs(pairs, pair_mode=args.pair_mode, contexts=contexts, tokenizer=tokenizer, independent_replay_authorized=True)
    donor_map = data.deterministic_donors(sources, args.shuffle_seed)
    by_id = {item["pair_id"]: item for item in sources}
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.model_revision, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=args.local_files_only)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows: list[dict[str, Any]] = []
    for source in sources:
        true = source["recovery"]
        donor = by_id[donor_map[source["pair_id"]]]["recovery"]
        shuffled = data.make_shuffled_row(source["clean"], donor, tokenizer)
        true_ids, true_start, true_end = first_target_span(tokenizer, true)
        shuffled_ids, shuffled_start, shuffled_end = first_target_span(tokenizer, shuffled)
        if true_ids[true_start:true_end] != shuffled_ids[shuffled_start:shuffled_end]:
            raise RuntimeError(f"{source['pair_id']}: true/shuffled repair target drift")
        true_logp = score_span(model, torch, true_ids, true_start, true_end)
        shuffled_logp = score_span(model, torch, shuffled_ids, shuffled_start, shuffled_end)
        metadata = true["metadata"]
        rows.append({"protocol": SCORE_PROTOCOL, "arm": args.arm, "training_seed": args.training_seed, "pair_id": source["pair_id"], "task_identity": f"{metadata['domain']}:{metadata['task_id']}", "domain": metadata["domain"], "true_logp": true_logp, "shuffled_logp": shuffled_logp, "delta_context": true_logp - shuffled_logp, "target_tokens": true_end - true_start, "shuffle_donor_pair_id": donor["metadata"]["pair_id"], "model_revision": args.model_revision, "adapter": str(args.adapter) if args.adapter else None, "official_test_used": False})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-jsonl", type=Path, help="Summarize this already-scored JSONL instead of loading a model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=protocol.BOOTSTRAP_REPLICATES)
    parser.add_argument("--tau2-root", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--pair-mode", choices=sorted(v55.PAIR_MODES))
    parser.add_argument("--model", default=protocol.MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--arm", choices=protocol.ARMS)
    parser.add_argument("--training-seed", type=int, default=protocol.TRAINING_SEEDS[0])
    parser.add_argument("--shuffle-seed", type=int, default=protocol.TRAINING_SEEDS[0])
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.score_jsonl:
        rows = [json.loads(line) for line in args.score_jsonl.read_text(encoding="utf-8").splitlines() if line]
        summary = summarize(rows, replicates=args.replicates)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    required = (args.tau2_root, args.pairs, args.pair_mode, args.model_revision, args.arm)
    if any(value is None for value in required):
        parser.error("scoring requires --tau2-root, --pairs, --pair-mode, --model-revision and --arm")
    rows = score(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(summarize(rows, replicates=args.replicates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
