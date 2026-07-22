#!/usr/bin/env python3
"""Fixed held-out next-tool-call evaluator for QLoRA baseline v1."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

def parse_call(text):
    match = re.search(r'\{.*\}', text, re.S)
    if not match: return None
    try:
        value=json.loads(match.group(0)); return value if isinstance(value, dict) and "name" in value else None
    except json.JSONDecodeError: return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--test-file", type=Path, required=True); p.add_argument("--adapter", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); p.add_argument("--limit",type=int); p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(); rows=[json.loads(x) for x in args.test_file.read_text().splitlines() if x.strip()]; rows=rows[:args.limit] if args.limit else rows
    if args.dry_run:
        out={"examples":len(rows),"recovery_examples":sum(x["is_error_resolution_target"] for x in rows),"mode":"dry_run_no_model_loaded"}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out)); return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True); model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model,device_map="auto",torch_dtype=torch.float16,trust_remote_code=True),args.adapter); model.eval()
    scored=[]
    for row in rows:
        inputs=tok(row["prompt"],return_tensors="pt").to(model.device)
        out=model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
        generated=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        pred=parse_call(generated); target=json.loads(row["completion"])
        scored.append({"example_id":row["example_id"],"recovery":row["is_error_resolution_target"],"tool_correct":bool(pred and pred["name"]==target["name"]),"call_exact":pred==target})
    def metric(items,key): return sum(x[key] for x in items)/len(items) if items else None
    recovery=[x for x in scored if x["recovery"]]
    result={"examples":len(scored),"tool_name_accuracy":metric(scored,"tool_correct"),"full_tool_call_exact_match":metric(scored,"call_exact"),"recovery_examples":len(recovery),"recovery_tool_name_accuracy":metric(recovery,"tool_correct"),"recovery_full_tool_call_exact_match":metric(recovery,"call_exact")}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
