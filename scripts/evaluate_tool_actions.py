#!/usr/bin/env python3
"""Fixed held-out next-tool-call evaluator for QLoRA baseline v1.

The optional 4-bit loader exists for the frozen cross-machine OOM fallback.
All arms must use the same loader mode when their metrics are compared.
"""
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
    p=argparse.ArgumentParser(); p.add_argument("--test-file", type=Path, required=True); p.add_argument("--adapter", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); p.add_argument("--limit",type=int); p.add_argument("--dry-run",action="store_true"); p.add_argument("--load-in-4bit", action="store_true"); p.add_argument("--progress-every", type=int, default=25); p.add_argument("--max-prompt-tokens", type=int, default=512)
    args=p.parse_args(); rows=[json.loads(x) for x in args.test_file.read_text().splitlines() if x.strip()]; rows=rows[:args.limit] if args.limit else rows
    if args.dry_run:
        out={"examples":len(rows),"recovery_examples":sum(x["is_error_resolution_target"] for x in rows),"mode":"dry_run_no_model_loaded"}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out)); return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    model_kwargs={"device_map":"auto", "trust_remote_code":True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    else:
        model_kwargs["torch_dtype"]=torch.float16
    model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model,**model_kwargs),args.adapter); model.eval()
    scored=[]
    for index, row in enumerate(rows, start=1):
        # Training uses max_seq_len=512 with left truncation when histories are
        # longer.  Evaluation must use the same bounded, most-recent context;
        # otherwise a 4k-token test prompt is an out-of-protocol input.
        previous_side=tok.truncation_side; tok.truncation_side="left"
        inputs=tok(row["prompt"],return_tensors="pt",truncation=True,max_length=args.max_prompt_tokens)
        tok.truncation_side=previous_side
        inputs=inputs.to(model.device)
        with torch.inference_mode():
            out=model.generate(**inputs,max_new_tokens=128,do_sample=False,pad_token_id=tok.eos_token_id)
        generated=tok.decode(out[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True)
        pred=parse_call(generated); target=json.loads(row["completion"])
        scored.append({"example_id":row["example_id"],"recovery":row["is_error_resolution_target"],"tool_correct":bool(pred and pred["name"]==target["name"]),"call_exact":pred==target})
        if args.progress_every and index % args.progress_every == 0:
            print(f"evaluated {index}/{len(rows)}")
    def metric(items,key): return sum(x[key] for x in items)/len(items) if items else None
    recovery=[x for x in scored if x["recovery"]]
    result={"examples":len(scored),"tool_name_accuracy":metric(scored,"tool_correct"),"full_tool_call_exact_match":metric(scored,"call_exact"),"recovery_examples":len(recovery),"recovery_tool_name_accuracy":metric(recovery,"tool_correct"),"recovery_full_tool_call_exact_match":metric(recovery,"call_exact"),"base_model_loading":"nf4_4bit" if args.load_in_4bit else "fp16","prompt_context":{"max_tokens":args.max_prompt_tokens,"truncation":"left_keep_most_recent"},"generation":{"do_sample":False,"max_new_tokens":128}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
