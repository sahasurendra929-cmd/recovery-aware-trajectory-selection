#!/usr/bin/env python3
"""Offline held-out diagnostic for cross-session exploratory adapters only."""
from __future__ import annotations

import argparse, hashlib, json, os, random, re, tempfile
from pathlib import Path

import numpy as np
import torch

CLASS = "EXPLORATORY_NOT_FOR_FORMAL_GATE"
EVAL_TASKS = {"retail:31", "retail:11", "retail:104", "retail:72"}


def canonical(v): return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def atomic(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f: f.write(data); tmp=f.name
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()
def compact(m):
    o={"role":m["role"],"content":m.get("content")}
    if m.get("tool_calls"): o["tool_calls"]=m["tool_calls"]
    return o
def text(m):
    body=m.get("content") or ""
    if m.get("tool_calls"): body += "\n<tool_call>"+canonical(m["tool_calls"])+"</tool_call>"
    return f"<{m['role']}>\n{body}\n</{m['role']}>\n"
def first_call(messages):
    for m in messages:
        if m.get("tool_calls"):
            c=m["tool_calls"][0]; return (c.get("name"), canonical(c.get("arguments", {})))
    return None
def parsed_call(generation):
    m=re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", generation, re.S)
    if not m: return None
    try:
        v=json.loads(m.group(1)); c=v[0] if isinstance(v,list) else v
        return (c.get("name"), canonical(c.get("arguments", {}))) if isinstance(c,dict) and c.get("name") else None
    except (json.JSONDecodeError, TypeError): return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pool-index",type=Path,required=True); ap.add_argument("--model",required=True); ap.add_argument("--adapter",type=Path); ap.add_argument("--variant",required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--max-new-tokens",type=int,default=256); args=ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    entries=json.loads(args.pool_index.read_text())["files"]
    cases=[]
    for e in entries:
        if e["session"] not in {"A08","A12"} or e["task_identity"] not in EVAL_TASKS: continue
        for line in Path(e["path"]).read_text().splitlines():
            pair=json.loads(line); system=pair["training_system_message"]; system=system.get("content") if isinstance(system,dict) else system
            for b in pair["branches"]:
                prompt=[{"role":"system","content":system}]+[compact(m) for m in b["recovery_prompt"]]
                suffix=[compact(m) for m in b["recovery_suffix"]]
                cases.append((e["session"],pair,b,prompt,suffix))
    if len(cases)!=24: raise SystemExit(f"expected 24 held-out branches, got {len(cases)}")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,quantization_config=q,device_map={"":0},torch_dtype=torch.bfloat16)
    if args.adapter: model=PeftModel.from_pretrained(model,args.adapter,local_files_only=True)
    model.eval(); receipts=[]
    for session,pair,b,prompt,suffix in cases:
      prefix="".join(text(m) for m in prompt); suffix_text="".join(text(m) for m in suffix)
      expected=first_call(suffix); failed=first_call(prompt)
      for seed in (20260811,20260812):
        key=f"{args.variant}-{pair['candidate_pair_id']}-{b['branch_id']}-{seed}"; out=args.output_dir/f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        pi=tok(prefix,return_tensors="pt",add_special_tokens=False).input_ids.cuda(); full=tok(prefix+suffix_text,return_tensors="pt",add_special_tokens=False).input_ids.cuda()
        labels=full.clone(); labels[:,:pi.shape[1]]=-100
        with torch.inference_mode():
          loss=float(model(input_ids=full,labels=labels).loss)
          generated=model.generate(pi,max_new_tokens=args.max_new_tokens,do_sample=True,temperature=.7,top_p=.9,pad_token_id=tok.eos_token_id)
        generation=tok.decode(generated[0,pi.shape[1]:],skip_special_tokens=True); got=parsed_call(generation)
        rec={"classification":CLASS,"status":"PASS","variant":args.variant,"session":session,"task_identity":pair["task_identity"],"candidate_pair_id":pair["candidate_pair_id"],"branch_id":b["branch_id"],"seed":seed,"recovery_suffix_nll":loss,"generated_tokens":int(generated.shape[1]-pi.shape[1]),"valid_tool_call":got is not None,"first_repair_action_agreement":got==expected if got else False,"repeated_known_failure":got==failed if got else False,"expected_first_repair":expected,"generated_first_tool_call":got,"generation_sha256":hashlib.sha256(generation.encode()).hexdigest(),"official_test_used":False}
        atomic(out,rec); receipts.append({"path":out.name,"sha256":sha(out),"task_identity":rec["task_identity"],"seed":seed})
    atomic(args.output_dir/"OFFLINE_EVALUATION_RECEIPT_INDEX.json",{"classification":CLASS,"variant":args.variant,"items":receipts,"count":len(receipts),"status":"PASS","official_test_used":False})

if __name__=="__main__": main()
