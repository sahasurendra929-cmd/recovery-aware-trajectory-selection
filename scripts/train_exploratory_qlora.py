#!/usr/bin/env python3
"""Small, explicitly exploratory QLoRA screen trainer (not a formal V6 arm)."""
from __future__ import annotations
import argparse, hashlib, json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

CLASS="EXPLORATORY_NOT_FOR_FORMAL_GATE"

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(path,obj):
 data=(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode(); path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f: f.write(data); n=f.name
 os.replace(n,path); path.with_suffix(path.suffix+'.sha256').write_text(f'{hashlib.sha256(data).hexdigest()}  {path.name}\n')
def message_text(m):
 x=m.get('content') or ''
 if m.get('tool_calls'): x += '\n<tool_call>'+json.dumps(m['tool_calls'],sort_keys=True,separators=(',',':'))+'</tool_call>'
 return f"<{m['role']}>\n{x}\n</{m['role']}>\n"
def collate(rows,tok,max_len):
 ids=[]; labels=[]
 for r in rows:
  parts=[]; labs=[]
  for m,lab in zip(r['messages'],r['label_mask']):
   q=tok(message_text(m),add_special_tokens=False).input_ids
   parts+=q; labs += q if lab else [-100]*len(q)
  parts=parts[:max_len]; labs=labs[:max_len]
  ids.append(torch.tensor(parts)); labels.append(torch.tensor(labs))
 n=max(map(len,ids)); pad=tok.pad_token_id
 return {'input_ids':torch.stack([torch.nn.functional.pad(x,(0,n-len(x)),value=pad) for x in ids]),'attention_mask':torch.stack([torch.nn.functional.pad(torch.ones_like(x),(0,n-len(x))) for x in ids]),'labels':torch.stack([torch.nn.functional.pad(x,(0,n-len(x)),value=-100) for x in labels])}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--model',required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--arm',required=True); p.add_argument('--epochs',type=int,default=2); p.add_argument('--seed',type=int,default=20260810); a=p.parse_args()
 from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
 from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
 rows=[json.loads(x) for x in a.data.read_text().splitlines() if x]
 if len(rows)!=48 or any(x['metadata']['classification']!=CLASS for x in rows): raise SystemExit('exploratory input boundary drift')
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
 a.out.mkdir(parents=True,exist_ok=False); atomic(a.out/'RUN_MANIFEST.json',{'classification':CLASS,'arm':a.arm,'data_sha256':sha(a.data),'model_path':a.model,'epochs':a.epochs,'seed':a.seed,'status':'RUNNING','formal_authorization':False})
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='right'
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,quantization_config=q,device_map={'':0},torch_dtype=torch.bfloat16); model.config.use_cache=False
 model=prepare_model_for_kbit_training(model); model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],bias='none',task_type='CAUSAL_LM'))
 opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=2e-4); dl=DataLoader(rows,batch_size=1,shuffle=True,collate_fn=lambda x:collate(x,tok,2048)); log=a.out/'train.log'
 step=0
 for epoch in range(1,a.epochs+1):
  model.train(); opt.zero_grad(); losses=[]
  for batch in dl:
   batch={k:v.cuda() for k,v in batch.items()}; out=model(**batch); (out.loss/4).backward(); losses.append(float(out.loss.detach())); step+=1
   if step%4==0: opt.step(); opt.zero_grad()
   with log.open('a') as f: f.write(json.dumps({'epoch':epoch,'step':step,'loss':losses[-1]})+'\n')
  if step%4: opt.step(); opt.zero_grad()
  d=a.out/f'checkpoint-epoch-{epoch}'; model.save_pretrained(d,safe_serialization=True); tok.save_pretrained(d)
  files={str(x.relative_to(d)):sha(x) for x in d.rglob('*') if x.is_file()}; atomic(d/'CHECKPOINT_MANIFEST.json',{'classification':CLASS,'arm':a.arm,'epoch':epoch,'step':step,'mean_loss':sum(losses)/len(losses),'files':files,'data_sha256':sha(a.data),'portable_adapter_only':True})
 atomic(a.out/'RUN_MANIFEST.json',{'classification':CLASS,'arm':a.arm,'data_sha256':sha(a.data),'model_path':a.model,'epochs':a.epochs,'seed':a.seed,'status':'TRAINING_COMPLETE','final_step':step,'formal_authorization':False})
if __name__=='__main__': main()
