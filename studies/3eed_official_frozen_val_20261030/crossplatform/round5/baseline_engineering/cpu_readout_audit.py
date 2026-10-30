"""CPU token capacity and released positive-map bridge; no target labels indexed."""
import json,time,re
from pathlib import Path
import numpy as np
from transformers import RobertaTokenizerFast
from common import R,P,U,CLASS_NAMES,SENTINEL,class_masks,input_masks,sha,WAYMO_SYNONYMS,M3ED_SYNONYMS
from tta import text_span

def independent_offset_mask(tokenizer,formatted,beg,end):
 off=tokenizer(formatted,return_offsets_mapping=True)['offset_mapping']
 def find(ch):
  for i,(a,b) in enumerate(off):
   if a<=ch<b:return i
  return None
 bp=next((find(beg+k) for k in [0,1,2] if find(beg+k) is not None),None);ep=next((find(end-k) for k in [1,2,3] if find(end-k) is not None),None);out=np.zeros(256,bool)
 if bp is not None and ep is not None:out[bp:ep+1]=True
 return out

def main():
 t=time.perf_counter();samples=json.loads((R/'inputs/all.json').read_text());smoke=json.loads((R/'inputs/smoke48.json').read_text());tokenizer=RobertaTokenizerFast.from_pretrained(str((U/'data/roberta_base').resolve()));lengths=[]
 for start in range(0,len(samples),256):
  chunk=samples[start:start+256]
  for key in ['caption','noun_caption']:
   tok=tokenizer([s[key]+SENTINEL for s in chunk],padding=False,truncation=False);lengths.extend(len(ids) for ids in tok['input_ids'])
 assert max(lengths)<=256,(max(lengths),sum(x>256 for x in lengths));checks=0;mask_nonzero=0
 for s in smoke:
  for key in ['caption','noun_caption']:
   caption=s[key];cm,found,nz,span,mapids=class_masks(tokenizer,caption,s['platform']);formatted=' '+caption+' '
   for c in range(6):
    if found[c]:assert np.array_equal(cm[c],independent_offset_mask(tokenizer,formatted,*span[c]));checks+=1
   nm,ids,att,off=input_masks(tokenizer,caption+SENTINEL);expected,_=text_span(tokenizer,caption+SENTINEL);assert np.array_equal(nm[:len(expected)],expected.numpy());mask_nonzero+=int(nz.sum())
 out={'state':'CPU_inputs_and_readout_engineering_passed','n_raw_samples':len(samples),'n_conditions':2,'all_actual_token_lengths_minmax':[min(lengths),max(lengths)],'token_lengths_over256':sum(x>256 for x in lengths),'smoke_leading_space_officialmap_independent_offset_comparisons':checks,'smoke_allclass_nonzero_mask_count':mask_nonzero,'smoke_actual_firstnoun_mask_bridge_cases':len(smoke)*2,'actual_GT_class_or_bbox_indexed':False,'GPU_or_model_forward':False,'seconds':time.perf_counter()-t,'frozen_code_sha256':{f:sha(R/f) for f in ['common.py','run_frozen.py']},'official_map_source_sha256':sha(U/'src/joint_det_dataset.py'),'audit_code_sha256':sha(Path(__file__))};(R/'CPU_READOUT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
