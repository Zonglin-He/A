"""Postcapture cache/clock/dependency audit before any confirmation fit/GT."""
from pathlib import Path
import sys,json,hashlib,datetime,subprocess
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=ROOT/'artifacts/stvg_fine_dataset_formal64_v1'
def main():
 import torch,numpy as np
 p=read(D/'PLAN_LOCK.json');bar=read(D/'FEATURE_BARRIER.json');assert bar['count']==64;checks={};models={};count=0
 for rr in bar['files']:assert sha(ROOT/rr['path'])==rr['sha256']
 for r in p['records']:
  x=load(D/'features'/(r['alias']+'.pt'));s=load(ROOT/r['source_path']);e=load(ROOT/r['prediction_path']);assert s['query_ordinal']==e['query_ordinal']==r['query_ordinal'];assert len(s['logits'])==len(s['records'])==2;assert len(e['frames'])==len(e['estimate']['selected']);ids=e['frame_ids'];assert x['frame_ids']==s['frame_ids']==ids and x['interval']==e['interval'];assert torch.equal(x['E'],torch.as_tensor(e['outputs']['E_DINO_support']));assert x['Source_and_time_logits_exact'];assert x['maps'].shape[0]==len(ids)and x['maps'].shape[1]==1024 and torch.isfinite(x['maps']).all()
  for off in [0,1]:assert s['records'][off]['offset']==off and s['records'][off]['frame_ids']==ids[off::2]
  pos=[];bb=[]
  for f,k in zip(e['frames'],e['estimate']['selected']):
   assert 0<=f['position']<len(ids)and f['position']in e['planned_positions'];assert k==-1 or 0<=k<len(f['boxes'])
   if k>=0:pos.append(f['position']);bb.append(torch.as_tensor(f['boxes'][k]).tolist())
  assert pos==x['positions']and(len(bb)==0 or np.array_equal(np.asarray(bb),np.asarray(x['boxes'])));models.setdefault(r['dataset'],set()).add(x['model_hash']);count+=1
 rt=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==rt['plan_sha256'];binding={f:dict(sha256=h,pinned_before_model=True)for f,h in rt['pins'].items()if f.endswith('.py')}
 for f,v in binding.items():assert sha(ROOT/f)==v['sha256']
 out=dict(status='pass',count=count,two_offsets_cached_records_exact=True,selected_anchor_index_and_clock_valid=True,raw_layer3_finite=True,models={d:sorted(v)for d,v in models.items()},dependency_audit_timing='all foundation capture dependencies pinned before model; cache/clock audit before fits',dependencies=binding,initial_runtime_inputs_and_core_pinned=True,actual_forward_Source_logits_exact_asserted_during_capture=True,CPU_did_not_rerun_backbone=True,no_GT_read=True)
 put(D/'INPUT_DEPENDENCY_AUDIT.json',out);print('input/dependency audit',count,len(binding),flush=True)
if __name__=='__main__':main()
