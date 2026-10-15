"""Fit the frozen chosen temporal objective from fresh, sealed frozen-teacher responses."""
import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
from scripts.stvg_evidence_fresh_custody012 import checked_load_case as load_case
def main():
 import torch
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts.stvg_evidence_temporal_fit_v1 import fit
 verify_runtime();sys.addaudithook(guard);b=BASE/'confirmation';lock=read(BASE/'FROZEN_METHOD_LOCK.json');assert lock['status']=='locked';winner=lock['temporal_variant'];bar=read(b/'TEMPORAL_TEACHER_GLOBAL_BARRIER.json');assert bar['count']==32 and not bar['GT_read'];files=[]
 for rec in roster('confirmation'):
  cid=rec['probe_alias'];case,cp=load_case(rec);p=b/'temporal_teacher'/(cid+'.pt');item=next(x for x in bar['files']if ROOT/x['path']==p);assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);assert d['source_file_sha256']==sha(cp) and d['case_id']==cid and not d['GT_read'];assert d['request_sha256']==sha(b/'requests'/(cid+'.json'));t=d['temporal'];fits={}
  if 'temporal_fit'in t:
   rw=[ta._alignment_reward(c['canonical_id'],r)for c,r in zip(case['temporal_candidates'],t['teacher_responses'])];f=fit(case,ta._hidden_from_source_case(case),case['temporal_candidates'],rw,winner);iv=f['final_interval'];fits[winner]=f
  else:iv=case['Native_interval']
  out=dict(case_id=cid,dataset=rec['dataset'],fits=fits,selected_interval=list(iv),old_interval=list(t['final_interval']),source_temporal_teacher_sha256=sha(p),request_sha256=d['request_sha256'],source_case_sha256=case['source_case_sha256'],temporal_selection_sha256=sha(BASE/'TEMPORAL_SELECTION_LOCK.json'),frozen_method_sha256=sha(BASE/'FROZEN_METHOD_LOCK.json'),GT_read=False)
  files.append(save_pt(b/'temporal'/(cid+'.pt'),out));print('fresh temporal fit',cid,flush=True)
 write(b/'TEMPORAL_FIT_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False))
if __name__=='__main__':main()
