"""Four matched temporal arms, identical saved Teacher feedback and fixed spatial outputs."""
import sys,copy,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
 import torch
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts.stvg_evidence_temporal_fit_v1 import fit
 verify_runtime();sys.addaudithook(guard);torch.set_num_threads(2);b=BASE/'round1';files=[];start=time.perf_counter()
 tb=read(E/'TEMPORAL_GLOBAL_BARRIER.json');oldfiles={x['path']:x for x in tb['files']}
 for rec in roster():
  cid=rec['probe_alias'];case,cp=load_case(rec);p=E/'temporal'/(cid+'.pt');assert sha(p)==oldfiles[str(p.relative_to(ROOT))]['sha256'];t=torch.load(p,map_location='cpu',weights_only=False)['temporal'];sp,spp,z=fixed_dev_spatial(rec);assert z['source_case_sha256']==case['source_case_sha256']
  arms={'F':native_arm(case),'S_fixed':copy.deepcopy(sp)};fits={};has='temporal_fit'in t
  for name in ['projection','original','stable','offset_trust']:
   if has:
    rw=[ta._alignment_reward(c['canonical_id'],r)for c,r in zip(case['temporal_candidates'],t['teacher_responses'])];f=fit(case,ta._hidden_from_source_case(case),case['temporal_candidates'],rw,name);iv=f['final_interval'];fits[name]=f
    if name=='original':assert iv==t['final_interval']
   else:iv=case['Native_interval']
   arms['T_'+name]=native_arm(case,iv);arms['TS_'+name]=dict(boxes=copy.deepcopy(sp['boxes']),interval=list(iv))
  d=dict(case_id=cid,dataset=case['dataset'],parent_sha256=rec['parent_sha256'],arms=arms,fits=fits,frame_ids=case['frame_ids'],source_case_sha256=case['source_case_sha256'],source_file_sha256=sha(cp),reference_record=z['reference_record'],fixed_spatial_file_sha256=sha(spp),old_temporal_file_sha256=sha(p),source_restored=True,GT_read=False)
  files.append(save_pt(b/'student'/(cid+'.pt'),d));write(b/'STATUS.json',dict(status='running',done=len(files),total=32,GT_read=False),mutable=True);print('round1',cid,flush=True)
 write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,design_sha256=sha(BASE/'DESIGN_LOCK.json'),runtime_sha256=sha(BASE/'RUNTIME.json'),scorer_sha256=sha(ROOT/'scripts/score_stvg_evidence_v1.py'),wall_seconds=time.perf_counter()-start,GT_read=False))
if __name__=='__main__':main()
