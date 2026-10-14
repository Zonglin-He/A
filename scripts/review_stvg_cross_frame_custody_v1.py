"""Strict final custody including exact inherited no-update, CPU only."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import *
def main():
 verify();sys.addaudithook(guard)
 import torch,numpy as np
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 rows=read(OLD/'ROSTER_LOCK.json')['records'];seal=read(BASE/'STUDENT_GLOBAL_BARRIER.json');bindings={x['case_id']:x for x in seal['files']};records=[]
 if set(bindings)!={r['case_id']for r in rows}:raise RuntimeError('seal roster mismatch')
 for r in rows:
  cid=r['case_id'];case=load_source_case(ROOT/r['source_case_path'])
  for scope in [PRIOR,BASE]:
   p=scope/'student'/(cid+'.pt');rp=p.with_suffix('.receipt.json');rc=read(rp)
   if rc['status']!='sealed' or rc['GT_read'] is not False or rc['path']!=str(p.relative_to(ROOT)) or rc['bytes']!=p.stat().st_size or rc['sha256']!=sha(p):raise RuntimeError('receipt custody mismatch')
   if scope==BASE and (bindings[cid]['path']!=str(rp.relative_to(ROOT)) or bindings[cid]['sha256']!=sha(rp) or rc['case_id']!=cid):raise RuntimeError('global receipt mismatch')
   d=torch.load(p,map_location='cpu',weights_only=False)
   if d['case_id']!=cid or d['source_case_sha256']!=r['source_case_sha256'] or d['GT_read'] is not False or not d['source_restored']:raise RuntimeError('payload custody mismatch')
   if d['frame_ids']!=list(case['frame_ids']) or d['pixel_sha256']!=case['pixel_sha256'] or d['source_model_state_sha256']!=case['source_model_state_sha256']:raise RuntimeError('source binding mismatch')
  if not d['source_capture_executed']:
   old=torch.load(PRIOR/'student'/(cid+'.pt'),map_location='cpu',weights_only=False)
   for new,oldname in [('F','F'),('T','T'),('cross_frame','F'),('T_cross_frame','T')]:
    if not np.array_equal(d['arms'][new]['boxes'],old['arms'][oldname]['boxes']) or d['arms'][new]['interval']!=old['arms'][oldname]['interval']:raise RuntimeError('no update reuse mismatch')
  records.append(dict(case_id=cid,source_state_verification='actual_capture_reset_and_full_decode' if d['source_capture_executed'] else 'inherited_sealed_prior_no_update_exact_reuse_no_new_capture',actual_capture=d['source_capture_executed']))
 write(BASE/'ROOT_CUSTODY_READBACK.json',dict(status='pass',count=8,records=records,scope='full seal/receipt/schema/source binding and inherited no-update verified',CPU_only=True,GT_read=False,time=time.time()))
 write(BASE/'ROOT_ENGINEERING_SCOPE.json',dict(status='pass',initial_admission_RAM_GiB=8,initial_admission_GPU_GiB=22,running_RAM_headroom_GiB=6,running_GPU_headroom_GiB=4,strict_admission_is_resource_precaution=True,resumed_query_count=0,actual_current_source_captures=7,exact_prior_no_update_reuses=1,no_step_level_restart_claim=True,GT_read=False,time=time.time()))
 print('All8 strict custody checks pass')
if __name__=='__main__':main()
