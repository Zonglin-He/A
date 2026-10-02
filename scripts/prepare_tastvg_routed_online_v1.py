"""Bind the already authorized old cohort and reused A without reading GT."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_routed_common_v1 import *
def run():
 assert not BASE.exists(),'Never overwrite namespace'
 assert read(QUAL/'FINAL_COMPLETION.json')['status']=='completed'
 inputs={}
 def bind(f):inputs[str(f.relative_to(ROOT))]=sha(f)
 for ds in DATASETS:
  p=read(PRIOR/ds/'PLAN.json');write(BASE/ds/'PLAN.json',p);bind(BASE/ds/'PLAN.json');bind(PRIOR/ds/'PLAN.json');bind(POOL/ds/'CAPTURE_BARRIER.json')
  for a in ['A','R']:
   req=read(PRIOR/ds/'A/REQUEST.json');req.update(arm=a,tag=a);write(BASE/ds/a/'REQUEST.json',req);bind(BASE/ds/a/'REQUEST.json')
  for f in (PRIOR/ds/'A').rglob('*'):
   if f.is_file() and (f.suffix=='.pt' or f.name in ['SUPPORT.json','PREDICTION_BARRIER.json','STATUS.json'] or f.parent.name in ['order1','order2']):
    dst=BASE/ds/'A'/f.relative_to(PRIOR/ds/'A');dst.parent.mkdir(parents=True,exist_ok=True);os.link(f,dst);bind(f)
  write(BASE/ds/'A/REUSE.json',dict(status='sealed_A_exact_reuse',original=str((PRIOR/ds/'A').relative_to(ROOT)),cells=384,GT_read=False))
  for f in [QUAL/ds/'PLAN.json',QUAL/ds/'PREDICTION_BARRIER.json']:bind(f)
 for f in [ROOT/'methods/CURRENT_METHOD.json',PRIOR/'FINAL_COMPLETION.json',QUAL/'FINAL_COMPLETION.json',PRIOR/'BASIS.pt']:bind(f)
 os.link(PRIOR/'BASIS.pt',BASE/'BASIS.pt');write(BASE/'BASIS_LOCK.json',read(PRIOR/'BASIS_LOCK.json'))
 names=['protocols/tastvg_routed_online_token_v1.md','docs/tastvg_routed_online_token_v1/EXECUTION.md','scripts/tastvg_routed_common_v1.py','scripts/prepare_tastvg_routed_online_v1.py','scripts/run_tastvg_routed_online_v1.py','scripts/continue_tastvg_routed_online_v1.py','vg_tta/tastvg_reference_selection_v1.py','vg_tta/tastvg_event_support_v1.py','vg_tta/tastvg_selected_rollout_v1.py','scripts/run_tastvg_evidence_vulnerability_v2.py','scripts/run_tastvg_full_b1_experts_v1.py','scripts/with_local_cuda.sh','vg_tta/tastvg_paper48_hc2_decode_v1.py','vg_tta/tastvg_deployment_corruption_v2.py']
 for f in [ROOT/'checkpoints/Sa2VA-4B/DOWNLOAD_RECEIPT.json',ROOT/'checkpoints/Sa2VA-4B/OFFICIAL_CODE_RECEIPT.json']:bind(f)
 write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in names},inputs=inputs,max_new_specialist_calls=194,primary='hc2',cells=1536,new_cells=768,token_cells=60,GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='locked_ready_online',done=0,total=768,GT_read=False))
 print('LOCKED online768 plus reused768; P0 fixed60',flush=True)
if __name__=='__main__':run()
