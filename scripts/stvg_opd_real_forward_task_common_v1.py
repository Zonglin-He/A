"""Human-authorized bounded complete-task test; old screen decision and monitor stop remain intact."""
import sys,time,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts import stvg_opd_forward_candidate_common_v1 as screen
prior=screen.prior
BASE=ROOT/'artifacts/stvg_opd_real_forward_task_validation_v1'
OUT=ROOT/'results/stvg_opd_real_forward_task_validation/2026-10-13'
ARMS=['A','SEL','HALF','LN'];BOOT=10000;SEED=20261013;EPS=1e-12
SCIENCE=['protocols/stvg_opd_real_forward_task_validation_v1.md','vg_tta/stvg_opd_real_forward_task_v1.py','scripts/stvg_opd_real_forward_task_common_v1.py','scripts/run_stvg_opd_real_forward_task_v1.py','scripts/score_stvg_opd_real_forward_task_v1.py','scripts/audit_stvg_opd_real_forward_task_v1.py']
def verify():
 screen.verify();r=read(BASE/'RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
 pins=dict(r['pins'])
 for p in sorted(BASE.glob('*/REVISION_RUNTIME.json'),key=lambda p:read(p)['time']):
  v=read(p);assert v['original_runtime_sha256']==sha(BASE/'RUNTIME.json');pins.update(v['pin_overrides'])
 for n,h in {**pins,**r['bindings']}.items():assert sha(ROOT/n)==h,n
 prior.standby();return r

def prepare():
 import torch
 from vg_tta.stvg_opd_real_forward_task_v1 import contracts
 torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not BASE.exists()
 screen.verify();assert read(screen.BASE/'FINAL_COMPLETION.json')['status']=='complete'
 assert read(screen.BASE/'TECHNICAL_DECISION.json')['status']=='stop_fixed_forward_candidate_recipe_before_GT'
 rr=read(prior.BASE/'DESIGN_LOCK.json')['records'];assert len(rr)==64
 bindings={}
 for r in rr:
  for n,h in [(r['A_path'],r['A_sha256']),(r['pair_path'],r['pair_sha256'])]:assert sha(ROOT/n)==h;bindings[n]=h
 for b in [screen.BASE,prior.BASE]:
  for n in ['RUNTIME.json','DESIGN_LOCK.json','FINAL_COMPLETION.json','TECHNICAL_DECISION.json']:
   bindings[str((b/n).relative_to(ROOT))]=sha(b/n)
 BASE.mkdir();OUT.mkdir(parents=True)
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_GPU_GT',records=rr,parents=64,qualification_indices=[0,1,32,36],qualification_fits=32,qualification_rounds=800,formal_arms=ARMS,
  control_replays=64,candidate_fits=192,total_complete_formal_fits=256,total_complete_formal_rounds=6400,
  entering_state='Inherited_order1',candidates=['A','LN','Query','Half'],rho=.9,no_GT_online=True,
  historical_exposure=True,old_screen_coverage_rule_preserved=True,new_task_screen_coverage_gate=False,
  fresh_Adam_per_arm_query=True,masked_proposal_moments_advanced_all1792=True,automatic_expansion=False,production_promoted=False,monitor_recreated=False,time=time.time()))
 write(BASE/'CPU_CONTRACTS.json',dict(contracts(),time=time.time()))
 write(BASE/'BASELINE_STANDBY_IDENTITY.json',read(prior.BASE/'BASELINE_STANDBY_IDENTITY.json'))
 write(BASE/'RUNTIME.json',dict(status='locked_before_new_GPU_GT',pins={n:sha(ROOT/n) for n in SCIENCE},bindings=bindings,
  design_sha256=sha(BASE/'DESIGN_LOCK.json'),GT_read=False,new_DINO_calls=0,time=time.time()))
 write(BASE/'SOURCE_STATE_CUSTODY_READBACK.json',dict(status='pass',parents=64,reference_fits=64,scope='original complete A fits, Inherited-order1 entering state/QR0/Omega0/expert/pixels/frames/source/config; recomputed live proposals rather than old trajectory splice',CPU_only=True,GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='prepared_pending_actual_qualification',done=0,total=32,GT_read=False,time=time.time()));verify()
