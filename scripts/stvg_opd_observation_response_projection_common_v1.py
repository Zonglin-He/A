"""Immutable fixed panel/qualification authority for one new update operator."""
import collections,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts import stvg_opd_LN_state_intervention_common_v1 as old
BASE=ROOT/'artifacts/stvg_opd_observation_response_projection_v1'
OUT=ROOT/'results/stvg_opd_observation_response_projection/2026-10-13'
ARMS=['P','F','R'];BOOT=10000;SEED=20261013
SCIENCE=['protocols/stvg_opd_observation_response_projection_v1.md',
 'vg_tta/stvg_opd_observation_response_projection_v1.py',
 'scripts/stvg_opd_observation_response_projection_common_v1.py',
 'scripts/run_stvg_opd_observation_response_projection_v1.py',
 'scripts/audit_stvg_opd_observation_response_projection_qualification_v1.py']

def revision_bindings():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(BASE.glob('*/REVISION_RUNTIME.json'))}

def standby():
 from scripts.pause_stvg_opd_for_geometry_v1 import process
 for expected in read(BASE/'BASELINE_STANDBY_IDENTITY.json')['processes']:
  p=process(expected['pid']);assert p['state']=='T' and p['start_ticks']==expected['start_ticks'] and p['command']==expected['command']

def verify():
 r=read(BASE/'RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
 pins=dict(r['pins'])
 for p in sorted(BASE.glob('*/REVISION_RUNTIME.json'),key=lambda p:read(p)['time']):
  rev=read(p);assert rev['original_runtime_sha256']==sha(BASE/'RUNTIME.json')
  pins.update(rev['pin_overrides'])
 for n,h in {**pins,**r['bindings'],**r['protected']}.items():assert sha(ROOT/n)==h,n
 assert read(old.S/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')['status']=='remaining_formal_cancelled_by_explicit_human_instruction'
 standby();return r

def cpu_contracts():
 from vg_tta.stvg_opd_observation_response_projection_v1 import solve
 import torch
 assert not torch.cuda.is_initialized();g=np.random.default_rng(SEED);maxeq=maxnorm=maxstation=0.
 # Explicit example, and full-rank/rank-deficient/zero-response/boundary ball.
 p,f,w=solve([[1.,0.,0.]],[[1.,1.,0.]],[0.],[1.,0.,1.],1)
 assert np.allclose(p,[1,0,0]) and np.allclose(f,[1,-1,0])
 for i in range(100):
  n=10+(i%11);o=g.normal(size=(4,n));u=g.normal(size=(12,n));e=g.normal(size=12);a=g.normal(size=n)
  if i%5==0:o[3]=o[0]
  if i%7==0:e*=0
  p,f,w=solve(o,u,e,a,3)
  eq=np.linalg.norm(o@(f-a));dn=np.linalg.norm(f)-np.linalg.norm(a)
  assert eq<1e-8 and dn<1e-8 and w['linear_F_objective']<=w['linear_A_objective']+1e-8
  assert w['stationarity_norm']<1e-7
  maxeq=max(maxeq,eq);maxnorm=max(maxnorm,dn);maxstation=max(maxstation,w['stationarity_norm'])
 assert not torch.cuda.is_initialized()
 return dict(status='pass',count=101,scope='synthetic constrained ball least-squares/observed response/norm/KKT/example; not actual GPU qualification',
  max_equality_error=maxeq,max_norm_excess=maxnorm,max_stationarity=maxstation,CPU_only=True,GT_read=False,time=time.time())

def prepare():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not BASE.exists() and not OUT.exists()
 old.verify();assert read(old.BASE/'FINAL_COMPLETION.json')['status']=='complete'
 retention=ROOT/'artifacts/stvg_opd_state_benefit_retention_audit_v1'
 assert read(retention/'FINAL_COMPLETION.json')['status']=='complete'
 d=read(old.BASE/'DESIGN_LOCK.json');b=read(old.BASE/'GLOBAL_PREDICTION_BARRIER.json')
 records=[];bindings={};empty={}
 for original in d['records']:
  r=dict(original);p=old.BASE/'formal'/f"{r['index']:05}"/'Inherited_order1_repeat0.pt';rel=str(p.relative_to(ROOT))
  assert sha(p)==b['files'][rel];z=load(p);pair=old.pair_for(r)
  assert old.digest(z['fit']['initial'])==r['initial_hashes']['Inherited_order1']
  assert z['fit']['config']==r['config'] and not z['GT_read']
  r.update(A_path=rel,A_sha256=sha(p),A_receipt=str(p.with_suffix('.json').relative_to(ROOT)),visited=pair['visited'][0])
  bindings[rel]=sha(p);bindings[r['A_receipt']]=sha(p.with_suffix('.json'))
  bindings[r['pair_path']]=r['pair_sha256'];bindings.update(pair['source_bindings'])
  empty[r['index']]=z['fit']['empty'];records.append(r)
 assert len(records)==64 and collections.Counter(r['dataset'] for r in records)=={'hc2':32,'vidstg':32}
 qual=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in records if r['dataset']==ds];ee=[r['index'] for r in rr if empty[r['index']]]
  if ee:indices=[r['index'] for r in rr if not empty[r['index']]][:3]+ee[:1]
  else:indices=[r['index'] for r in rr][:4]
  assert len(indices)==4;qual+=sorted(indices)
 BASE.mkdir();OUT.mkdir(parents=True)
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_GPU_GT',records=records,qualification_indices=qual,
  parents=64,formal_arms=ARMS,reference_arms=['A','N'],formal_fits=192,total_formal_rounds=4800,
  qualification_queries=8,qualification_repeats=2,qualification_boundaries='0,floor(K/2),K-1',
  original_complete_replays=16,candidate_local_proposals=144,acceptance_fraction_per_domain=.5,
  historically_exposed_panel=True,entering_LN='Inherited_order1',active_parameters=1792,
  common_Before_reference=True,physical_Omega0_fixed=True,new_DINO_calls=0,bootstrap=BOOT,seed=SEED,
  relative_observed_response_tolerance=.1,Vid_mvIoU_noninferiority_margin_pp=.5,
  temporal_projection_or_observation_reallocation=False,parameter_search=False,
  production_promoted=False,main_protocol_complete=False,remaining_formal_baseline_authority=False,time=time.time()))
 write(BASE/'CPU_CONTRACTS.json',cpu_contracts())
 from scripts.pause_stvg_opd_for_geometry_v1 import process
 ps=[process(i) for i in [509176,3091846]]
 prior=read(old.BASE/'BASELINE_STANDBY_IDENTITY.json')['processes']
 assert all(p['state']=='T' and p['start_ticks']==q['start_ticks'] for p,q in zip(ps,prior))
 write(BASE/'BASELINE_STANDBY_IDENTITY.json',dict(status='pass',processes=ps,scope='explicit cancellation; same original T process identity; no signals',GT_read=False,time=time.time()))
 shared=read(old.BASE/'RUNTIME.json')['pins']
 for n in ['RUNTIME.json','DESIGN_LOCK.json','GLOBAL_PREDICTION_BARRIER.json','ROOT_READBACK.json','FINAL_COMPLETION.json']:
  p=old.BASE/n;bindings[str(p.relative_to(ROOT))]=sha(p)
 bindings[str((retention/'FINAL_COMPLETION.json').relative_to(ROOT))]=sha(retention/'FINAL_COMPLETION.json')
 bindings[str((BASE/'BASELINE_STANDBY_IDENTITY.json').relative_to(ROOT))]=sha(BASE/'BASELINE_STANDBY_IDENTITY.json')
 protected=read(old.BASE/'RUNTIME.json')['protected']
 write(BASE/'RUNTIME.json',dict(status='locked_before_new_GPU_GT',pins={**shared,**{n:sha(ROOT/n) for n in SCIENCE}},bindings=bindings,protected=protected,
  design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),GT_read=False,new_DINO_calls=0,time=time.time()))
 write(BASE/'SOURCE_STATE_CUSTODY_READBACK.json',dict(status='pass',parents=64,reference_complete_fits=64,
  LN_initial_coordinates=98304,Query_initial_coordinates=16384,source_bindings=len(bindings),CPU_only=True,GT_read=False,time=time.time()))
 status(BASE/'STATUS.json',dict(status='prepared_pending_actual_technical_qualification',done=0,total=8,CPU_only=True,GT_read=False,time=time.time()))
 verify()
