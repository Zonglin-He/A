"""Bounded six-arm supervision experiment; immutable prior evidence and production."""
import sys,time,collections,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts import stvg_opd_real_forward_task_common_v1 as previous
prior=previous.prior
BASE=ROOT/'artifacts/stvg_opd_admission_association_v1'
OUT=ROOT/'results/stvg_opd_admission_association/2026-10-13'
ARMS=['H_Top1','H_Path','R_Top1','R_Path','R_Add','R_Shuffle'];BOOT=10000;SEED=20261013;EPS=1e-12
SCIENCE=['protocols/stvg_opd_admission_association_v1.md','vg_tta/stvg_opd_admission_association_v1.py','scripts/stvg_opd_admission_association_common_v1.py','scripts/run_stvg_opd_admission_association_v1.py','scripts/score_stvg_opd_admission_association_v1.py','scripts/audit_stvg_opd_admission_association_v1.py','scripts/review_stvg_opd_admission_association_independent_v1.py','scripts/continue_stvg_opd_admission_association_v1.py']

def verify():
 previous.verify();r=read(BASE/'RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
 pins=dict(r['pins'])
 for p in sorted(BASE.glob('*/REVISION_RUNTIME.json'),key=lambda p:read(p)['time']):
  v=read(p);assert v['original_runtime_sha256']==sha(BASE/'RUNTIME.json');pins.update(v['pin_overrides'])
 for n,h in {**pins,**r['bindings']}.items():assert sha(ROOT/n)==h,n
 prior.standby();return r

def evidence_digest(fixed):return prior.old.digest(fixed['rows'])

def logical_path(index,arm,phase='formal',repeat=0):return BASE/phase/f'{index:05}'/(arm+f'_repeat{repeat}.json')

def resolve(index,arm,phase='formal',repeat=0):
 p=logical_path(index,arm,phase,repeat);receipt=read(p);q=ROOT/receipt['payload_path'];assert sha(q)==receipt['payload_sha256']
 z=load(q);selection=receipt['selection'];assert evidence_digest(selection)==evidence_digest(z['fixed_evidence'])
 z.update(arm=arm,logical_arm=arm,selection=selection,canonical_arm=z['arm'],canonical_reused=receipt['canonical_reused'],
  canonical_payload_sha256=receipt['payload_sha256'],canonical_payload_path=receipt['payload_path'],canonical_GT_used=z['fit']['GT_used'],logical_GT_assisted=selection['GT_assisted'])
 return z

def prepare():
 import torch
 from vg_tta.stvg_opd_admission_association_v1 import contracts,pools_from_expert,select,fixed_support
 from scripts.run_decota_paper_main_v1 import unpack_expert
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
 torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not BASE.exists();previous.verify()
 assert read(previous.BASE/'FINAL_COMPLETION.json')['status']=='complete'
 cc=contracts();rs=copy.deepcopy(read(previous.BASE/'DESIGN_LOCK.json')['records']);bindings={};counts=collections.Counter();planned=0
 p7=ROOT/'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1';pr=read(p7/'DESIGN_LOCK.json')['records'];BASE.mkdir();OUT.mkdir(parents=True)
 for r in rs:
  inp=normalized(load(ROOT/r['originals'][0]['input_path']));ex=unpack_expert(inp['expert']);ref=load(ROOT/r['A_path'])['fit']
  from scripts.run_decota_paper_main_v1 import read_row
  pools=pools_from_expert(ex,inp['frame_ids']);key=read_row(r['dataset'],r['query_ordinal'])['key']
  selected=select(pools,key);en=fixed_support(ex,ref['before'],selected['arms']['H_Top1']);orig=Energy(ex,ref['before'],'top1')
  assert en.frame_metadata==orig.frame_metadata==ref['frame_metadata'] and [f[0] for f in en.frames]==ref['positions']
  assert all(torch.equal(a[1],b[1]) for a,b in zip(en.frames,orig.frames))
  assert r['visited']==inp['expert']['actual_observation_positions']==[p['position'] for p in pools]
  match=[q for q in pr if q['dataset']==r['dataset'] and q['query_ordinal']==r['query_ordinal'] and q['order']=='order1' and q['previous_P6']];assert len(match)==1
  tp=p7/'P7_A/readouts'/f"{match[0]['public_cell']:05}.pt";tz=load(tp);tb=tz['temporal_binding'];packet=load(ROOT/tb['path']);cap=packet['capture']
  assert sha(ROOT/tb['path'])==tb['sha256'] and cap['source_model_state_sha256']==inp['source_model_state_sha256'] and packet['pixel_sha256']==inp['pixel_sha256'] and cap['frame_ids']==inp['frame_ids']
  assert list(tz['intervals']['T0'])==list(inp['interval'])
  r.update(intervals=dict(I0=list(tz['intervals']['T0']),I1=list(tz['intervals']['T1'])),time_readout_path=str(tp.relative_to(ROOT)),time_readout_sha256=sha(tp),key=key)
  p=BASE/'selections'/f"{r['index']:05}.json";write(p,selected);r.update(selection_path=str(p.relative_to(ROOT)),selection_sha256=sha(p))
  for n,h in [(r['A_path'],r['A_sha256']),(r['pair_path'],r['pair_sha256']),(r['selection_path'],r['selection_sha256']),(str(tp.relative_to(ROOT)),sha(tp)),(tb['path'],tb['sha256'])]:assert sha(ROOT/n)==h;bindings[n]=h
  group={evidence_digest(v) for v in selected['arms'].values()};planned+=len(group)
  counts.update(dict(observation_positions=len(pools),candidates=sum(len(p['boxes']) for p in pools),H_positions=len(selected['arms']['H_Top1']['rows']),R_positions=len(selected['arms']['R_Top1']['rows'])))
  for a in ARMS:counts[a+'_different_H_Top1_queries']+=evidence_digest(selected['arms'][a])!=evidence_digest(selected['arms']['H_Top1'])
  for p0 in pools:counts[p0['reason']]+=1
  print('ADMISSION_SELECTION',r['index'],len(group),flush=True)
 write(BASE/'CPU_CONTRACTS.json',dict(cc,time=time.time()))
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_GPU_GT',records=rs,parents=64,arms=ARMS,qualification_indices=[2,20,36,44,60],
  qualification_logical_fits=60,qualification_logical_rounds=1320,qualification_captures=10,logical_complete_fits=384,logical_complete_rounds=9600,
  planned_physical_formal_fits=planned,planned_formal_captures=64,Oracle_logical_fits=64,Oracle_logical_rounds=1600,
  primary='I1_dense_v',secondary='I0_dense_v',bootstrap=BOOT,seed=SEED,historical_exposure=True,GT_before_B_global=False,
  automatic_expansion=False,production_promoted=False,monitor_recreated=False,time=time.time()))
 write(BASE/'CPU_SELECTION_READBACK.json',dict(status='pass',counts=dict(counts),parents=64,planned_unique_physical_formal_fits=planned,
  original_H_exact_support=True,original_pools_unchanged=True,I1_source_pixel_frames_bound=True,CPU_only=True,GT_read=False,time=time.time()))
 write(BASE/'BASELINE_STANDBY_IDENTITY.json',read(prior.BASE/'BASELINE_STANDBY_IDENTITY.json'))
 write(BASE/'RUNTIME.json',dict(status='locked_before_new_GPU_GT',pins={n:sha(ROOT/n) for n in SCIENCE},bindings=bindings,
  design_sha256=sha(BASE/'DESIGN_LOCK.json'),GT_read=False,new_DINO_calls=0,time=time.time()))
 write(BASE/'SOURCE_STATE_CUSTODY_READBACK.json',dict(status='pass',parents=64,reference_fits=64,I1_caches=64,GT_read=False,CPU_only=True,time=time.time()))
 status(BASE/'STATUS.json',dict(status='prepared_pending_actual_qualification',done=0,total=60,GT_read=False,time=time.time()));verify()
