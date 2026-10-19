"""Finite matched A/B OPD from existing exposed development caches only."""
import os,sys,time,hashlib,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_coordinate_spatial_opd64_v1';P=ROOT/'results/stvg_error_state/2026-10-19/coordinate_spatial_opd64'
ARMS=['E_MBR','A_Fixed','A_Live','B_Fixed','B_Live']
ENDPOINTS=['v','t','s','unobserved_actual_s','observed_actual_s','benchmark_interpolated_s']
CONTRASTS=[(a,'E_MBR')for a in ARMS[1:]]+[('B_Live','A_Live'),('B_Live','B_Fixed'),('A_Live','A_Fixed'),('B_Fixed','A_Fixed')]
VER=None

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 pins=dict(r['pins'])
 if (D/'RECOVERY_LOCK.json').exists():
  fix=read(D/'RECOVERY_LOCK.json');assert sha(D/'RUNTIME_LOCK.json')==fix['original_runtime_sha256'] and sha(D/'PLAN_LOCK.json')==fix['original_plan_sha256'];pins.update(fix['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def prepare():
 from vg_tta.stvg_coordinate_spatial_opd64_v1 import properties
 assert not D.exists();D.mkdir();P.mkdir();records=read(ROOT/'artifacts/stvg_pair_direction_dev64_v1/PLAN_LOCK.json')['records'];assert len(records)==len({r['parent']for r in records})==64
 for r in records:
  a=r['alias'];r['feature_path']=f'artifacts/stvg_fine_feature32_v1/features/{a}.pt'if r['panel']=='original'else f'artifacts/stvg_fine_step_confirm32_v1/features/{a}.pt'
  r['fixed_cache_path']=None if a.startswith('NH')else r['path'];r['fixed_cache_arm']=None if a.startswith('NH')else('Fine_fixed'if a.startswith('NV')else'Fixed_step10'if r['dataset']=='hc2'else'Fixed_step05')
 files=['scripts/run_stvg_coordinate_spatial_opd64_v1.py','vg_tta/stvg_coordinate_spatial_opd64_v1.py','protocols/stvg_coordinate_spatial_opd64_v1.md','vg_tta/stvg_roi_layout_opd32_v1.py','vg_tta/stvg_fine_roi_scoped_opd32_v1.py','vg_tta/stvg_discriminative_dense32_v1.py','configs/stvg_fine_center_dataset_candidate_v1.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json','artifacts/stvg_fine_dataset_formal64_v1/GLOBAL_PREDICTION_BARRIER.json','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','vg_tta/tastvg_oracle_event5_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py']+list({r[k]for r in records for k in ['path','feature_path','result_path']})
 plan=dict(status='locked',records=records,count=64,arms=ARMS,contrasts=CONTRASTS,configuration=dict(temperature=.1,center_scale=.25,rounds=2,candidates_per_round=17,step_size_by_dataset=dict(hc2=1.,vidstg=.5),grid=64,outside_atom=True,epsilon_mixture=0.,B_step_calibration=False),user_approved_outside_amendment=True,new_models_GPU_DINO_training=0,development_exposed=True,formal64_excluded=True,CPU_phase_cap_seconds=900,created_epoch=time.time(),all_outputs_before_GT_required=True)
 put(D/'PLAN_LOCK.json',plan);put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));put(P/'CPU_PROPERTIES.json',properties());print('locked64 and properties passed',flush=True)

def measure():
 import torch,gc
 from vg_tta.stvg_roi_layout_opd32_v1 import build
 from vg_tta.stvg_fine_roi_scoped_opd32_v1 import scoped_fit
 from vg_tta.stvg_coordinate_spatial_opd64_v1 import fit
 torch.set_num_threads(4);p=verify(True);start=time.perf_counter();files=[];counts=collections.Counter();A_seconds=0.;B_seconds=0.
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;a=r['alias'];x=load(ROOT/r['feature_path']);old=load(ROOT/r['path']);E=x['E'].double();assert torch.equal(E,torch.as_tensor(old['outputs']['E_MBR']));ctx=build(x['maps'],x['valid'],x['positions'],torch.stack([torch.as_tensor(b)for b in x['boxes']])if len(x['boxes'])else[]);eps=old['noise'];allowed=old['allowed'];cfg=dict(p['configuration'],step_size=r['step']);fits={'A_Live':old['fits'][r['arm']]}
  if r['fixed_cache_path']:fits['A_Fixed']=old['fits'][r['fixed_cache_arm']];counts['A_cached_fits']+=2
  else:
   tick=time.perf_counter();fits['A_Fixed']=scoped_fit(E,ctx,eps,cfg,x['planned_positions'],'Fixed_dense',list(range(len(E))),allowed);A_seconds+=time.perf_counter()-tick;counts['A_new_fits']+=1;counts['A_cached_fits']+=1
  cache={}
  for arm in ['A_Fixed','A_Live']:
   f=fits[arm];assert len(f['rounds'])==2 and torch.equal(f['path'][0]['boxes'],E)
   for t in f['rounds']:
    key=hashlib.sha256(t['candidates'].numpy().tobytes()+eps[t['step']].numpy().tobytes()).hexdigest();cache[key]=t['feedback']
  tick=time.perf_counter()
  for arm,mode in [('B_Fixed','Fixed'),('B_Live','Live')]:fits[arm]=fit(E,ctx,eps,cfg,x['planned_positions'],mode,allowed,cache,counts)
  B_seconds+=time.perf_counter()-tick
  for f in fits.values():
   assert torch.equal(f['rounds'][0]['candidates'],fits['A_Live']['rounds'][0]['candidates'])and torch.equal(f['rounds'][0]['feedback']['prob'],fits['A_Live']['rounds'][0]['feedback']['prob'])
   assert torch.equal(f['final'][~allowed],E[~allowed])and torch.equal(f['final'][:,2:],E[:,2:])
  for route in ['A','B']:assert torch.equal(fits[route+'_Fixed']['path'][1]['boxes'],fits[route+'_Live']['path'][1]['boxes'])
  outputs=dict(E_MBR=E.numpy(),**{arm:f['final'].numpy()for arm,f in fits.items()});payload=dict(alias=a,dataset=r['dataset'],panel=r['panel'],frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],allowed=allowed,noise=eps,active=bool(x['positions']),reference_count=len(x['positions']),fits=fits,outputs=outputs,GT_read=False,configuration=cfg,feature_path=r['feature_path'],feature_sha256=sha(ROOT/r['feature_path']))
  files.append(save(D/'predictions'/(a+'.pt'),payload));print(a,'all5 sealed',round(time.perf_counter()-start,2),flush=True);del x,old,ctx,fits,cache,payload;gc.collect()
 verify(True);cost=dict(CPU_seconds=time.perf_counter()-start,A_new_fit_seconds=A_seconds,B_fit_seconds=B_seconds,**dict(counts),actual_new_student_fits=counts['A_new_fits']+128,actual_new_round_updates=2*(counts['A_new_fits']+128),cached_A_fits=counts['A_cached_fits'],total_arm_round_slots=512,total_Source_captures_completed=0,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_training=0,new_GPU_seconds=0,old_feature_capture_and_DINO_cost_excluded=True)
 put(P/'COST_BREAKDOWN.json',cost);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=files,all_outputs_before_GT=True,CPU_seconds=cost['CPU_seconds'],new_GPU_models_experts_training=0));print('global seal64 complete',flush=True)

def aggregate(rows):
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 return dict(parents=len(rows),active=sum(r['active']for r in rows),means={a:{e:mean([r['values'][a][e]for r in rows])for e in ENDPOINTS}for a in ARMS},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rows])for e in ENDPOINTS}for a,b in CONTRASTS})

def score():
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_GT']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 start=time.perf_counter();gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];mechanisms=[]
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;x=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];j=r['query_ordinal'];m=metrics(x,plans[ds][j],gt[ds][0][j],gt[ds][1][j]);old=next(z for z in read(ROOT/r['result_path'])['rows']if z['alias']==r['alias'])
  for a,b in [('E_MBR','E_MBR'),('A_Live',r['arm'])]:
   for e in ENDPOINTS:
    u,v=m['values'][a][e],old['values'][b][e];assert u is None and v is None or abs(u-v)<1e-10
  rows.append(dict(alias=r['alias'],dataset=ds,panel=r['panel'],active=x['active'],references=x['reference_count'],**m));armstats={}
  for arm,f in x['fits'].items():
   ts=[]
   for t in f['rounds']:
    active=x['allowed']&t['feedback']['informative'];grad=t['gradient'];rec=dict(step=t['step'],eligible_nodes=int(active.sum()),nonzero_gradient_nodes=int((grad.abs().amax(-1)>1e-12).sum()),gradient_norm_sum=float(grad.norm(dim=-1).sum()),eta_step_norm_sum=float((t['eta_after']-t['eta_before']).norm(dim=-1).sum()),output_step_L2=t['output_step_L2'],KL=t['KL'],outside_valid_candidates=t.get('outside_valid_candidates'),outside_active_candidates=t.get('outside_active_candidates'))
    if arm.startswith('B'):
     enc=t['spatial_encoding'];deg=((enc['x']-enc['x'][:,:1]).abs().amax((1,2))<1e-14)&((enc['y']-enc['y'][:,:1]).abs().amax((1,2))<1e-14)&(enc['outside']==enc['outside'][:,:1]).all(1);rec['identical_all_candidate_kernels_active_nodes']=int((deg&active).sum());rec['max_target_mass_error']=float((t['spatial_target'].sum(-1)-1).abs().max());rec['max_student_mass_error']=float((t['spatial_student'].sum(-1)-1).abs().max())
    ts.append(rec)
   armstats[arm]=dict(status='complete',rounds=len(ts),changed_actual_nodes=int((f['final']-torch.as_tensor(x['outputs']['E_MBR'])).abs().amax(-1).gt(1e-12).sum()),rounds_summary=ts)
  mechanisms.append(dict(alias=r['alias'],dataset=ds,panel=r['panel'],arms=armstats))
 groups={pan:{ds:aggregate([r for r in rows if r['dataset']==ds and(pan=='combined'or r['panel']==pan)])for ds in gt}for pan in ['original','selection','combined']}
 put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=64,arms=ARMS,groups=groups,rows=rows,all_outputs_before_GT=True,formal64_excluded=True,development_exposed=True,independent_confirmation=False,actual_coordinate_and_spatial_OPD=True));put(P/'MECHANISM_PUBLIC.json',dict(status='complete',rows=mechanisms,raw_candidate_coordinates_spatial_maps_and_parentIDs_excluded=True));put(D/'SCORE_COMPLETION.json',dict(status='complete',count=64,CPU_seconds=time.perf_counter()-start));print('score64 complete',flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
