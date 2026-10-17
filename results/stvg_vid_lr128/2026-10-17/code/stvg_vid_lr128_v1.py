"""Two new Vid Student LR controls; exact sealed current-LR baseline reuse."""
from pathlib import Path
import copy,inspect,json,sys,time,types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
BASE=ROOT/'artifacts/stvg_vid_lr128_v1'
PAPER=ROOT/'artifacts/stvg_paper_t2_b_v1'
OLD=PAPER/'arms/B/trials/G32'
ARMS={'LR_HALF':.0135,'LR_QUARTER':.00675}
def put(p,d,replace=False):write(p,d,mutable=replace)
def verify():
 d=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<d['deadline_epoch']
 for p,h in d['pins'].items():assert sha(ROOT/p)==h,p
 return d
def setup():
 BASE.mkdir(exist_ok=False);assert read(OLD/'ROOT_READBACK.json')['status']=='pass'
 assert read(PAPER/'CPU_COMPLETION.json')['count']==256
 rr=copy.deepcopy(read(PAPER/'ROSTER_LOCK.json'));rr['records']=[r for r in rr['records'] if r['dataset']=='vidstg'];rr.update(count=128,per_domain={'vidstg':128});assert len(rr['records'])==128
 ids={r['probe_alias'] for r in rr['records']};put(BASE/'ROSTER_LOCK.json',rr)
 sources=copy.deepcopy(read(PAPER/'arms/B/SOURCE_CAPTURE_BARRIER.json'));sources.update(count=128,files=[r for r in sources['files'] if r['case_id'] in ids]);assert len(sources['files'])==128
 put(BASE/'SOURCE_CAPTURE_BARRIER.json',sources)
 baseline=copy.deepcopy(read(OLD/'GLOBAL_PREDICTION_BARRIER.json'));baseline.update(count=128,files=[r for r in baseline['files'] if r['case_id'] in ids]);assert len(baseline['files'])==128
 for r in baseline['files']:assert sha(ROOT/r['path'])==r['sha256']
 put(BASE/'BASELINE_BARRIER.json',baseline)
 for arm,lr in ARMS.items():
  a=BASE/'arms'/arm;b=a/'trials/G32';b.mkdir(parents=True);put(a/'ROSTER_LOCK.json',rr);put(a/'SOURCE_CAPTURE_BARRIER.json',sources)
  for name in ['REFERENCE_LOCK.json','TEMPORAL_LOCK.json']:
   d=copy.deepcopy(read(OLD/name));d.update(count=128,records=[r for r in d['records'] if r['case_id'] in ids]);assert len(d['records'])==128;put(b/name,d)
  d=copy.deepcopy(read(OLD/'DESIGN_LOCK.json'));cfg=copy.deepcopy(d['configs']);assert cfg['vidstg']['M2']['lr']==.027;cfg['vidstg']['M2']['lr']=lr
  d.update(count=128,cohort='same exposed core256 Vid128; not independent confirmation',scope=arm,configs=cfg,trial_id='G32');put(b/'DESIGN_LOCK.json',d)
  assert {k:v for k,v in cfg['vidstg']['M2'].items() if k!='lr'}=={k:v for k,v in read(OLD/'DESIGN_LOCK.json')['configs']['vidstg']['M2'].items() if k!='lr'}
  put(a/'SCOPE_CONFIG.json',dict(recipe='lite_fullstep',core_scope=str(PAPER.relative_to(ROOT)),label=arm))
  prior=read(OLD/'RUNTIME.json');pins=dict(prior['pins'])
  for p in [Path(__file__),BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',a/'ROSTER_LOCK.json',a/'SOURCE_CAPTURE_BARRIER.json',a/'SCOPE_CONFIG.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',b/'DESIGN_LOCK.json']:pins[str(p.relative_to(ROOT))]=sha(p)
  put(b/'RUNTIME.json',dict(status='locked',trial_id='G32',pins=pins,protected=prior['protected'],GT_read=False))
 put(BASE/'METHOD_LOCK.json',dict(status='locked',count=128,dataset='vidstg',arms={'LR_CURRENT':.027,**ARMS},only_changed_parameter='Spatial Adam lr',baseline='exact whole-query sealed T2+B reuse',same_Temporal_Caption_WHO_references_Gaussian32_seed_sigma_tau_steps_mu_acceptance_rollback=True,new_Teacher_calls=0,new_Temporal_fits=0,maximum_new_Spatial_fits=256,selection='largest mean vIoU on all exposed Vid128; ties within1e-9 prefer current then half then quarter; report all arms and intervals, not independent confirmation',no_automatic_paper_or_CURRENT_change=True,GT_read=False))
 paths=[Path(__file__),ROOT/'scripts/run_stvg_vid_lr128_v1.py',BASE/'ROSTER_LOCK.json',BASE/'SOURCE_CAPTURE_BARRIER.json',BASE/'BASELINE_BARRIER.json',BASE/'METHOD_LOCK.json',OLD/'ROOT_READBACK.json',OLD/'GLOBAL_PREDICTION_BARRIER.json',PAPER/'TEMPORAL_LOCK.json',ROOT/'methods/PAPER_METHOD.json',ROOT/'methods/CURRENT_METHOD.json',ROOT/'scripts/stvg_overnight_student_v1.py',ROOT/'scripts/run_stvg_instance_student32_v1.py',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py']
 put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=128,pins={str(p.relative_to(ROOT)):sha(p) for p in paths},deadline_epoch=time.time()+7200,maximum_new_fits=256,GT_read=False))
 put(BASE/'STATUS.json',dict(status='prepared',count=128,GT_read=False));print('VID_LR128_LOCKED_BASELINE128_NEW_ARMS2')
def configure(arm):
 assert arm in ARMS;verify()
 from scripts import stvg_overnight_student_v1 as s
 s.BASE=BASE/'arms'/arm;s.CORE=PAPER;s.N=128;s.CFG=read(s.BASE/'SCOPE_CONFIG.json')
 s.common=types.SimpleNamespace(BASE=BASE,verify=verify)
 s.verify=verify
 return s
def stage(arm,mode):
 s=configure(arm)
 if mode=='root':return s.root()
 from scripts import stvg_pipeline_speed_v1 as speed
 ns=dict(s.__dict__);ns['speed']=types.SimpleNamespace(BASE=speed.BASE,Engine=lambda label,enabled:speed.Engine('vid_lr128_'+label,enabled))
 exec(compile(inspect.getsource(s.worker),__file__,'exec'),ns);return ns['worker']()
def matching_root():
 import torch
 from scripts.stvg_pipeline_speed_v1 import compare_tree
 verify();old={r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']};checks=0
 for arm,lr in ARMS.items():
  b=BASE/'arms'/arm/'trials/G32';assert read(b/'ROOT_READBACK.json')['status']=='pass';bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert bar['status']=='complete' and bar['count']==128
  for f in bar['files']:
   assert sha(ROOT/f['path'])==f['sha256'];o=old[f['case_id']];assert sha(ROOT/o['path'])==o['sha256'];new=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);prior=torch.load(ROOT/o['path'],map_location='cpu',weights_only=False)
   compare_tree(new['reference_records'],prior['reference_records']);compare_tree(new['temporal_record'],prior['temporal_record'])
   a,z=new['fits']['M2'],prior['fits']['M2'];assert (a is None)==(z is None)
   if a is not None:
    assert a['config']['lr']==lr and {k:v for k,v in a['config'].items() if k!='lr'}=={k:v for k,v in z['config'].items() if k!='lr'}
    assert a['seed_key']==z['seed_key'];compare_tree(a['initial_state'],z['initial_state'])
    for key in ['state_before','rollout','gradient']:compare_tree(a['path'][0][key],z['path'][0][key],arm+'.'+key)
   checks+=1
 put(BASE/'MATCHED_QUALIFICATION_COMPLETION.json',dict(status='pass',count=128,whole_query_comparisons=checks,first_rollout_gradient_and_initial_state_exact=True,only_lr_changed=True,GT_read=False))
 put(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=128,arm_barriers={arm:sha(BASE/'arms'/arm/'trials/G32/GLOBAL_PREDICTION_BARRIER.json') for arm in ARMS},baseline_sha256=sha(BASE/'BASELINE_BARRIER.json'),GT_read=False))
def score():
 import gc,numpy as np,torch
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 from scripts.score_stvg_temporal_t2_v1 import subset_aggregate
 verify();assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='complete';assert read(BASE/'MATCHED_QUALIFICATION_COMPLETION.json')['status']=='pass'
 maps={'LR_CURRENT':{r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']}}
 for a in ARMS:maps[a]={r['case_id']:r for r in read(BASE/'arms'/a/'trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']}
 rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,bindings=truth._truths_after_seal(cases);off,ind,_=truth._metric_functions();rows=[];usage=[]
 for r in rr:
  cid=r['probe_alias'];case=cases[cid];preds={};reference_time=None
  for a,mm in maps.items():
   f=mm[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)
   if a=='LR_CURRENT':preds.update(Frozen=d['arms']['F'],T2_only=d['arms']['T']);reference_time=d['arms']['T']['interval']
   preds[a]=d['arms']['TS_M2'];assert preds[a]['interval']==reference_time;fit=d['fits']['M2'];usage.append(dict(case_id=cid,dataset='vidstg',arm=a,actual_fit=fit is not None,accepted=fit['optimizer_steps'] if fit else 0,proposed=fit['proposed_optimizer_steps'] if fit else 0));del d
  pp=PAPER/'spatial/E1/proposals'/(cid+'.json');support={w['source_grid_position'] for w in read(pp)['windows']} if pp.exists() else set();v={}
  for a,p in preds.items():
   m,_=metric._score_student_arm(p,case,targets[cid],support,off,ind);m.update(R_vIoU_at_0p3=float(m['m_vIoU']>=.3),R_vIoU_at_0p5=float(m['m_vIoU']>=.5));rows.append(dict(case_id=cid,dataset='vidstg',arm=a,metrics=m));v[a]=m
  assert all(v[a]['m_tIoU']==v['T2_only']['m_tIoU'] for a in maps);gc.collect()
 summary,paired=subset_aggregate(rows);means={a:next(s['mean'] for s in summary if s['arm']==a and s['metric']=='m_vIoU') for a in maps};best='LR_CURRENT'
 for a in ARMS:
  if means[a]>means[best]+1e-9:best=a
 put(BASE/'POSTSEAL_RESULTS.json',dict(status='complete',count=128,rows=rows,summary=summary,paired=paired,usage=usage,GT_bindings=bindings,development_selection=dict(selected=best,lr={'LR_CURRENT':.027,**ARMS}[best],means=means,not_independent_confirmation=True,paper_registry_unchanged=True),GT_read=True))
 put(BASE/'CPU_COMPLETION.json',dict(status='complete',count=128,result_sha256=sha(BASE/'POSTSEAL_RESULTS.json'),GT_read=True));print('VID_LR128_POSTSEAL_COMPLETE',best)
if __name__=='__main__':
 mode=sys.argv[1]
 if mode in ('setup','matching_root','score'):globals()[mode]()
 else:stage(sys.argv[2],mode)
