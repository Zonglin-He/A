"""Bounded CPU fixed tent trajectory actual OPD32."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_tent_layout_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/tent_layout_opd32'
PRE=ROOT/'artifacts/stvg_discriminative_dense32_v1';OLD=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';OLDP=ROOT/'results/stvg_error_state/2026-10-18/cosine_dense_opd32';LAY=ROOT/'artifacts/stvg_roi_layout_opd32_v1'
CODE=['vg_tta/stvg_tent_layout_opd32_v1.py','scripts/run_stvg_tent_layout_opd32_v1.py','scripts/continue_stvg_tent_layout_opd32_v1.py','protocols/stvg_tent_layout_opd32_v1.md']
VERIFIER=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
 global VERIFIER
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 from main_method.runtime import PinVerifier,resource_guard
 if VERIFIER is None:VERIFIER=PinVerifier(r['pins'])
 VERIFIER.verify(full=full);resource_guard()
 if live:assert time.time()<p['deadline_epoch']
 return p

def prepare():
 from vg_tta.stvg_tent_layout_opd32_v1 import properties,ARMS
 old=read(LAY/'PLAN_LOCK.json');assert read(LAY/'FINAL_COMPLETION.json')['status']=='complete';night=read(ROOT/'artifacts/stvg_opd_overnight_20261018_v1/STATE.json')
 D.mkdir(parents=True,exist_ok=True);P.mkdir(parents=True,exist_ok=True)
 put(D/'CPU_PROPERTIES.json',properties())
 p=dict(status='locked',records=old['records'],count=32,per_domain=16,configuration=old['configuration'],arms=ARMS,mechanism='fixed physical-time anchor-between tent6D joint dense Layout feedback; scoped independent marginal-amplitude and both projection controls',new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0,new_model_training=0,new_head=0,actual_update_branches=4,actual_fit_episodes=128,actual_rounds=256,changed_state='current-query tent6D or matched independent center state only; inactive latent axes excluded and frozen',MBR_and_E_sizes_and_planned_positions_fixed=True,no_parameter_search=True,all32_outputs_before_NEW_GT=True,no_intermediate_GT_gate=True,development_exposed=True,prior_GT_informed_mechanism_choice=True,independent_confirmation=False,CPU_seconds_cap=600,created_epoch=time.time(),deadline_epoch=min(time.time()+1800,night['deadline_epoch']),decision='Expand only if both-domain Joint_live minus E complete-v and unobserved actual-s positive, and Joint_live minus Fixed/Shuffle/Direct/matchedIndependent/bothProjections complete-v positive; CIs remain uncertainty')
 put(D/'PLAN_LOCK.json',p);pins={**read(LAY/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in CODE}}
 for rr in p['records']:
  f=str((LAY/'predictions'/(rr['alias']+'.pt')).relative_to(ROOT));pins[f]=sha(ROOT/f)
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));status('prepared',0,GT_read=False)

def measure():
 import torch
 from vg_tta.stvg_roi_layout_opd32_v1 import build
 from vg_tta.stvg_tent_layout_opd32_v1 import basis,noise,joint_fit,independent_fit,project,joint_feedback,FIT_ARMS
 torch.set_num_threads(4);p=verify(True);tick=time.perf_counter();receipts=[]
 for ix,r in enumerate(p['records']):
  verify(True);assert time.perf_counter()-tick<p['CPU_seconds_cap'];at=time.perf_counter();a=r['alias'];status('four_actual_fits_and_projection_controls',ix,alias=a,GT_read=False)
  x=load(PRE/'features'/(a+'.pt'));fb=load(PRE/'feedback'/(a+'.pt'))['feedback'];old=load(LAY/'predictions'/(a+'.pt'));E=x['E'].double();pos=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];bb=[f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0];context=build(x['maps'],x['valid'],pos,bb);B,scope=basis(x['frame_ids'],x['planned_positions'],x['interval']);active=B.sum(0)>0;eps=noise(a,active);cfg=p['configuration'];per=fb['permutation'];ft=time.perf_counter();fits={m:joint_fit(E,B,context,eps,cfg,m,per)for m in FIT_ARMS[:3]};fits['Independent_scoped']=independent_fit(E,B,context,fb['noise'],cfg);fitseconds=time.perf_counter()-ft
  assert torch.equal(fits['Joint_fixed']['path'][1]['boxes'],fits['Joint_live']['path'][1]['boxes']);assert torch.equal(fits['Joint_fixed']['rounds'][0]['feedback']['prob'],fits['Joint_live']['rounds'][0]['feedback']['prob'])
  ee=torch.cat(tuple(eps),0);rd=joint_feedback(E,B,context,ee,ee,cfg);best=int(rd['prob'].argmax());direct=rd['boxes'][:,best]if rd['informative']else E.clone();ip=project(E,B,fits['Independent_scoped']['final'],cfg['center_scale']);prev=project(E,B,torch.as_tensor(old['outputs']['On_policy_dense']),cfg['center_scale']);outputs=dict(E_MBR=E.numpy(),Joint_direct=direct.numpy(),Independent_projection=ip['boxes'].numpy(),Previous_projection=prev['boxes'].numpy(),**{m:f['final'].numpy()for m,f in fits.items()})
  for out in outputs.values():
   out=torch.as_tensor(out);assert torch.equal(out[:,2:],E[:,2:])and torch.equal(out[x['planned_positions']],E[x['planned_positions']])and torch.equal(out[~scope],E[~scope])
   if not pos:assert torch.equal(out,E)
  receipts.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],active=bool(pos),B=B,scope=scope,basis_rank=int(torch.linalg.matrix_rank(B)),active_axes=active,noise=eps,outputs=outputs,fits=fits,direct=rd,direct_best=best,projections=dict(independent=ip,previous=prev),permutation=per,GT_read=False,compute=dict(case_seconds=time.perf_counter()-at,four_fit_seconds=fitseconds,new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0))))
  print(a,ix+1,round(time.perf_counter()-tick,3),flush=True)
 verify(full=True);assert not torch.cuda.is_initialized();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=receipts,all32_before_NEW_GT=True,all_four_update_arms_actually_run=True,historical_target_GT_exposure=True,CPU_construction_and_four_fits_seconds=time.perf_counter()-tick,new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0));status('all32_eight_outputs_and_paths_sealed',32,GT_read=False)

def score():
    import torch,numpy as np
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.stvg_tent_layout_opd32_v1 import ARMS,FIT_ARMS
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT']and bar['all_four_update_arms_actually_run']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    strata=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')['rows'];strata={ds:{r['query_ordinal']:r for r in rr}for ds,rr in strata.items()};prior={r['alias']:r for r in read(OLDP/'RESULTS_PUBLIC.json')['rows']};rows=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));m=metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);s=strata[ds][q]
        for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
            x,y=m['values']['E_MBR'][k],prior[a]['values']['E_MBR'][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
        fits={}
        for arm,f in o['fits'].items():
            residual=(f['final'][:,:2]-o['outputs']['E_MBR'][:,:2])/(.25*o['outputs']['E_MBR'][:,2:])
            fits[arm]=dict(rounds=len(f['rounds']),state_norm=float(f['path'][-1]['state'].norm()),output_L2=float((f['final']-o['outputs']['E_MBR']).norm()),changed_actual_frames=int((f['final'][:,:2]!=o['outputs']['E_MBR'][:,:2]).any(1).sum()),residual_first_difference_L2=float(residual.diff(dim=0).norm()),residual_second_difference_L2=float(residual.diff(n=2,dim=0).norm()),teacher_ESS=[v.get('teacher_ESS',float((1/v['feedback']['prob'].square().sum(-1)).mean()))for v in f['rounds']],prior_ESS=[v.get('prior_ESS',float((1/v['feedback']['prior'].square().sum(-1)).mean()))for v in f['rounds']])
        rows.append(dict(alias=a,dataset=ds,active=o['active'],target_area_bin=s['bins']['target_area'],motion_bin=s['bins']['speed_sizes_sec'],query_form=s['query_form'],fits=fits,compute=o['compute'],basis_rank=o['basis_rank'],eligible_nodes=int((o['B'].sum(-1)>0).sum()),**m))
    eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('Joint_live',a)for a in ARMS[1:]if a!='Joint_live']
    def aggregate(rr):
        return dict(count=len(rr),parents=len(rr),active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in eps}for a in ARMS},
            paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in eps if k!='t'}for a,b in contrasts},
            interpolation_v_contributions_pp={a:{support:mean([r['interpolation_effects'][a][support]['v_contribution_pp']for r in rr])for support in ['observed_actual','unobserved_actual','benchmark_interpolated']}for a in ARMS[1:]})
    groups={ds:aggregate([r for r in rows if r['dataset']==ds])for ds in ['hc2','vidstg']};subgroups={ds:{key:{str(v):aggregate([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds},key=str)}for key in ['target_area_bin','motion_bin','active']}for ds in groups}
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,strata=subgroups,all32_outputs_before_NEW_GT=True,cohort_historically_development_exposed=True,independent_confirmation=False,all_four_update_arms_actually_run=True,config=p['configuration'],MBR_fixed=True,no_parameter_search=True))
    put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in groups.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
