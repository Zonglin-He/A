"""One bounded frozen-coarse ordered ROI feedback actual OPD experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_roi_layout_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-19/roi_layout_opd32'
PRE=ROOT/'artifacts/stvg_discriminative_dense32_v1';OLD=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';OLDP=ROOT/'results/stvg_error_state/2026-10-18/cosine_dense_opd32'
CODE=['vg_tta/stvg_roi_layout_opd32_v1.py','scripts/run_stvg_roi_layout_opd32_v1.py','scripts/continue_stvg_roi_layout_opd32_v1.py','protocols/stvg_roi_layout_opd32_v1.md']
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
 from vg_tta.stvg_roi_layout_opd32_v1 import properties,ARMS
 p=read(OLD/'PLAN_LOCK.json');assert read(OLD/'FINAL_COMPLETION.json')['status']=='complete';assert read(ROOT/'artifacts/stvg_reference_seed32_v1/FINAL_COMPLETION.json')['status']=='complete';night=read(ROOT/'artifacts/stvg_opd_overnight_20261018_v1/STATE.json')
 files=CODE+['methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json']
 for r in p['records']:files +=[str((PRE/k/(r['alias']+'.pt')).relative_to(ROOT))for k in ['features','feedback']]+[str((OLD/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))]
 put(D/'CPU_PROPERTIES.json',properties())
 put(D/'PLAN_LOCK.json',dict(status='locked',records=p['records'],count=32,per_domain=16,configuration=p['configuration'],arms=ARMS,
  mechanism='Position-preserving3x3 ROI-bin corresponding-patch cosine, then equal anchor mean; matched unordered all-bin-pairs cosine control',
  feature='original coarse input_proj only; no Fine gate reopening; no training',new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0,new_model_training=0,new_head=0,
  actual_update_branches=4,actual_fit_episodes=128,actual_rounds=256,changed_state='existing per-query independent unobserved-frame center variables only; reset each query',
  MBR_and_E_sizes_and_planned_positions_fixed=True,no_parameter_search=True,all32_outputs_before_NEW_GT=True,no_intermediate_GT_gate=True,development_exposed=True,prior_GT_informed_mechanism_choice=True,independent_confirmation=False,
  reviewer='GPT-6.1 Sol; same reviewing subagent for next round',CPU_seconds_cap=600,created_epoch=time.time(),deadline_epoch=min(time.time()+1800,night['deadline_epoch']),
  source_baseline_commit='30746bcc741816f9978efb84dbfce08452817260',fine_qualification_commit='5411061697410d6dc27fe4f18c594c36b4e5484f',
  decision='Expand only if both-domain Live minus E complete-v and unobserved actual-s positive, and Live minus Fixed/Shuffle/Unordered-live/Layout direct complete-v positive; CIs remain uncertainty; no parameter search'))
 pins={**read(OLD/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in files}};put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));status('prepared',0,GT_read=False)


def measure():
 import torch,numpy as np
 from vg_tta.stvg_roi_layout_opd32_v1 import build,direct,fit,FIT_ARMS
 torch.set_num_threads(4);p=verify(True);tick=time.perf_counter();files=[]
 for ix,r in enumerate(p['records']):
  verify(True);assert time.perf_counter()-tick<p['CPU_seconds_cap'];at=time.perf_counter();a=r['alias'];status('ordered_ROI_feedback_and_four_actual_fits',ix,alias=a,GT_read=False)
  x=load(PRE/'features'/(a+'.pt'));fb=load(PRE/'feedback'/(a+'.pt'))['feedback'];old=load(OLD/'predictions'/(a+'.pt'));E=x['E'].double();positions=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];boxes=[f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]
  context=build(x['maps'],x['valid'],positions,boxes);eps=fb['noise'];perm=fb['permutation'];cfg=p['configuration']
  ctxreceipt=save(D/'context'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],templates=context['templates'],weights=context['weights'],positions=context['positions'],frame_ids=x['frame_ids'],GT_read=False))
  rd=direct(E,context,eps,cfg,x['planned_positions']);assert torch.equal(rd['candidates'],old['direct']['candidates']);pd=direct(E,context,eps,cfg,x['planned_positions'],pooled=True);ft=time.perf_counter();fits={mode:fit(E,context,eps,cfg,x['planned_positions'],mode,perm)for mode in FIT_ARMS};fit_seconds=time.perf_counter()-ft
  fixed,on=fits['Fixed_dense'],fits['On_policy_dense'];assert torch.equal(fixed['path'][1]['boxes'],on['path'][1]['boxes'])and torch.equal(fixed['rounds'][0]['feedback']['prob'],on['rounds'][0]['feedback']['prob'])
  outputs=dict(E_MBR=E.numpy(),Layout_readout=rd['boxes'].numpy(),Unordered_readout=pd['boxes'].numpy(),**{mode:f['final'].numpy()for mode,f in fits.items()},
   Forward_readout=old['outputs']['Cosine_readout'],Forward_Fixed=old['outputs']['Fixed_dense'],Forward_On=old['outputs']['On_policy_dense'],Forward_Shuffled=old['outputs']['Shuffled_OPD'])
  files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=r['dataset'],frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],active=bool(positions),outputs=outputs,fits=fits,direct=rd,pooled=pd,permutation=perm,context_receipt=ctxreceipt,GT_read=False,pretrained_weights_frozen=True,
   compute=dict(case_seconds=time.perf_counter()-at,four_fit_seconds=fit_seconds,new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0))))
  print(a,ix+1,round(time.perf_counter()-tick,3),flush=True)
 verify(full=True);assert not torch.cuda.is_initialized();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_before_NEW_GT=True,historical_target_GT_exposure=True,CPU_construction_and_four_fits_seconds=time.perf_counter()-tick,
  new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0,all_four_update_arms_actually_run=True));status('all32_eleven_outputs_and_paths_sealed',32,GT_read=False)


def score():
    import torch,numpy as np
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.stvg_roi_layout_opd32_v1 import ARMS,FIT_ARMS
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
            fits[arm]=dict(rounds=len(f['rounds']),updated_frames=[v['updated_frames']for v in f['rounds']],nonzero_gradient_frames=[v['nonzero_gradient_frames']for v in f['rounds']],
                actual_center_change_frames=[int((f['path'][i+1]['boxes'][:,:2]!=f['path'][i]['boxes'][:,:2]).any(1).sum())for i in range(2)],
                eta_norm=float(f['path'][-1]['eta'].norm()),step_KL=[v['KL']for v in f['rounds']],output_step_L2=[v['output_step_L2']for v in f['rounds']],
                second_sampling_center_norm=float(f['rounds'][1]['sampling_center'].norm()),teacher_ESS=[float((1/v['feedback']['prob'].square().sum(-1)).mean())for v in f['rounds']],
                output_delta_from_E_L2=float((f['final']-f['path'][0]['boxes']).norm()))
        rows.append(dict(alias=a,dataset=ds,active=o['active'],target_area_bin=s['bins']['target_area'],motion_bin=s['bins']['speed_sizes_sec'],query_form=s['query_form'],fits=fits,compute=o['compute'],**m))
    eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('On_policy_dense',a)for a in ['Fixed_dense','Shuffled_OPD','Unordered_OPD','Layout_readout','Forward_On','Forward_Fixed','Forward_readout']]+[('Layout_readout','Unordered_readout')]
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
