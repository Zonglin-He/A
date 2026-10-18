"""Seal all32 cached cosine-policy outputs before new offline GT evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_cosine_dense_opd32_v1';P=ROOT/'results/stvg_error_state/2026-10-18/cosine_dense_opd32';PRE=ROOT/'artifacts/stvg_discriminative_dense32_v1';OLDP=ROOT/'results/stvg_error_state/2026-10-18/discriminative_dense32'
CODE=['vg_tta/stvg_cosine_dense_opd32_v1.py','scripts/run_stvg_cosine_dense_opd32_v1.py','scripts/audit_stvg_cosine_dense_opd32_v1.py','scripts/continue_stvg_cosine_dense_opd32_v1.py','protocols/stvg_cosine_dense_opd32_v1.md']
VERIFIER=None

def status(stage,count,**kw):put(D/'STATUS.json',dict(status='running',stage=stage,count=count,total=32,pid=os.getpid(),epoch=time.time(),**kw),True)
def verify(live=False,full=False):
    global VERIFIER
    p=read(D/'PLAN_LOCK.json');l=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==l['plan_sha256']
    from main_method.runtime import PinVerifier,resource_guard
    if VERIFIER is None:VERIFIER=PinVerifier(l['pins'])
    VERIFIER.verify(full=full);resource_guard()
    if live:assert time.time()<p['deadline_epoch']
    assert read(OLDP/'GATE.json')['passed']is False
    return p

def prepare():
    import torch
    from vg_tta.stvg_cosine_dense_opd32_v1 import contract,ARMS
    old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
    cfg={k:old['configuration'][k]for k in ['temperature','center_scale','step_size','rounds','candidates_per_round']};assert cfg==dict(temperature=.1,center_scale=.25,step_size=.5,rounds=2,candidates_per_round=17)
    put(D/'PLAN_LOCK.json',dict(status='locked',count=32,per_domain=16,records=old['records'],configuration=cfg,arms=ARMS,source_baseline_commit='fd0388f658801e405f70d6d8eb9814b41dc70211',
        source_checkpoints_unchanged=True,cohort_development_exposed=True,independent_confirmation=False,previous_target_GT_informed_teacher_choice=True,ridge_failed_gate_preserved=True,
        no_scientific_pre_gate_this_run=True,all_three_update_arms_required=True,all32_outputs_before_NEW_GT=True,MBR_and_E_sizes_and_planned_positions_fixed=True,
        new_GPU_or_model_or_DINO_calls=0,no_parameter_search=True,CPU_measurement_seconds_cap=600,created_epoch=time.time(),deadline_epoch=time.time()+3600))
    files=CODE+[str((OLDP/'GATE.json').relative_to(ROOT)),str((OLDP/'PHASE1_RESULTS_PUBLIC.json').relative_to(ROOT))]
    for r in old['records']:files +=[str((PRE/k/(r['alias']+'.pt')).relative_to(ROOT))for k in ['features','feedback']]
    pins={**read(PRE/'RUNTIME_LOCK.json')['pins'],**{f:sha(ROOT/f)for f in set(files)}};put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=pins));put(D/'CPU_PROPERTIES.json',contract());assert not torch.cuda.is_initialized();status('prepared',0,GT_read=False)

def measure():
    import torch
    from vg_tta.stvg_cosine_dense_opd32_v1 import direct,fit,FIT_ARMS
    from vg_tta.stvg_soft_correspondence_v1 import propagate
    p=verify(True);torch.set_num_threads(4);start=time.perf_counter();files=[];qualification=[];domain_qualified=set()
    for ix,r in enumerate(p['records']):
        verify(True);assert time.perf_counter()-start<p['CPU_measurement_seconds_cap'];tick=time.perf_counter();a=r['alias'];ds=r['dataset'];status('cached_cosine_three_real_fits',ix,alias=a,GT_read=False)
        x=load(PRE/'features'/(a+'.pt'));old=load(PRE/'feedback'/(a+'.pt'));f=old['feedback'];E=x['E'].double();soft=f['cosine_all_soft_maps'];eps=f['noise'];perm=f['permutation'];valid=x['valid'];planned=x['planned_positions'];cfg=p['configuration']
        assert f['cosine_anchor_positions']==[g['position']for g,k in zip(x['frames'],x['selected'])if k>=0];assert len(soft)==sum(k>=0 for k in x['selected'])
        rd=direct(E,soft,valid,eps,cfg,planned);assert torch.equal(rd['candidates'],f['candidates']);err=float((rd['feedback']['prob']-f['cosine']).abs().max());assert err<1e-12;assert torch.equal(rd['feedback']['rewards'],f['cosine_all_rewards'])
        if ds not in domain_qualified and len(soft):
            boxes=[torch.tensor(g['boxes'][k])for g,k in zip(x['frames'],x['selected'])if k>=0];ap=f['cosine_anchor_positions']
            # Target-row derangement equals shuffling raw features BEFORE matching.
            replay=propagate(x['maps'][perm],valid[perm],ap,boxes,.07);pe=float((replay-soft[:,perm]).abs().max());assert pe<1e-7;qualification.append(dict(dataset=ds,raw_target_feature_shuffle_equivalence_error=pe));domain_qualified.add(ds)
        fit_start=time.perf_counter();fits={arm:fit(E,soft,valid,eps,cfg,planned,arm,perm)for arm in FIT_ARMS};fit_seconds=time.perf_counter()-fit_start
        fixed,on=fits['Fixed_dense'],fits['On_policy_dense'];assert torch.equal(fixed['path'][1]['boxes'],on['path'][1]['boxes'])and torch.equal(fixed['rounds'][0]['gradient'],on['rounds'][0]['gradient'])and torch.equal(fixed['rounds'][0]['feedback']['prob'],on['rounds'][0]['feedback']['prob'])
        outputs=dict(E_MBR=E.numpy(),Cosine_readout=rd['boxes'].numpy(),**{k:v['final'].numpy()for k,v in fits.items()})
        files.append(save(D/'predictions'/(a+'.pt'),dict(alias=a,dataset=ds,frame_ids=x['frame_ids'],planned_positions=planned,interval=x['interval'],active=bool(len(soft)),
            outputs=outputs,fits=fits,direct=rd,initial_cache_probability_error=err,permutation=perm,GT_read=False,pretrained_weights_frozen=True,
            compute=dict(case_seconds=time.perf_counter()-tick,three_fit_seconds=fit_seconds,new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0))))
        print(a,ix+1,round(time.perf_counter()-start,3),flush=True)
    verify(full=True);assert not torch.cuda.is_initialized();put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all32_before_NEW_GT=True,historical_target_GT_exposure=True,
        CPU_construction_and_three_fits_seconds=time.perf_counter()-start,new_GPU_seconds=0,new_Source_captures=0,new_DINO_calls=0,raw_feature_shuffle_qualifications=qualification,all_three_arms_actually_run=True));status('all32_five_outputs_sealed_before_NEW_GT',32,GT_read=False)

def score():
    import torch,numpy as np
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_stvg_dino_local_residual32_v1 import metrics,mean
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from vg_tta.stvg_cosine_dense_opd32_v1 import ARMS,FIT_ARMS
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT']and bar['all_three_arms_actually_run']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    strata=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')['rows'];strata={ds:{r['query_ordinal']:r for r in rr}for ds,rr in strata.items()};prior={r['alias']:r for r in read(OLDP/'PHASE1_RESULTS_PUBLIC.json')['rows']};rows=[]
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
    eps=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];contrasts=[(a,'E_MBR')for a in ARMS[1:]]+[('On_policy_dense',a)for a in ['Fixed_dense','Shuffled_OPD','Cosine_readout']]
    def aggregate(rr):
        return dict(count=len(rr),parents=len(rr),active=sum(r['active']for r in rr),metrics={a:{k:mean([r['values'][a][k]for r in rr])for k in eps}for a in ARMS},
            paired={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else 100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in eps if k!='t'}for a,b in contrasts},
            interpolation_v_contributions_pp={a:{support:mean([r['interpolation_effects'][a][support]['v_contribution_pp']for r in rr])for support in ['observed_actual','unobserved_actual','benchmark_interpolated']}for a in ARMS[1:]})
    groups={ds:aggregate([r for r in rows if r['dataset']==ds])for ds in ['hc2','vidstg']};subgroups={ds:{key:{str(v):aggregate([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds},key=str)}for key in ['target_area_bin','motion_bin','active']}for ds in groups}
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,strata=subgroups,all32_outputs_before_NEW_GT=True,cohort_historically_development_exposed=True,independent_confirmation=False,ridge_gate_unchanged_failed=True,all_three_update_arms_actually_run=True,config=p['configuration'],MBR_fixed=True,no_parameter_search=True))
    put(D/'CPU_SCORE_COMPLETION.json',dict(status='complete',count=32));print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in groups.items()},flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);a=ap.parse_args()
    try:globals()[a.stage]()
    except BaseException as e:put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(status='failed',stage=a.stage,type=type(e).__name__,error=str(e),all_partials_preserved=True));raise
