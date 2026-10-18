"""CPU cached output construction, all32 seal, then offline evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
from vg_tta.stvg_dino_endpoint32_v1 import ARMS,construct,properties
D=ROOT/'artifacts/stvg_dino_endpoint32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_endpoint32'
PRE=ROOT/'artifacts/stvg_dino_mask_align32_v1'
LOCAL=ROOT/'artifacts/stvg_dino_local_residual32_v1'
CODE=['vg_tta/stvg_dino_endpoint32_v1.py','scripts/run_stvg_dino_endpoint32_v1.py','scripts/audit_stvg_dino_endpoint32_v1.py','scripts/report_stvg_dino_endpoint32_v1.py','protocols/stvg_dino_endpoint32_v1.md']

def verify():
    lock=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==lock['plan_sha256']
    for f,h in lock['pins'].items():assert sha(ROOT/f)==h,f
    return read(D/'PLAN_LOCK.json')

def prepare():
    old=read(LOCAL/'PLAN_LOCK.json');assert read(LOCAL/'FINAL_COMPLETION.json')['status']=='complete'
    put(D/'CPU_PROPERTIES.json',properties())
    p=dict(status='locked',records=old['records'],count=32,per_domain=16,arms=ARMS,threshold=.35,denominator=24,
        temporal='exact final-MBR',changed_factor='teacher-minus-E instead of teacher-minus-sampled-input',
        new_model_calls=0,new_expert_calls=0,no_step_size_search=True,no_new_cohort=True,no_dynamic_OPD=True,
        all32_seal_before_new_GT=True,old_target_exposure=True,CPU_construction_cap=600)
    put(D/'PLAN_LOCK.json',p)
    files=CODE+['methods/CURRENT_METHOD.json','scripts/run_stvg_dino_local_residual32_v1.py','vg_tta/stvg_error_state_correction_v1.py']
    files+=[str((d/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))for d in [PRE,LOCAL]for r in p['records']]
    put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32,GT_read=False),True)

def measure():
    p=verify();files=[];start=time.perf_counter()
    for r in p['records']:
        old=load(PRE/'predictions'/(r['alias']+'.pt'));local=load(LOCAL/'predictions'/(r['alias']+'.pt'))
        z=dict(alias=r['alias'],dataset=r['dataset'],active=old['active'],frame_ids=old['frame_ids'],
            planned_positions=old['planned_positions'],interval=old['interval'],GT_read=False,**construct(old,local))
        files.append(save(D/'predictions'/(r['alias']+'.pt'),z));assert time.perf_counter()-start<p['CPU_construction_cap']
    verify();cost=dict(CPU_output_seconds=time.perf_counter()-start,GPU_seconds=0,new_DINO_frames=0,new_model_calls=0,new_Source_captures=0,old_Source_E_cost_excluded=True)
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,all_outputs_before_GT=True,GT_read=False,files=files,compute=cost))
    put(D/'STATUS.json',dict(status='sealed_before_scoring',count=32,total=32,GT_read=False),True)
    print(cost,flush=True)

from scripts.run_stvg_dino_local_residual32_v1 import mean,metrics

def aggregate(rows):
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    endpoints=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']
    pairs=[(a,'E_MBR')for a in ARMS[1:]]+[('Endpoint_local','Residual_local')]
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];g=dict(queries=len(rr),parents=len(rr),active=sum(r['active']for r in rr))
        g['metrics']={a:{k:mean([r['values'][a][k]for r in rr])for k in endpoints}for a in ARMS}
        g['paired']={a+'_minus_'+b:{k:paired([None if r['values'][a][k]is None or r['values'][b][k]is None else
            100*(r['values'][a][k]-r['values'][b][k])for r in rr])for k in endpoints if k!='t'}for a,b in pairs}
        g['displacement']={a:{scope:{k:mean([r['geometry'][a][scope][k]for r in rr])for k in rr[0]['geometry'][a][scope]}
                            for scope in ['all','observed','unobserved']}for a in ARMS}
        g['v_effect_contributions_pp']={a:{scope:paired([r['interpolation_effects'][a][scope]['v_contribution_pp']for r in rr])
                    for scope in ['observed_actual','unobserved_actual','benchmark_interpolated']}for a in ARMS}
        g['known_observed_query_count']=sum(r['known_observed_actual']>0 for r in rr)
        g['known_unobserved_query_count']=sum(r['known_unobserved_actual']>0 for r in rr)
        groups[ds]=g
    return groups

def score():
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/dino_local_residual32/RESULTS_PUBLIC.json')['rows']};rows=[]
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];z=load(D/'predictions'/(a+'.pt'));row=plans[ds][q]
        assert list(row['frame_ids'])==list(z['frame_ids']);m=metrics(z,row,gt[ds][0][q],gt[ds][1][q])
        for now,old in [('E_MBR','E_MBR'),('Residual_local','Independent4_observed')]:
            for k in ['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']:
                x,y=m['values'][now][k],prior[a]['values'][old][k]
                assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12,(a,now,k)
        assert m['values']['Endpoint_local']['unobserved_actual_s']==m['values']['E_MBR']['unobserved_actual_s']
        rows.append(dict(alias=a,dataset=ds,active=z['active'],accepted_feedback_per_planned_node=z['accepted_counts'].tolist(),**m))
    result=dict(status='complete',count=32,rows=rows,groups=aggregate(rows),compute=bar['compute'],all_outputs_before_new_GT=True,
        changed_estimation_target=True,identity_teacher_noop_not_guaranteed=True,unknown_GT_null=True,independent_confirmation=False,
        no_new_cohort_or_center_query_or_dynamic_OPD=True)
    put(P/'RESULTS_PUBLIC.json',result);put(D/'SCORING_COMPLETION.json',dict(status='complete',count=32,sha256=sha(P/'RESULTS_PUBLIC.json')))
    print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in result['groups'].items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure','score']);a=p.parse_args()
    {'prepare':prepare,'measure':measure,'score':score}[a.stage]()
