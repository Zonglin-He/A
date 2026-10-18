"""CPU cached output construction, all32 seal, then offline evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,save,load,sha
from vg_tta.stvg_dino_local_residual32_v1 import ARMS,construct,contract
D=ROOT/'artifacts/stvg_dino_local_residual32_v1'
P=ROOT/'results/stvg_error_state/2026-10-18/dino_local_residual32'
PRE=ROOT/'artifacts/stvg_dino_mask_align32_v1'
CODE=['vg_tta/stvg_dino_local_residual32_v1.py','scripts/run_stvg_dino_local_residual32_v1.py',
      'scripts/audit_stvg_dino_local_residual32_v1.py','scripts/report_stvg_dino_local_residual32_v1.py',
      'protocols/stvg_dino_local_residual32_v1.md']

def verify():
    lock=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==lock['plan_sha256']
    for f,h in lock['pins'].items():assert sha(ROOT/f)==h,f
    return read(D/'PLAN_LOCK.json')

def prepare():
    old=read(PRE/'PLAN_LOCK.json');assert read(PRE/'FINAL_COMPLETION.json')['status']=='complete'
    props=contract();put(D/'CPU_PROPERTY_COMPLETION.json',props)
    plan=dict(status='locked',count=32,per_domain=16,records=old['records'],arms=ARMS,threshold=.35,
              denominator=24,source_scales=old['scale'],time='exact stored final-MBR',
              new_model_calls=0,new_expert_calls=0,network_updates=0,new_features=0,
              candidate_cache_commit='505103c12212c34b33e0bbb0c5dcfea71eb28d07',
              target_development_exposed=True,new_GT_after_all32_seal=True,
              only_change='shared4 application scope and independent per-anchor residual aggregation',
              dynamic_OPD=False,created_epoch=time.time(),CPU_measurement_wall_cap=600)
    put(D/'PLAN_LOCK.json',plan)
    files=CODE+['methods/CURRENT_METHOD.json','vg_tta/stvg_error_state_correction_v1.py',
        str((PRE/'PLAN_LOCK.json').relative_to(ROOT)),str((PRE/'GLOBAL_PREDICTION_BARRIER.json').relative_to(ROOT))]
    files += [str((PRE/'predictions'/(r['alias']+'.pt')).relative_to(ROOT))for r in old['records']]
    put(D/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f)for f in files},plan_sha256=sha(D/'PLAN_LOCK.json')))
    put(D/'STATUS.json',dict(status='prepared',count=0,total=32,GT_read=False),True)

def measure():
    p=verify();start=time.perf_counter();files=[]
    for r in p['records']:
        old=load(PRE/'predictions'/(r['alias']+'.pt'));out=construct(old)
        z=dict(alias=r['alias'],dataset=r['dataset'],active=old['active'],frame_ids=old['frame_ids'],
               planned_positions=old['planned_positions'],interval=old['interval'],**out,GT_read=False)
        files.append(save(D/'predictions'/(r['alias']+'.pt'),z))
        assert time.perf_counter()-start<p['CPU_measurement_wall_cap']
    verify();cost=dict(CPU_output_seconds=time.perf_counter()-start,new_model_calls=0,new_DINO_frames=0,
                       new_Source_captures=0,new_backbone_forwards=0,GPU_seconds=0,old_E_Source_cost_excluded=True)
    put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,all_outputs_before_GT=True,GT_read=False,compute=cost))
    put(D/'STATUS.json',dict(status='sealed_before_scoring',count=32,total=32,GT_read=False),True)
    print(cost,flush=True)

def mean(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None

def metrics(z,row,gt,span):
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
    from vg_tta.stvg_error_state_correction_v1 import encode
    ids=np.asarray(z['frame_ids']);pos=z['planned_positions'];observed=np.isin(np.arange(len(ids)),pos)
    known=np.array([int(f)in gt for f in ids]);ds=z['dataset'];iv=z['interval'];values={};tubes={};geometry={}
    for arm,b in z['outputs'].items():
        tube=DenseTube(b,row,gt,span,clip=ds=='hc2');v=official(b,row,gt,span,iv,ds)
        assert max(abs(v[k]-tube.score(iv)[k])for k in ['v','t','s'])<2e-10
        si=np.array([iou_gt(b[j],gt[int(f)],row,ds)if known[j]else np.nan for j,f in enumerate(ids)])
        interp=~np.isin(tube.fids,ids)
        values[arm]=dict(**v,observed_actual_s=mean(si[known&observed]),unobserved_actual_s=mean(si[known&~observed]),
                        benchmark_interpolated_s=mean(tube.iou[interp]))
        tubes[arm]=tube
        delta=encode(b)-encode(z['outputs']['E_MBR']);geometry[arm]={}
        for name,mask in [('all',np.ones(len(ids),bool)),('observed',observed),('unobserved',~observed)]:
            center=np.linalg.norm(delta[mask,:2],axis=1);size=np.linalg.norm(delta[mask,2:],axis=1)
            geometry[arm][name]=dict(center_mean=mean(center),center_RMS=float(np.sqrt(np.mean(center**2))),
                center_max=float(center.max()),log_size_mean=mean(size),log_size_RMS=float(np.sqrt(np.mean(size**2))),
                log_size_max=float(size.max()),changed_nodes=int(np.any(b[mask]!=z['outputs']['E_MBR'][mask],axis=1).sum()))
    assert len(set(x['t']for x in values.values()))==1
    base=tubes['E_MBR'];obsf=ids[observed];interp=~np.isin(base.fids,ids)
    masks=dict(observed_actual=np.isin(base.fids,obsf),unobserved_actual=np.isin(base.fids,ids[~observed]),benchmark_interpolated=interp)
    inside=(base.fids>=max(iv[0],span[0]))&(base.fids<min(iv[1],span[1]));den=max(max(iv[1],span[1])-min(iv[0],span[0]),1)
    effects={}
    for arm,tube in tubes.items():
        d=tube.iou-base.iou
        effects[arm]={name:dict(known_frames=int(mask.sum()),inside_overlap_frames=int((mask&inside).sum()),
              changed_IoU_frames=int((mask&(d!=0)).sum()),mean_s_delta_pp=100*mean(d[mask])if mask.any()else None,
              v_contribution_pp=float(100*d[mask&inside].sum()/den))for name,mask in masks.items()}
        assert abs(sum(t['v_contribution_pp']for t in effects[arm].values())-100*(values[arm]['v']-values['E_MBR']['v']))<1e-10
    return dict(values=values,geometry=geometry,interpolation_effects=effects,
                known_observed_actual=int((known&observed).sum()),known_unobserved_actual=int((known&~observed).sum()))

def aggregate(rows):
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    endpoints=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s']
    pairs=[(a,'E_MBR')for a in ARMS[1:]]+[('Shared4_observed','Shared4_global'),('Independent4_observed','Shared4_observed')]
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
    # Only after the complete-output barrier may evaluation access GT.
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    allgt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in allgt};rows=[]
    oldrows={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/dino_mask_align32/RESULTS_PUBLIC.json')['rows']}
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];z=load(D/'predictions'/(a+'.pt'));row=plans[ds][q]
        assert list(row['frame_ids'])==list(z['frame_ids'])
        m=metrics(z,row,allgt[ds][0][q],allgt[ds][1][q])
        for now,old in [('E_MBR','E_MBR'),('Shared4_global','B_aligned_once')]:
            for k in ['v','t','s','observed_actual_s','unobserved_actual_s']:
                x,y=m['values'][now][k],oldrows[a]['values'][old][k]
                assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12,(a,k,x,y)
        for arm in ARMS[2:]:assert m['values'][arm]['unobserved_actual_s']==m['values']['E_MBR']['unobserved_actual_s']
        rows.append(dict(alias=a,dataset=ds,active=z['active'],accepted_feedback_per_planned_node=z['accepted_counts'].tolist(),**m))
    result=dict(status='complete',count=32,rows=rows,groups=aggregate(rows),compute=bar['compute'],
                all_outputs_before_new_GT=True,unknown_GT_kept_null=True,independent_confirmation=False,
                no_dynamic_OPD=True,decision='finite_local_once_experiment_only_no_automatic_expansion')
    put(P/'RESULTS_PUBLIC.json',result);put(D/'SCORING_COMPLETION.json',dict(status='complete',count=32,sha256=sha(P/'RESULTS_PUBLIC.json')))
    print({ds:{a:g['metrics'][a]['v']for a in ARMS}for ds,g in result['groups'].items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','measure','score']);a=p.parse_args()
    {'prepare':prepare,'measure':measure,'score':score}[a.stage]()
