"""All-readouts seal first; target GT only evaluates this finite frozen test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load
from scripts.run_stvg_query_e_opd64_v1 import D,P,PRE,ARMS,verify,status
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def mean(v):
    a=[float(x)for x in v if x is not None];return float(np.mean(a))if a else None


def summary(rows):
    means={a:{ep:mean([r['values'][a][ep]for r in rows])for ep in ['v','t','s','unobserved_actual_s','R03','R05']}for a in ARMS}
    contrasts={}
    for a in ARMS[2:]:
        for c in (['Source_MBR','E_MBR'] if a!='Dense_OPD' else ['Source_MBR','E_MBR','Sparse_OPD','Fixed_rollout','Dense_Direct']):
            contrasts[a+'_minus_'+c]={}
            for ep in ['v','s','unobserved_actual_s']:
                x=paired([None if r['values'][a][ep]is None or r['values'][c][ep]is None else 100*(r['values'][a][ep]-r['values'][c][ep])for r in rows]);v=np.array(x.get('per_parent_pp',[]))
                x.update(severe_loss_below_minus20pp=int((v< -20).sum()),severe_loss_fraction=None if not len(v)else float((v< -5).mean()))
                contrasts[a+'_minus_'+c][ep]=x
    return dict(parents=len(rows),queries=len(rows),query_mean=means,parent_mean=means,paired=contrasts,endpoint_available_counts={ep:sum(r['values']['Source_MBR'][ep]is not None for r in rows)for ep in ['v','t','s','unobserved_actual_s']},missing_GT_not_scored_as_failure=True)


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/source_expanded256/RESULTS.json')['rows']}
    features=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json');feats={ds:{r['query_ordinal']:r for r in features['rows'][ds]}for ds in targets}
    rows=[];checks=0;started=time.perf_counter()
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];ids=np.array(z['frame_ids'])
        known=np.array([int(t)in gt for t in ids]);unobserved=known&~np.isin(np.arange(len(ids)),z['planned_positions']);values={}
        for a,b in z['outputs'].items():
            v=official(b,row,gt,span,z['interval'],ds);other=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval']);assert max(abs(v[k]-other[k])for k in ['v','t','s'])<2e-10;checks+=3
            xy=np.concatenate((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            ui=box_iou(xy[unobserved],np.array([gt[int(fid)]for fid in ids[unobserved]]))if unobserved.any()else []
            values[a]=dict(**v,unobserved_actual_s=mean(ui),unobserved_known_frames=int(unobserved.sum()),R03=float(v['v']>.3),R05=float(v['v']>.5))
        assert len({v['t']for v in values.values()})==1
        for a,prior in [('Source_MBR','Source_no_correction'),('E_MBR','E_DINO_support')]:
            for ep in ['v','t','s','unobserved_actual_s']:
                x,y=values[a][ep],old[r['alias']]['values'][prior][ep];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12;checks+=1
        fits={}
        for arm,f in z['fits'].items():
            rounds=f['rounds'];fits[arm]=dict(active_parameters=f['active_parameters'],gradient_calls=f['gradient_calls'],empty=f['empty'],
                final_query_norm=f['parameter_displacement_norm'],final_output_displacement=f['output_displacement_l2'],
                parameter_step_norms=[v.get('parameter_step_norm')for v in rounds],parameter_gradient_norms=[v.get('parameter_gradient_norm')for v in rounds],
                output_step_displacements=[v.get('output_displacement_l2')for v in rounds],reward_variance=[v.get('reward_variance')for v in rounds],ESS=[v.get('ESS')for v in rounds],
                feedback_positions_count=len(f['feedback_positions']),pretrained_weights_frozen_exact=f['pretrained_weights_frozen_exact'])
        rows.append(dict(alias=r['alias'],dataset=ds,values=values,fits=fits,active=z['active'],compute=z['compute'],planned_slots=len(z['planned_positions']),
            bins={k:feats[ds][q]['bins'][k]for k in ['speed_sizes_sec','target_area','event_fraction']}))
    groups={ds:summary([r for r in rows if r['dataset']==ds])for ds in targets};macro={};rng=np.random.default_rng(20261018)
    for pair in groups['hc2']['paired']:
        macro[pair]={}
        for ep in ['v','s','unobserved_actual_s']:
            arr=[np.array(groups[ds]['paired'][pair][ep]['per_parent_pp'])for ds in targets];bs=[a[rng.integers(len(a),size=(10000,len(a)))].mean(1)for a in arr]
            macro[pair][ep]=dict(mean_pp=float(np.mean([a.mean()for a in arr])),CI95_pp=np.quantile(np.mean(bs,0),[.025,.975]).tolist(),equal_domain_weight=True)
    strata={ds:{key:{v:summary([r for r in rows if r['dataset']==ds and r['bins'][key]==v])for v in sorted({r['bins'][key]for r in rows if r['dataset']==ds})}for key in ['speed_sizes_sec','target_area','event_fraction']}for ds in targets}
    put(P/'RESULTS.json',dict(status='complete',count=64,rows=rows,summary=groups,equal_domain_macro=macro,strata=strata,shared_bin_edges=features['bins'],all64_sealed_before_GT=True,old_Source_E_baselines_exact=True,independent_metric_and_old_scalar_checks=checks,
        query_only_OPD_actually_executed=True,all_pretrained_weights_frozen=True,all_query_residuals_reset=True,historically_exposed_development=True,independent_confirmation=False))
    tests={c:all(groups[ds]['paired']['Dense_OPD_minus_'+c][ep]['mean']>0 for ds in targets for ep in ['v','unobserved_actual_s'])for c in ['Source_MBR','E_MBR','Sparse_OPD','Fixed_rollout','Dense_Direct']}
    put(P/'DECISION.json',dict(status='complete',Dense_OPD_positive_both_domains_both_endpoints_against=tests,
        action='report_current_video_query_only_scope_no_automatic_expansion_or_hyperparameter_change',
        no_pretrained_or_LN_weight_updates=True,no_automatic_second_backbone=True,not_joint_temporal_adaptation=True))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=64,GT_read=True,CPU_score_seconds=time.perf_counter()-started,result_sha256=sha(P/'RESULTS.json')));status('CPU_complete_independent_audit_required',64,GT_read=True)
    assert not torch.cuda.is_initialized()


if __name__=='__main__':main()
