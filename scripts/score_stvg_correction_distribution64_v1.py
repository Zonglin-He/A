"""All-readouts seal first; target GT only evaluates this finite frozen test."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha,load
from scripts.run_stvg_correction_distribution64_v1 import D,P,PRE,ARMS,verify,status
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def mean(v):
    a=[float(x)for x in v if x is not None];return float(np.mean(a))if a else None


def summary(rows):
    means={a:{ep:mean([r['values'][a][ep]for r in rows])for ep in ['v','t','s','unobserved_actual_s','R03','R05']}for a in ARMS}
    contrasts={}
    for a in ARMS[2:]:
        for c in ['E_MBR','Source_MBR']:
            contrasts[a+'_minus_'+c]={}
            for ep in ['v','unobserved_actual_s']:
                x=paired([None if r['values'][a][ep]is None or r['values'][c][ep]is None else 100*(r['values'][a][ep]-r['values'][c][ep])for r in rows]);v=np.array(x.get('per_parent_pp',[]))
                x.update(severe_loss_below_minus20pp=int((v< -20).sum()),severe_loss_fraction=None if not len(v)else float((v< -5).mean()))
                contrasts[a+'_minus_'+c][ep]=x
    return dict(parents=len(rows),queries=len(rows),query_mean=means,parent_mean=means,paired=contrasts)


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
        d=z['decision'];h=d['distribution'];w=h['weights'];ix=d['selected_index']
        rows.append(dict(alias=r['alias'],dataset=ds,values=values,
            distribution=dict(latent_assignments=len(h['assignments']),unique_corrections=len(w),ESS=float(1/np.square(w).sum()),entropy=float(-np.sum(w[w>0]*np.log(w[w>0]))),maximum_weight=float(w.max()),
                MBR_selected_weight=float(w[ix]),MBR_expected_utility=float(d['expected_utilities'][ix]),mean_exact_E=bool(np.array_equal(z['outputs']['Correction_mean'],z['outputs']['E_MBR'])),MBR_exact_E=bool(np.array_equal(z['outputs']['Correction_MBR'],z['outputs']['E_MBR'])),
                no_support=not len(z['frames']),E_support_frames=load(ROOT/r['prior_prediction'])['estimate']['support_frames']),
            CPU_readout_seconds=z['compute']['CPU_readout_seconds'],planned_slots=len(z['planned_positions']),
            bins={k:feats[ds][q]['bins'][k]for k in ['speed_sizes_sec','target_area','event_fraction']}))
    groups={ds:summary([r for r in rows if r['dataset']==ds])for ds in targets};macro={};rng=np.random.default_rng(20261018)
    for pair in groups['hc2']['paired']:
        macro[pair]={}
        for ep in ['v','unobserved_actual_s']:
            arr=[np.array(groups[ds]['paired'][pair][ep]['per_parent_pp'])for ds in targets];bs=[a[rng.integers(len(a),size=(10000,len(a)))].mean(1)for a in arr]
            macro[pair][ep]=dict(mean_pp=float(np.mean([a.mean()for a in arr])),CI95_pp=np.quantile(np.mean(bs,0),[.025,.975]).tolist(),equal_domain_weight=True)
    strata={ds:{key:{v:summary([r for r in rows if r['dataset']==ds and r['bins'][key]==v])for v in sorted({r['bins'][key]for r in rows if r['dataset']==ds})}for key in ['speed_sizes_sec','target_area','event_fraction']}for ds in targets}
    put(P/'RESULTS.json',dict(status='complete',count=64,rows=rows,summary=groups,equal_domain_macro=macro,strata=strata,shared_bin_edges=features['bins'],all64_sealed_before_GT=True,old_Source_E_baselines_exact=True,independent_metric_and_old_scalar_checks=checks,
        output_space_inference_not_OPD=True,working_distribution_not_calibrated_identity_probability=True,historically_exposed_development=True,independent_confirmation=False))
    mean_pass={arm:all(groups[ds]['paired'][arm+'_minus_E_MBR'][ep]['mean']>0 for ds in targets for ep in ['v','unobserved_actual_s'])for arm in ARMS[2:]}
    put(P/'DECISION.json',dict(status='complete',readout_positive_vs_E_both_endpoints_both_domains=mean_pass,
        action='retain_deterministic_E_no_automatic_new_model_or_weight_update'if not any(mean_pass.values())else'development_signal_report_no_automatic_promotion_or_expansion',
        no_temperature_threshold_sweep=True,no_OPD_or_network_update=True,no_automatic_second_backbone=True,old_E32_negative_gate_preserved=True))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=64,GT_read=True,CPU_score_seconds=time.perf_counter()-started,result_sha256=sha(P/'RESULTS.json')));status('CPU_complete_independent_audit_required',64,GT_read=True)
    assert not torch.cuda.is_initialized()


if __name__=='__main__':main()
