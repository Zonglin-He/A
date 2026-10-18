"""All32-seal then target-GT evaluation; no selection or estimator calibration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_robust_correction32_v1 import D,P,ERROR,BASES,ARMS,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha


def mean(v):
    v=[float(x)for x in v if x is not None]
    return float(np.mean(v))if v else None


def paired(values):
    v=np.asarray([x for x in values if x is not None],dtype=float)
    if not len(v):return dict(mean=None,CI95=None,parents=0)
    ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)))
    loo=(v.sum()-v)/(len(v)-1)if len(v)>1 else None
    return dict(mean=float(v.mean()),CI95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),parents=len(v),
        positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),neutral=int((abs(v)<=1e-10).sum()),
        severe_loss_below_minus5pp=int((v< -5).sum()),positive_sum=float(v[v>0].sum()),negative_sum=float(v[v<0].sum()),
        leave_one_saved_mean_range=None if loo is None else [float(loo.min()),float(loo.max())],per_parent_pp=v.tolist())


def endpoint(row,base,arm,key):
    v=row['values'][base][arm]
    return v['metrics'][key]if key in ['v','t','s','R03','R05']else v['supports'][key]['s']


def delta(row,base,a,b,key):
    x,y=endpoint(row,base,a,key),endpoint(row,base,b,key)
    return None if x is None or y is None else 100*(x-y)


def summarize(rows):
    groups={};contrasts=[(a,b)for a in ARMS for b in ['Base','A_mean']if a!=b]+[('C_joint','B_robust_top1'),('D_C_center','C_joint')]
    for base in BASES:
        metrics={a:{ep:mean([endpoint(r,base,a,ep)for r in rows])for ep in ['v','t','s','R03','R05','unobserved','observed','unobserved_inside_MBR','benchmark_interpolated']}for a in ['Base']+ARMS}
        pairs={a+'_minus_'+b:{ep:paired([delta(r,base,a,b,ep)for r in rows])for ep in ['v','s','unobserved','unobserved_inside_MBR','benchmark_interpolated']}for a,b in contrasts}
        behavior={}
        for a in ['B_robust_top1','C_joint']:
            behavior[a]=dict(support_histogram={str(k):sum(r['behavior'][base][a]['support_frames']==k for r in rows)for k in range(5)},
                rejected_frames=sum(r['behavior'][base][a]['rejected_frames']for r in rows),
                whole_query_exact_noops=sum(r['behavior'][base][a]['exact_noop']for r in rows),
                alternative_to_Top1_frames=sum(r['behavior'][base][a]['alternative_to_Top1_frames']for r in rows),
                weak_score_selected_frames=sum(r['behavior'][base][a]['weak_score_selected_frames']for r in rows),
                max_leave_one_normalized_change=mean([r['behavior'][base][a]['max_leave_one_normalized_change']for r in rows]),
                protected_A_negative_queries=sum(delta(r,base,'A_mean','Base','v')< -1e-10 and delta(r,base,a,'A_mean','v')>1e-10 for r in rows),
                rescued_A_negative_to_nonnegative=sum(delta(r,base,'A_mean','Base','v')< -1e-10 and delta(r,base,a,'Base','v')>=-1e-10 for r in rows),
                active_improvement_over_Base=sum(delta(r,base,a,'Base','v')>1e-10 for r in rows))
        groups[base]=dict(queries=len(rows),parents=len(rows),metrics=metrics,paired=pairs,behavior=behavior)
    return groups


def score():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
    torch.set_num_threads(4);p=verify(inputs=True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['status']=='complete'and bar['count']==32 and bar['all_target_outputs_before_new_target_GT']
    assert sha(D/'SOURCE_CALIBRATION.json')==bar['source_calibration_sha256']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    # This is the first target-GT/old-identity-label read in this run.
    gtall={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gtall}
    oldscores={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/cached32/RESULTS.json')['rows']}
    review={r['source_alias']:r for r in read(ROOT/'artifacts/stvg_dino_error_attribution32_v1/VISUAL_REVIEW.json')['rows']}
    strata=read(ROOT/'artifacts/stvg_evidence_strata_v1/FEATURES_PRIVATE.json')['rows']
    strata={ds:{r['query_ordinal']:r for r in rr}for ds,rr in strata.items()}
    rows=[];checks=0;started=time.perf_counter()
    for r in p['records']:
        a=r['alias'];z=load(D/'predictions'/(a+'.pt'));ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q]
        gt,span=gtall[ds][0][q],gtall[ds][1][q];ids=np.array(z['frame_ids']);n=len(ids);iv=z['interval']
        known=np.array([int(fid)in gt for fid in ids]);unobs=known&~np.isin(np.arange(n),z['planned_positions'])
        inside=(ids>=iv[0])&(ids<iv[1]);scale=np.array([row['input']['width'],row['input']['height']]*2)
        true=np.zeros((n,4))
        for j in np.flatnonzero(known):true[j]=gt[int(ids[j])]
        masks=dict(unobserved=unobs,observed=known&~unobs,unobserved_inside_MBR=unobs&inside)
        values={};behavior={};geometries={};rv=review[a]
        for base in BASES:
            values[base]={};behavior[base]={};zz=z['bases'][base]
            for arm,b in dict(Base=zz['Base'],**zz['outputs']).items():
                tube=DenseTube(b,row,gt,span,clip=ds=='hc2');m=official(b,row,gt,span,iv,ds);check=tube.score(iv)
                assert max(abs(m[k]-check[k])for k in ['v','t','s'])<2e-10;checks+=3
                m.update(R03=float(m['v']>.3),R05=float(m['v']>.5))
                if arm!='Base':assert m['t']==values[base]['Base']['metrics']['t'];checks+=1
                px=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*scale
                if ds=='hc2':px=np.maximum(px,0)
                iou=np.zeros(n);iou[known]=box_iou(px[known],true[known])
                supports={name:dict(s=mean(iou[mask]),frames=int(mask.sum()))for name,mask in masks.items()}
                interp=~np.isin(tube.fids,ids);supports['benchmark_interpolated']=dict(s=mean(tube.iou[interp]),frames=int(interp.sum()))
                values[base][arm]=dict(metrics=m,supports=supports)
                if arm in ['Base','A_mean']:
                    old=oldscores[a]['values'][base]['DINO_full']['Base'if arm=='Base'else'GlobalError']
                    assert max(abs(old['clocks']['final_MBR'][k]-m[k])for k in ['v','t','s'])<1e-12
                    assert abs(old['supports']['unobserved_decoder']['s']-supports['unobserved']['s'])<1e-12;checks+=4
            for arm,est in [('B_robust_top1',zz['estimate_B']),('C_joint',zz['estimate_C'])]:
                selected=[];changed=0;weak=0
                for f,k in zip(z['frames'],est['selected']):
                    if k>=0:
                        idx=f['top1']if arm=='B_robust_top1'else k
                        selected.append(dict(slot=z['planned_positions'].index(f['position']),candidate_index=int(idx)))
                        changed+=idx!=f['top1'];weak+=f['scores'][idx]<.35
                behavior[base][arm]={k:est[k]for k in ['objective','support_frames','rejected_frames','assignments_evaluated','competitor_gap','competitor_delta_distance','leave_one_delta_distances']}
                behavior[base][arm].update(max_leave_one_normalized_change=max(est['leave_one_delta_distances'],default=0.),
                    exact_noop=bool(np.array_equal(zz['outputs'][arm],zz['Base'])),selected_candidates=selected,
                    alternative_to_Top1_frames=int(changed),weak_score_selected_frames=int(weak))
            gg=[]
            for fi,f in enumerate(z['frames']):
                pos=f['position'];slot=z['planned_positions'].index(pos);k=zz['estimate_C']['selected'][fi]
                oldcat='UN'if rv['semantic_role_unknown']else rv['slots'][slot]
                entry=dict(slot=slot,original_Top1_old_category=oldcat,unknown_role=rv['semantic_role_unknown'],
                    candidate_count=len(f['boxes']),selected_index=int(k),original_Top1_index=f['top1'],
                    exact_GT_available=bool(known[pos]),alternative_identity_not_labeled=True)
                if known[pos]:
                    bb=f['boxes'];xy=np.column_stack((bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2))*scale
                    ii=np.asarray(box_iou(xy,np.tile(true[pos],(len(bb),1))));best=float(ii.max())
                    entry.update(original_Top1_IoU=float(ii[f['top1']]),best_retained_IoU=best,
                        selected_IoU=None if k<0 else float(ii[k]),candidate_IoUs=ii.tolist(),
                        has_geometric_IoU_ge_half=best>=.5,
                        selected_geometric_IoU_ge_half=bool(k>=0 and ii[k]>=.5),
                        available_good_but_not_selected=bool(best>=.5 and (k<0 or ii[k]<.5)))
                else:entry.update(original_Top1_IoU=None,best_retained_IoU=None,selected_IoU=None,candidate_IoUs=None)
                gg.append(entry)
            geometries[base]=gg
        feat=strata[ds][q]
        rows.append(dict(alias=a,dataset=ds,parent_alias=a,admitted_frames=len(z['frames']),
            retained_candidates=sum(len(f['boxes'])for f in z['frames']),known_decoder_frames=int(known.sum()),
            unknown_decoder_frames=int((~known).sum()),values=values,behavior=behavior,geometry=geometries,
            semantic_role_unknown=rv['semantic_role_unknown'],old_labels_model_assisted=True,new_identity_labels=0,
            motion_bin=feat['bins']['speed_sizes_sec'],target_area_bin=feat['bins']['target_area'],
            event_length_bin=feat['bins']['event_fraction'],query_form=feat['query_form']))
        assert time.perf_counter()-started<p['CPU_cap_seconds'];print('SCORED',a,flush=True)
    groups={ds:summarize([r for r in rows if r['dataset']==ds])for ds in gtall}
    subgroups={ds:{key:{v:summarize([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds})}for key in ['motion_bin','target_area_bin','event_length_bin','query_form']}for ds in gtall}
    tests={arm:{ds:{ep:groups[ds]['Before']['paired'][arm+'_minus_Base'][ep]['mean']>0 for ep in ['v','unobserved']}for ds in gtall}for arm in ARMS}
    cpositive=all(v for dd in tests['C_joint'].values()for v in dd.values())
    cbeats=all(groups[ds]['Before']['paired']['C_joint_minus_'+other][ep]['mean']>0 for ds in gtall for other in ['A_mean','B_robust_top1']for ep in ['v','unobserved'])
    decision=dict(status='complete',mean_screens=tests,C_positive_both_endpoints_both_domains=cpositive,
        C_beats_A_B_both_endpoints_both_domains=cbeats,
        action='development_signal_only_propose_independent_confirmation'if cpositive and cbeats else'stop_this_fixed_geometry_only_joint_estimator_no_sweep',
        no_automatic_confirmation_or_dense_OPD=True,no_production_promotion=True,screen_not_significance_test=True)
    P.mkdir(parents=True,exist_ok=True)
    put(P/'RESULTS.json',dict(status='complete',count=32,primary='Before/final_MBR/original_admitted_full_DINO_support',
        groups=groups,strata=subgroups,rows=rows,all32_sealed_before_target_GT=True,
        output_intervention_not_parameter_TTA=True,fixed_inherited_outputs_not_online_replay=True,
        unknown_GT_not_zero_filled=True,source_train_only_calibration=True))
    put(P/'DECISION.json',decision)
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,metric_and_old_endpoint_checks=checks,
        target_GT_read_after_all_outputs_sealed=True,CPU_score_seconds=time.perf_counter()-started,
        result_sha256=sha(P/'RESULTS.json'),new_models_experts_training_Student_updates=0))
    put(D/'STATUS.json',dict(status='complete',stage='measured_audit_publication_pending',count=32),True)
    print('DECISION',decision['action'],flush=True)


if __name__=='__main__':score()
