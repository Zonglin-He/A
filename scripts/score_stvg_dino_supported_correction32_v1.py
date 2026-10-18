"""Postseal E evaluation, exact old-method reuse and choice/acceptance attribution."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_supported_correction32_v1 import D,P,OLD,OLDP,BASES,verify
from scripts.score_stvg_joint_robust_correction32_v1 import paired,mean


def value(r,base,arm,ep):
    v=r['values'][base][arm];return v['metrics'][ep]if ep in ['v','t','s','R03','R05']else v['supports'][ep]['s']


def diff(r,base,arm,control,ep):
    a,b=value(r,base,arm,ep),value(r,base,control,ep)
    return None if a is None or b is None else (a-b)*100


def summarize(rows):
    result={}
    for base in BASES:
        metrics={a:{ep:mean([value(r,base,a,ep)for r in rows])for ep in ['v','t','s','R03','R05','unobserved','observed','unobserved_inside_MBR','benchmark_interpolated']}for a in ['Base','A_mean','B_robust_top1','C_joint','D_C_center','E_DINO_support']}
        pairs={c:{ep:paired([diff(r,base,'E_DINO_support',c,ep)for r in rows])for ep in ['v','s','unobserved','unobserved_inside_MBR','benchmark_interpolated']}for c in ['Base','A_mean','B_robust_top1','C_joint','D_C_center']}
        attribution={ep:{axis:paired([r['attribution'][base][ep][axis]for r in rows])for axis in ['candidate_choices','accepted_frame_mask','interaction']}for ep in ['v','unobserved']}
        result[base]=dict(queries=len(rows),parents=len(rows),metrics=metrics,E_minus=pairs,E_minus_C_factorial_attribution=attribution,
            behavior={k:sum(r['behavior'][base][k]for r in rows)for k in ['support_frames','rejected_frames','both_accepted_choice_changes',
                'C_accepted_E_rejected','C_rejected_E_accepted','both_rejected','alternative_to_Top1_frames','weak_score_selected_frames','exact_noop']},
            support_histogram={str(k):sum(r['behavior'][base]['support_frames']==k for r in rows)for k in range(5)})
    return result


def score():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou
    torch.set_num_threads(4);p=verify(inputs=True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['count']==32 and bar['all_E_and_diagnostics_before_target_GT']
    assert sha(D/'BETA_CALIBRATION.json')==bar['beta_calibration_sha256']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    old=read(OLDP/'RESULTS.json');olds={r['alias']:r for r in old['rows']}
    target={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in target}
    rows=[];checks=0;started=time.perf_counter()
    for r in p['records']:
        a=r['alias'];ds=r['dataset'];q=r['query_ordinal'];z=load(D/'predictions'/(a+'.pt'));saved=olds[a]
        row=plans[ds][q];gt,span=target[ds][0][q],target[ds][1][q];ids=np.array(z['frame_ids']);iv=z['interval']
        known=np.array([int(fid)in gt for fid in ids]);unobs=known&~np.isin(np.arange(len(ids)),z['planned_positions'])
        scale=np.array([row['input']['width'],row['input']['height']]*2);truthxy=np.zeros((len(ids),4))
        for i in np.flatnonzero(known):truthxy[i]=gt[int(ids[i])]
        masks=dict(unobserved=unobs,observed=known&~unobs,unobserved_inside_MBR=unobs&(ids>=iv[0])&(ids<iv[1]))
        values={};behavior={};geometry={};attribution={};diagnostic_values={}
        for base in BASES:
            zz=z['bases'][base];vals={}
            for label,b in {**zz['candidate_acceptance_diagnostics'],'E':zz['E']}.items():
                tube=DenseTube(b,row,gt,span,clip=ds=='hc2');metrics=official(b,row,gt,span,iv,ds);other=tube.score(iv)
                assert max(abs(metrics[k]-other[k])for k in ['v','s','t'])<2e-10;checks+=3
                metrics.update(R03=float(metrics['v']>.3),R05=float(metrics['v']>.5))
                assert metrics['t']==saved['values'][base]['Base']['metrics']['t'];checks+=1
                px=np.column_stack((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2))*scale
                if ds=='hc2':px=np.maximum(px,0)
                iou=np.zeros(len(ids));iou[known]=box_iou(px[known],truthxy[known])
                supports={k:dict(s=mean(iou[m]),frames=int(m.sum()))for k,m in masks.items()}
                bm=~np.isin(tube.fids,ids);supports['benchmark_interpolated']=dict(s=mean(tube.iou[bm]),frames=int(bm.sum()))
                vals[label]=dict(metrics=metrics,supports=supports)
            assert vals['E']==vals['3'];assert vals['0']==saved['values'][base]['C_joint'];checks+=2
            values[base]={**saved['values'][base],'E_DINO_support':vals['E']};diagnostic_values[base]=vals
            attr={}
            for ep in ['v','unobserved']:
                v=[vals[str(i)]['metrics']['v']if ep=='v'else vals[str(i)]['supports']['unobserved']['s']for i in range(4)]
                if any(t is None for t in v):attr[ep]=dict(candidate_choices=None,accepted_frame_mask=None,interaction=None)
                else:
                    choice=.5*((v[2]-v[0])+(v[3]-v[1]))*100;mask=.5*((v[1]-v[0])+(v[3]-v[2]))*100
                    assert abs(choice+mask-100*(v[3]-v[0]))<1e-9;checks+=1
                    attr[ep]=dict(candidate_choices=choice,accepted_frame_mask=mask,interaction=(v[3]-v[2]-v[1]+v[0])*100)
            attribution[base]=attr;e=zz['estimate'];c=zz['old_selected'];sel=e['selected']
            behavior[base]={k:e[k]for k in ['objective','geometry_objective','score_objective','support_frames','rejected_frames','assignments_evaluated','leave_one_delta_distances']}
            behavior[base].update(both_accepted_choice_changes=sum(ci>=0 and ei>=0 and ci!=ei for ci,ei in zip(c,sel)),
                C_accepted_E_rejected=sum(ci>=0 and ei<0 for ci,ei in zip(c,sel)),C_rejected_E_accepted=sum(ci<0 and ei>=0 for ci,ei in zip(c,sel)),
                both_rejected=sum(ci<0 and ei<0 for ci,ei in zip(c,sel)),exact_noop=bool(np.array_equal(zz['E'],load(OLD/'predictions'/(a+'.pt'))['bases'][base]['Base'])),
                alternative_to_Top1_frames=sum(k>=0 and k!=f['top1']for f,k in zip(z['frames'],sel)),
                weak_score_selected_frames=sum(k>=0 and f['scores'][k]<.35 for f,k in zip(z['frames'],sel)))
            gg=[]
            for f,ci,ei,prior in zip(z['frames'],c,sel,saved['geometry'][base]):
                assert prior['slot']==z['planned_positions'].index(f['position'])
                entry=dict(slot=prior['slot'],candidate_count=len(f['boxes']),C_index=ci,E_index=ei,
                    old_Top1_category=prior['original_Top1_old_category'],role_unknown=prior['unknown_role'],
                    exact_GT_available=prior['exact_GT_available'],alternative_identity_not_labeled=True)
                ious=prior['candidate_IoUs'];entry.update(best_retained_IoU=None if ious is None else max(ious),
                    E_selected_IoU=None if ious is None or ei<0 else ious[ei],C_selected_IoU=None if ious is None or ci<0 else ious[ci],
                    Top1_IoU=prior['original_Top1_IoU'],candidate_IoUs=ious,E_selected_score=None if ei<0 else float(f['scores'][ei]))
                gg.append(entry)
            geometry[base]=gg
        rows.append(dict(alias=a,dataset=ds,parent_alias=a,admitted_frames=len(z['frames']),values=values,behavior=behavior,
            geometry=geometry,attribution=attribution,diagnostic_factorial_values=diagnostic_values,
            old_labels_model_assisted=True,new_identity_labels=0,
            **{k:saved[k]for k in ['motion_bin','target_area_bin','event_length_bin','query_form']}))
        print('SCORED_E',a,flush=True)
    groups={ds:summarize([r for r in rows if r['dataset']==ds])for ds in target}
    strata={ds:{key:{v:summarize([r for r in rows if r['dataset']==ds and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds})}for key in ['motion_bin','target_area_bin','query_form']}for ds in target}
    positive=all(groups[ds]['Before']['E_minus']['Base'][ep]['mean']>0 for ds in target for ep in ['v','unobserved'])
    hc=all(groups['hc2']['Before']['E_minus']['A_mean'][ep]['mean']>0 for ep in ['v','unobserved'])
    vid=all(groups['vidstg']['Before']['E_minus']['A_mean'][ep]['mean']>=-1e-10 for ep in ['v','unobserved'])
    decision=dict(status='complete',E_vs_Base_positive_both_domains_both_endpoints=positive,HC_retains_gain_over_A=hc,
        Vid_restores_A_both_endpoints=vid,mean_gate=positive and hc and vid,
        action='eligible_for_separately_locked_independent_confirmation'if positive and hc and vid else 'stop_simple_geometry_plus_DINO_score_rule_family',
        no_automatic_dense_OPD=True,no_production_change=True,no_more_beta_transform_gate_kernel_or_grid_search=True,
        exposed_development_not_independent_confirmation=True)
    P.mkdir(parents=True,exist_ok=True);put(P/'RESULTS.json',dict(status='complete',count=32,primary='Before/final_MBR',
        new_formal_method_groups=1,groups=groups,strata=strata,rows=rows,old_ABCD_exact_metric_reuse=True,
        old_results_sha256=sha(OLDP/'RESULTS.json'),all_new_outputs_sealed_before_target_GT=True,
        source_validation_beta_selection_not_target=True,diagnostic_hybrids_not_selectable_methods=True,
        all_noops_unknown_and_negative_cases_retained=True,output_intervention_not_parameter_TTA=True))
    put(P/'DECISION.json',decision);put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,
        official_independent_and_factorial_checks=checks,CPU_score_seconds=time.perf_counter()-started,
        new_models_experts_training_Student_updates=0,result_sha256=sha(P/'RESULTS.json')))
    print('DECISION',decision['action'],flush=True)


if __name__=='__main__':score()
