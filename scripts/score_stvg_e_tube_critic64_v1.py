"""Post-seal, query-level feedback qualification; no optimizer experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,itertools,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_e_tube_critic64_v1 import D,P,verify


def average(x):
    x=[v for v in x if v is not None];return float(np.mean(x))if x else None


def interval(x,seed=20261018):
    x=np.array([v for v in x if v is not None],float)
    if not len(x):return dict(n=0,mean=None,CI95=None)
    rng=np.random.default_rng(seed);v=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),CI95=np.quantile(v,[.025,.975]).tolist(),positive=int((x>1e-8).sum()),negative=int((x< -1e-8).sum()),neutral=int((abs(x)<=1e-8).sum()))


def ranking(scores,quality,axis):
    strict=correct=ties=wrong=GTties=0
    for fixed in range(scores.shape[1-axis]):
        s=scores[:,fixed]if axis==0 else scores[fixed,:];y=quality[:,fixed]if axis==0 else quality[fixed,:]
        for a,b in itertools.combinations(range(len(s)),2):
            d=y[a]-y[b]
            if abs(d)<=1e-8:GTties+=1;continue
            strict+=1;z=(s[a]-s[b])*d
            if z>0:correct+=1
            elif z<0:wrong+=1
            else:ties+=1
    return dict(accuracy=(correct+.5*ties)/strict if strict else None,strict_pairs=strict,correct=correct,wrong=wrong,critic_ties=ties,GT_ties=GTties)


def json_scalars(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:json_scalars(v)for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_scalars(v)for v in value]
    return value


def export(path,value):put(path,json_scalars(value))


def score():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    torch.set_num_threads(4);p=verify();b=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert b['count']==64 and not b['GT_read'];start=time.perf_counter()
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets};rows=[];checks=collections.Counter()
    old=read(ROOT/'results/stvg_error_state/2026-10-18/query_e64/RESULTS.json');oldby={r['alias']:r for r in old['rows']}
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q]
        scores=z['scores'].double().numpy();metrics=[]
        for boxes in z['boxes']:
            mm=[]
            for I in z['intervals']:
                v=official(boxes,row,gt,span,I,ds);vv=DenseTube(boxes,row,gt,span,clip=ds=='hc2').score(I)
                assert all(abs(v[k]-vv[k])<2e-10 for k in ['v','t','s']);checks['independent_metrics']+=3;mm.append(v)
            metrics.append(mm)
        quality={k:np.array([[m[k]for m in rr]for rr in metrics])for k in ['v','t','s']}
        chosen=z['selection'];assert chosen==list(np.unravel_index(scores.argmax(),scores.shape));e=metrics[2][0]
        assert all(abs(e[k]-oldby[rec['alias']]['values']['E_MBR'][k])<2e-10 for k in ['v','t','s']);checks['E_baseline_exact']+=3
        j=int(scores[2].argmax());a=int(scores[:,0].argmax());oracle=list(np.unravel_index(quality['v'].argmax(),scores.shape))
        values=dict(Source_MBR=metrics[0][0],E_MBR=e,Joint=metrics[chosen[0]][chosen[1]],E_time_selection=metrics[2][j],Spatial_at_MBR=metrics[a][0],
            Oracle=metrics[oracle[0]][oracle[1]],Oracle_time_at_E=metrics[2][int(quality['v'][2].argmax())],Oracle_space_at_MBR=metrics[int(quality['v'][:,0].argmax())][0])
        native=z['intervals'][0];final=z['intervals'][chosen[1]];ids=np.array(z['frame_ids']);baseinside=(ids>=native[0])&(ids<native[1]);inside=(ids>=final[0])&(ids<final[1]);removed=baseinside&~inside;added=inside&~baseinside
        from vg_tta.tastvg_oracle_event5_v1 import box_iou
        eboxes=np.array(z['boxes'][2]);xy=np.c_[eboxes[:,:2]-eboxes[:,2:]/2,eboxes[:,:2]+eboxes[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
        if ds=='hc2':xy=np.maximum(xy,0)
        known=np.array([int(v)in gt for v in ids]);siou=np.full(len(ids),np.nan);siou[known]=box_iou(xy[known],np.array([gt[int(v)]for v in ids[known]]))
        def knownmean(mask):return float(np.mean(siou[mask&known]))if (mask&known).any()else None
        rows.append(dict(alias=rec['alias'],dataset=ds,active=z['active'],candidate_shape=list(scores.shape),candidate_count=int(scores.size),scores=scores.tolist(),
            time_candidates=[dict(index=i,start_fraction=(I[0]-ids[0])/(ids[-1]+1-ids[0]),duration_fraction=(I[1]-I[0])/(ids[-1]+1-ids[0]))for i,I in enumerate(z['intervals'])],
            all_metrics=metrics,selection=chosen,values=values,
            spatial_ranking_v=ranking(scores,quality['v'],0),temporal_ranking_v=ranking(scores,quality['v'],1),temporal_ranking_t=ranking(scores,quality['t'],1),
            ranking_v_all_GT_ties=bool(np.ptp(quality['v'])<=1e-8),
            selection_effect=dict(t_change_pp=100*(values['Joint']['t']-e['t']),v_change_pp=100*(values['Joint']['v']-e['v']),
                duration_change_fraction=((final[1]-final[0])-(native[1]-native[0]))/(ids[-1]+1-ids[0]),removed_decoder_frames=int(removed.sum()),added_decoder_frames=int(added.sum()),
                removed_known_GT_frames=int((removed&known).sum()),added_known_GT_frames=int((added&known).sum()),removed_E_sIoU=knownmean(removed),added_E_sIoU=knownmean(added),retained_E_sIoU=knownmean(baseinside&inside)),
            compute=z['compute']))
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds]
        groups[ds]=dict(count=32,active=sum(x['active']for x in rr),
            metrics={a:{ep:average([r['values'][a][ep]for r in rr])for ep in ['v','t','s']}for a in rows[0]['values']},
            ranking={name:interval([r[name]['accuracy']for r in rr])for name in ['spatial_ranking_v','temporal_ranking_v','temporal_ranking_t']},
            deltas_vs_E={a:{ep:interval([100*(r['values'][a][ep]-r['values']['E_MBR'][ep])for r in rr])for ep in ['v','t','s']}for a in ['Joint','E_time_selection','Spatial_at_MBR','Oracle','Oracle_time_at_E','Oracle_space_at_MBR']},
            time_change=dict(mean_duration_fraction=average([r['selection_effect']['duration_change_fraction']for r in rr]),shortened=sum(r['selection_effect']['duration_change_fraction']<0 for r in rr),
                v_improved_t_decreased=sum(r['selection_effect']['v_change_pp']>1e-8 and r['selection_effect']['t_change_pp']< -1e-8 for r in rr)),
            severe_Joint_E_v_loss=sum(100*(r['values']['Joint']['v']-r['values']['E_MBR']['v'])< -5 for r in rr))
    rng=np.random.default_rng(20261018);draw=[]
    for ds in ['hc2','vidstg']:
        x=np.array([100*(r['values']['Joint']['v']-r['values']['E_MBR']['v'])for r in rows if r['dataset']==ds]);draw.append(x[rng.integers(0,len(x),(10000,len(x)))].mean(1))
    macro=dict(mean_pp=float(np.mean([groups[d]['deltas_vs_E']['Joint']['v']['mean']for d in groups])),CI95_pp=np.quantile((draw[0]+draw[1])/2,[.025,.975]).tolist())
    rankpass=all(groups[d]['ranking'][name]['CI95']is not None and groups[d]['ranking'][name]['CI95'][0]>.5 for d in groups for name in ['spatial_ranking_v','temporal_ranking_v'])
    domainpass=all(groups[d]['deltas_vs_E']['Joint']['v']['mean']>=0 for d in groups);gate=rankpass and domainpass and macro['CI95_pp'][0]>0
    decision=dict(status='pass_stage1_start_stage2'if gate else'stop_current_critic_before_OPD',advance_stage2=gate,ranking_both_domains_positive_evidence=rankpass,domain_Joint_nonnegative=domainpass,macro_positive_CI=macro['CI95_pp'][0]>0,
        uncertainty_not_automatic_expansion=True,not_all_critic_or_OPD_impossible=True)
    export(P/'RESULTS.json',dict(status='complete',count=64,total_tubes=sum(r['candidate_count']for r in rows),groups=groups,macro=macro,rows=rows,independent_checks=dict(checks),
        decision=decision,GT_after_all_candidate_scores_sealed=True,no_optimizer_updates=True,old_development_exposure=True))
    export(D/'STAGE1_DECISION.json',decision);export(P/'DECISION.json',decision)
    export(D/'CPU_COMPLETION.json',dict(status='complete',count=64,CPU_seconds=time.perf_counter()-start,result_sha256=sha(P/'RESULTS.json')))
    print(dict(groups=groups,macro=macro,decision=decision))


if __name__=='__main__':score()
