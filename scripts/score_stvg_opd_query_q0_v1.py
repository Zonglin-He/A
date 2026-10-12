"""Official Q0 dense evaluation after all 64 complete frozen diagnostics seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,PUB,GROUPS,read,write,status,sha,load,verify


def statistics(rows,field):
    by=collections.defaultdict(list)
    for r in rows:
        if r.get(field) is not None:by[r['development_parent']].append(float(r[field]))
    values=np.asarray([math.fsum(v)/len(v) for v in by.values()])
    if not len(values):return dict(parents=0,mean=None,ci95=None)
    rng=np.random.default_rng(20261012);boot=[]
    for _ in range(100):boot.extend(values[rng.integers(len(values),size=(100,len(values)))].mean(1).tolist())
    return dict(parents=len(values),mean=math.fsum(values)/len(values),ci95=np.quantile(boot,[.025,.975]).tolist(),
                harm_5pp_parents=int(np.count_nonzero(values<-.05)),harm_20pp_parents=int(np.count_nonzero(values<-.20)))


def run():
    verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert b['status']=='sealed' and b['cells']==64 and not b['GT_read'] and b['all_three_groups_and_true_mean_decodes']
    assert read(BASE/'ROOT_DEPLOYMENT_READBACK.json')['status']=='pass'
    write(BASE/'GT_EXPOSURE.json',dict(status='offline_Q0_development_diagnosis_after_all_groups_and_means_global',
        barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),GT_used_for_directions_budget_rewards_or_means=False,time=time.time()))
    import torch;torch.set_num_threads(2)
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    from scripts.analyze_stvg_opd_mechanism_review_v1 import partition
    records=read(BASE/'DESIGN_LOCK.json')['records'];rows=[];candidate_rows=[];ds=None;checks=0;tick=time.time()
    PUB.mkdir(parents=True,exist_ok=True)
    for r in records:
        if ds!=r['dataset']:
            ds=r['dataset'];parents=[z['query_ordinal'] for z in records if z['dataset']==ds]
            gt,spans,provenance=truths(ds,dict(parents=parents))
        path=BASE/'formal'/f"{r['index']:05}.pt";assert sha(path)==b['files'][str(path.relative_to(ROOT))]
        z=load(path);ep=z['episode'];q=r['query_ordinal'];raw=read_row(ds,q)
        interval=z['interval'];before=ep['before']
        before_values=official(before.numpy(),raw,gt[q],spans[q],interval,ds)
        v,fids,before_ious=dense_metric(before.numpy(),raw,gt[q],spans[q],interval,ds)
        assert all(abs(v[k]-before_values[k])<2e-10 for k in v);checks+=3
        masks=partition(fids,raw['frame_ids'],[raw['frame_ids'][p] for p in ep['positions']],interval)
        event=(fids>=spans[q][0])&(fids<spans[q][1])
        base=dict(dataset=ds,development_parent=r['development_parent'],query_ordinal=q,arrival=r['arrival'],
            common_k=ep['common_k'],no_expert=not ep['positions'],output_RMS_budget=ep['output_RMS_budget'],
            Query_rank=ep['SVD']['Query']['rank'],LN_rank=ep['SVD']['LN']['rank'],
            Query_spectral_energy=ep['spectral_energy']['Query'],LN_spectral_energy=ep['spectral_energy']['LN'],
            actual_decoder_forwards=ep['actual_decoder_forwards'])
        for group in GROUPS:
            g=ep['groups'][group];metrics=[];curves=[]
            for c in g['candidates']:
                a=official(c['boxes'].numpy(),raw,gt[q],spans[q],interval,ds)
                independent,_,iou=dense_metric(c['boxes'].numpy(),raw,gt[q],spans[q],interval,ds)
                assert all(abs(a[k]-independent[k])<2e-10 for k in a);checks+=3
                metrics.append(a);curves.append(iou)
                candidate_rows.append(dict(**base,group=group,candidate=c['index'],sign=c['sign'],direction=c['direction'],
                    amplitude=c['amplitude'],output_RMS=c['output_RMS'],budget_attained=c['budget_attained'],
                    reward=g['rewards'][c['index']],weight=float(g['weights'][c['index']]),
                    **{k:float(v) for k,v in a.items()}))
            oracle=int(np.argmax([v['v'] for v in metrics]));expert=g['expert_top1_index']
            row=dict(**base,group=group,candidates=len(metrics),oracle_index=oracle,expert_index=expert,
                expert_top_tie_count=g['expert_top_tie_count'],all_rewards_equal=g['all_rewards_equal'],
                attained_directions=sum(c['budget_attained'] for c in g['candidates'][1:]),
                total_signed_directions=len(metrics)-1,Jensen_RMS=g['Jensen_RMS'],mean_output_RMS=g['mean_output_RMS'])
            for label,box,value,iou in [('oracle',g['candidates'][oracle]['boxes'],metrics[oracle],curves[oracle]),
                    ('expert_top1',g['candidates'][expert]['boxes'],metrics[expert],curves[expert]),
                    ('parameter_mean',g['mean_boxes'],None,None)]:
                if value is None:
                    value=official(box.numpy(),raw,gt[q],spans[q],interval,ds)
                    independent,_,iou=dense_metric(box.numpy(),raw,gt[q],spans[q],interval,ds)
                    assert all(abs(value[k]-independent[k])<2e-10 for k in value);checks+=3
                for field,val in value.items():row[label+'_'+field]=val;row[label+'_delta_'+field]=val-before_values[field]
                for name,mask in masks.items():
                    a=(iou-before_ious)[mask&event];row[label+'_'+name+'_delta_IoU']=float(a.mean()) if len(a) else None
            rows.append(row)
        status(BASE/'CPU_STATUS.json',dict(status='actual_postseal_Q0_scoring',done=len(rows)//3,total=64,CPU_only=True,GT_read=True,time=time.time()))
        print('Q0_POSTSEAL_PROGRESS',len(rows)//3,64,flush=True)
    assert len(rows)==192
    fields=[label+'_delta_v' for label in ['oracle','expert_top1','parameter_mean']]+[label+'_'+part+'_delta_IoU' for label in ['oracle','expert_top1','parameter_mean'] for part in ['observed','unobserved_decoder','interpolated','outside_native']]
    summary={}
    for ds in ['hc2','vidstg']:
        groups={}
        for group in GROUPS:
            rr=[r for r in rows if r['dataset']==ds and r['group']==group]
            assert len(rr)==32
            groups[group]=dict(cells=32,statistics={f:statistics(rr,f) for f in fields},no_expert=sum(r['no_expert'] for r in rr),
                reward_ties=sum(r['expert_top_tie_count']>1 for r in rr),all_equal_rewards=sum(r['all_rewards_equal'] for r in rr),
                attained_directions=sum(r['attained_directions'] for r in rr),total_signed_directions=sum(r['total_signed_directions'] for r in rr))
        contrasts={}
        for a,c in [('Q_observable','Q_random'),('Q_observable','LN_observable')]:
            pairs=[]
            aa={r['development_parent']:r for r in rows if r['dataset']==ds and r['group']==a}
            cc={r['development_parent']:r for r in rows if r['dataset']==ds and r['group']==c}
            for parent in aa:pairs.append(dict(development_parent=parent,**{f:aa[parent][f]-cc[parent][f] if aa[parent][f] is not None and cc[parent][f] is not None else None for f in fields}))
            contrasts[a+'_minus_'+c]={f:statistics(pairs,f) for f in fields}
        summary[ds]=dict(groups=groups,paired_contrasts=contrasts)
    for name,data in [('ROWS.jsonl.gz',rows),('CANDIDATE_ROWS.jsonl.gz',candidate_rows)]:
        with gzip.open(PUB/name,'wt') as f:
            for r in data:f.write(json.dumps(r,allow_nan=False)+'\n')
    write(PUB/'SUMMARY.json',dict(status='actual_postseal_Q0_development_diagnosis',datasets=summary,bootstrap=10000,seed=20261012,
        historically_exposed_development=True,whole_tube_oracle_not_framewise=True,no_automatic_Query_OPD_launch=True,
        online_stream_benefit_not_measured=True,unadjusted_CI=True))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_statistics_PDF_view_publication_archive',
        cells=64,group_rows=192,candidate_rows=len(candidate_rows),independent_dense_checks=checks,CPU_only=True,GT_read=True,
        files={str(p.relative_to(ROOT)):sha(p) for p in PUB.iterdir() if p.is_file()},elapsed_seconds=time.time()-tick,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        p=BASE/'CPU_failures'/str(time.time_ns());p.mkdir(parents=True,exist_ok=True);(p/'traceback.txt').write_text(traceback.format_exc());raise
