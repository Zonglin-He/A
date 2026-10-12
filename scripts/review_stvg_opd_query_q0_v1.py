"""Independent postseal Q0 reward/dense/complete-row and parent statistics review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,PUB,GROUPS,read,write,status,sha,load,verify


def stats(rows,field):
    by={}
    for r in rows:
        if r.get(field) is not None:by.setdefault(r['development_parent'],[]).append(float(r[field]))
    x=np.array([math.fsum(v)/len(v) for v in by.values()])
    if not len(x):return dict(parents=0,mean=None,ci95=None)
    rng=np.random.default_rng(20261012);boot=np.empty(10000)
    for low in range(0,10000,100):boot[low:low+100]=np.sum(x[rng.integers(len(x),size=(100,len(x)))],axis=1)/len(x)
    return dict(parents=len(x),mean=math.fsum(x)/len(x),ci95=np.percentile(boot,[2.5,97.5]).tolist(),
        harm_5pp_parents=int(sum(v<-.05 for v in x)),harm_20pp_parents=int(sum(v<-.20 for v in x)))


def compare(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b);return sum(compare(v,b[k]) for k,v in a.items())
    if isinstance(a,list):
        assert len(a)==len(b);return sum(compare(x,y) for x,y in zip(a,b))
    if a is None or isinstance(a,(str,bool,int)):assert a==b
    else:assert abs(a-b)<=2e-10*(1+abs(a)),(a,b)
    return 1


def run():
    verify();b=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');c=read(BASE/'CPU_COMPLETION.json')
    assert b['status']=='sealed' and b['cells']==64 and c['cells']==64 and c['GT_read']
    for p,h in c['files'].items():assert sha(ROOT/p)==h
    import torch;torch.set_num_threads(2)
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row,unpack_expert
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from scripts.analyze_stvg_opd_mechanism_review_v1 import partition
    from vg_tta.decota_fixed_full_audit_v1 import top1_support,overlap
    with gzip.open(PUB/'ROWS.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    with gzip.open(PUB/'CANDIDATE_ROWS.jsonl.gz','rt') as f:cr=[json.loads(s) for s in f]
    indexed={(r['dataset'],r['query_ordinal'],r['group']):r for r in rows}
    candidates={(r['dataset'],r['query_ordinal'],r['group'],r['candidate']):r for r in cr}
    records=read(BASE/'DESIGN_LOCK.json')['records'];assert len(indexed)==len(rows)==192
    ds=None;checks=0;hashes=0;total_candidates=0;mean_reads=0;tick=time.time()
    for r in records:
        if ds!=r['dataset']:
            ds=r['dataset'];gt,spans,_=truths(ds,dict(parents=[v['query_ordinal'] for v in records if v['dataset']==ds]))
        p=BASE/'formal'/f"{r['index']:05}.pt";rc=read(p.with_suffix('.json'))
        assert sha(p)==rc['sha256']==b['files'][str(p.relative_to(ROOT))] and not rc['GT_read'] and rc['time']<=b['time']
        for key in ['input','original_Full']:assert sha(ROOT/r[key]['path'])==r[key]['sha256'];hashes+=1
        z=load(p);ep=z['episode'];raw=read_row(ds,r['query_ordinal']);q=r['query_ordinal']
        inp=load(ROOT/r['input']['path']);expert=unpack_expert(inp['expert']);support=top1_support(expert)
        assert [p for p,e in support]==ep['positions']
        base_values,fids,base_iou=dense_metric(ep['before'].numpy(),raw,gt[q],spans[q],z['interval'],ds)
        masks=partition(fids,raw['frame_ids'],[raw['frame_ids'][p] for p in ep['positions']],z['interval'])
        event=(fids>=spans[q][0])&(fids<spans[q][1])
        for group in GROUPS:
            g=ep['groups'][group];row=indexed[ds,q,group];metrics=[];curves=[];rewards=[]
            for cand in g['candidates']:
                values,_,iou=dense_metric(cand['boxes'].numpy(),raw,gt[q],spans[q],z['interval'],ds)
                public=candidates[ds,q,group,cand['index']]
                for k,v in values.items():checks+=compare(v,public[k])
                reward=math.fsum(overlap(cand['boxes'][p].numpy().astype(float),np.asarray(e,dtype=float)) for p,e in support)/len(support) if support else 0.
                assert abs(reward-g['rewards'][cand['index']])<3e-6
                rewards.append(g['rewards'][cand['index']]);metrics.append(values);curves.append(iou);total_candidates+=1
                if cand['sign']:
                    direction=ep['directions'][group][cand['direction']].astype(np.float32)*cand['sign']
                    expected=np.zeros(1792,dtype=np.float32)
                    expected[:256] = direction*cand['amplitude'] if group.startswith('Q_') else np.zeros(256)
                    if group=='LN_observable':expected[256:]=direction*cand['amplitude']
                    assert np.array_equal(expected,cand['delta'])
                checks+=1
            oracle=int(np.argmax([v['v'] for v in metrics]));top=int(np.argmax(rewards))
            assert row['oracle_index']==oracle and row['expert_index']==top==g['expert_top1_index']
            mean_values,_,mean_iou=dense_metric(g['mean_boxes'].numpy(),raw,gt[q],spans[q],z['interval'],ds)
            mean_reads+=1
            for label,values,iou in [('oracle',metrics[oracle],curves[oracle]),('expert_top1',metrics[top],curves[top]),('parameter_mean',mean_values,mean_iou)]:
                for field,value in values.items():
                    checks+=compare(value,row[label+'_'+field]);checks+=compare(value-base_values[field],row[label+'_delta_'+field])
                for name,mask in masks.items():
                    selected=(iou-base_iou)[mask&event];v=float(selected.mean()) if len(selected) else None
                    checks+=compare(v,row[label+'_'+name+'_delta_IoU'])
            mean_box=sum(float(g['weights'][i])*v['boxes'].double() for i,v in enumerate(g['candidates']))
            assert torch.equal(mean_box,g['box_average_for_Jensen_diagnostic'])
            assert all(torch.equal(ep['initial'][n],v) for n,v in ep['restored'].items())
        status(BASE/'ROOT_STATUS.json',dict(status='actual_independent_full_Q0_review',done=r['index']+1,total=64,GT_read=True,CPU_only=True,time=time.time()))
        print('Q0_INDEPENDENT_ROOT',r['index']+1,64,flush=True)
    summary=read(PUB/'SUMMARY.json')
    for ds,block in summary['datasets'].items():
        for group,g in block['groups'].items():
            rr=[r for r in rows if r['dataset']==ds and r['group']==group];assert len(rr)==g['cells']==32
            for field,s in g['statistics'].items():checks+=compare(stats(rr,field),s)
            assert sum(r['no_expert'] for r in rr)==g['no_expert']
            assert sum(r['expert_top_tie_count']>1 for r in rr)==g['reward_ties']
            assert sum(r['all_rewards_equal'] for r in rr)==g['all_equal_rewards']
            assert sum(r['attained_directions'] for r in rr)==g['attained_directions']
            assert sum(r['total_signed_directions'] for r in rr)==g['total_signed_directions']
        for a,c in [('Q_observable','Q_random'),('Q_observable','LN_observable')]:
            old=block['paired_contrasts'][a+'_minus_'+c];paired=[]
            for parent in [r['development_parent'] for r in rows if r['dataset']==ds and r['group']==a]:
                aa=indexed[ds,parent,a];cc=indexed[ds,parent,c]
                paired.append(dict(development_parent=parent,**{f:aa[f]-cc[f] if aa[f] is not None and cc[f] is not None else None for f in old}))
            for field,s in old.items():checks+=compare(stats(paired,field),s)
    for p,h in b['files'].items():assert sha(ROOT/p)==h;hashes+=1
    for p,h in c['files'].items():assert sha(ROOT/p)==h;hashes+=1
    write(BASE/'ROOT_STATISTICS_READBACK.json',dict(status='pass',scope='all64 exact files/source bindings, actual rewards/latent deltas/complete-tube oracle/dense/true mean/propagation/parent bootstrap/negative tails',
        scalar_checks=checks,second_read_hashes=hashes,cells=64,group_rows=192,candidates=total_candidates,
        true_parameter_mean_decodes=mean_reads,CPU_only=True,GT_read=True,new_model_calls=0,
        actual_PDF_view_complete=False,task_complete=False,independent_complete_decoder_Jacobian=False,
        elapsed_seconds=time.time()-tick,time=time.time()))


if __name__=='__main__':run()
