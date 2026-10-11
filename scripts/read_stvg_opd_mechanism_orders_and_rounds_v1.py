"""Expose every original reward round and per-order scalar strata, without GT.

Only hash-bound saved Full fits and the already checked anonymous rows are read.
No model, new labels, re-selection or scientific modification is performed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import gzip
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,load
from scripts.analyze_stvg_opd_mechanism_review_v1 import aggregate,stats,verify
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1'
PUB=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'
QFIELDS=['expert_GT_IoU','student_expert_IoU','max_first_reward','first_zero_fraction','all_zero_fraction',
         'Full_current','Direct_current','Full_minus_Direct_current','Before_difference','Full_harm','Direct_harm']
FFIELDS=['After_minus_Before','After_minus_Native','After_minus_DINO']


def rows(name):
    with gzip.open(PUB/name,'rt') as f:return [json.loads(line) for line in f]


def check_stats(value,frames,fields):
    # Independent scalar arithmetic: frame -> arrival/condition/order -> parent.
    arrival=collections.defaultdict(list)
    for r in frames:arrival[(r['dataset'],r['source_id'],r['condition'],r['order'])].append(r)
    parents=collections.defaultdict(list)
    for k,rr in arrival.items():
        parents[k[:2]].append({f:math.fsum(r[f] for r in rr)/len(rr) for f in fields})
    assert value['parent_sources']==len(parents) and value['cells']==len(arrival)
    comparisons=2
    for f in fields:
        a=np.array([math.fsum(r[f] for r in rr)/len(rr) for rr in parents.values()])
        x=value['metrics'][f];assert x['parents']==len(a)
        if not len(a):assert x['mean'] is None;continue
        assert abs(math.fsum(a)/len(a)-x['mean'])<2e-12
        rng=np.random.default_rng(20261011);bootstrap=[]
        for _ in range(100):
            take=rng.integers(0,len(a),(100,len(a)))
            bootstrap.extend((np.sum(a[take],axis=1)/len(a)).tolist())
        ci=np.quantile(bootstrap,[.025,.975]);assert np.max(np.abs(ci-x['ci95']))<2e-12
        assert x['parent_negative_tail_5pp']==int((a<-.05).sum())
        assert abs(np.quantile(a,.05)-x['parent_p05'])<2e-12
        comparisons+=6
    return comparisons


def run():
    verify()
    assert read(BASE/'ROOT_STATISTICS_READBACK.json')['status']=='pass'
    pins={str((PUB/f).relative_to(ROOT)):sha(PUB/f) for f in
          ['QUALITY_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz','CELL_ROWS.jsonl.gz','SUMMARY.json']}
    pins.update({'artifacts/stvg_opd_mechanism_review_v1/PRIVATE_INPUT_BINDINGS.json':sha(BASE/'PRIVATE_INPUT_BINDINGS.json'),
                 'scripts/read_stvg_opd_mechanism_orders_and_rounds_v1.py':sha(Path(__file__))})
    runtime=BASE/'CPU_ORDER_ROUND_RUNTIME.json'
    write(runtime,dict(status='locked_before_saved_scalar_addendum',pins=pins,GT_read=False,new_model_calls=0,time=time.time()))
    import torch;torch.set_num_threads(2)
    quality=rows('QUALITY_ROWS.jsonl.gz');cells=rows('CELL_ROWS.jsonl.gz')
    bindings=read(BASE/'PRIVATE_INPUT_BINDINGS.json')['records'];assert len(bindings)==2*len(cells)==3584
    key=lambda r:(r['dataset'],r['source_id'],r['condition'],r['order'],r['arrival'])
    by=collections.defaultdict(list)
    for r in quality:by[key(r)].append(r)
    rounds=0;scalar_checks=0;fit_reads=0
    output=PUB/'REWARD_ROUND_ROWS.jsonl.gz'
    with gzip.open(output,'wt',encoding='utf-8') as stream:
        for i,cell in enumerate(cells):
            binding=bindings[2*i];p=ROOT/binding['path'];assert sha(p)==binding['sha256']
            z=load(p);fit=z['fit'];fit_reads+=1
            for r in by[key(cell)]:
                j=r['observation_index'];values=[]
                for at,record in enumerate(fit['rounds']):
                    reward=record['rollout']['rewards'].numpy()[j]
                    assert reward.shape==(32,) and np.isfinite(reward).all()
                    a=reward.astype(np.float64);zero=float(np.mean(a==0));values.append(a)
                    row={k:r[k] for k in ['dataset','source_id','condition','order','setting','development',
                                         'arrival','observation_index','quality','reachability']}
                    row.update(round_index=at,actual_saved_actions=32,reward_min=float(a.min()),
                               reward_max=float(a.max()),reward_mean=float(a.mean()),
                               reward_variance=float(a.var()),exact_zero_fraction=zero,
                               all_actions_zero=bool(np.all(a==0)),constant_reward=bool(a.min()==a.max()))
                    assert abs(row['reward_mean']-math.fsum(a)/32)<1e-15
                    assert abs(row['reward_variance']-math.fsum((v-row['reward_mean'])**2 for v in a)/32)<1e-15
                    assert zero==sum(v==0 for v in a)/32
                    stream.write(json.dumps(row,allow_nan=False)+'\n');rounds+=1;scalar_checks+=3
                assert len(values)==r['rounds']
                assert float(values[0].max())==r['max_first_reward']
                assert float(np.mean(values[0]==0))==r['first_zero_fraction']
                assert abs(sum(int(np.sum(a==0)) for a in values)/(32*len(values))-r['all_zero_fraction'])<1e-15
                scalar_checks+=4
            if i%256==0:print('ROUND_READBACK',i,rounds,flush=True)
    frames=rows('FRAME_ROWS.jsonl.gz');groups=['observed','unobserved_decoder','interpolated','outside_native']
    summaries={};comparisons=scalar_checks
    for setting in sorted({r['setting'] for r in cells}):
        for pop in ['all_historical','excluding_development']:
            for order in sorted({r['order'] for r in cells if r['setting']==setting}):
                select=lambda r:r['setting']==setting and r['order']==order and (pop=='all_historical' or not r['development'])
                qq=[r for r in quality if select(r)];ff=[r for r in frames if select(r)]
                qstats={};fstats={}
                for q in ['correct','incorrect','all']:
                    for a in ['zero','positive','all']:
                        rr=[r for r in qq if (q=='all' or r['quality']==q) and (a=='all' or r['reachability']==a)]
                        v=stats(aggregate(rr,['dataset','source_id','condition','order'],QFIELDS),QFIELDS)
                        comparisons+=check_stats(v,rr,QFIELDS)
                        qstats[q+'/'+a]=dict(statistics=v,expert_frame_cells=len(rr))
                obs={(r['dataset'],r['source_id']) for r in ff if r['group']=='observed'}
                for group in groups:
                    rr=[r for r in ff if r['group']==group]
                    v=stats(aggregate(rr,['dataset','source_id','condition','order'],FFIELDS),FFIELDS)
                    comparisons+=check_stats(v,rr,FFIELDS)
                    common=obs&{(r['dataset'],r['source_id']) for r in rr}
                    same=[r for r in rr if (r['dataset'],r['source_id']) in common]
                    vv=stats(aggregate(same,['dataset','source_id','condition','order'],FFIELDS),FFIELDS)
                    comparisons+=check_stats(vv,same,FFIELDS)
                    fstats[group]=dict(all_eligible=v,frame_cells=len(rr),common_with_observed=vv,
                                       common_parent_sources=len(common))
                summaries[setting+'/'+pop+'/'+order]=dict(quality=qstats,propagation=fstats)
    with gzip.open(PUB/'PER_ORDER_DIAGNOSTICS.json.gz','wt',encoding='utf-8') as stream:
        json.dump(dict(status='actual_existing_stream_order_diagnoses',statistics=summaries,
                       bootstrap=10000,seed=20261011,historically_exposed=True,
                       unadjusted_descriptive_CI=True),stream,allow_nan=False)
    for rel,h in pins.items():assert sha(ROOT/rel)==h
    write(BASE/'ROOT_ORDER_ROUND_READBACK.json',dict(status='pass',scope='per-original-order anonymous strata and complete actual saved reward-round scalars',
        complete_saved_Full_fits=fit_reads,quality_observations=len(quality),reward_round_observation_rows=rounds,
        original_order_population_cells=len(summaries),independent_scalar_comparisons=comparisons,
        runtime_sha256=sha(runtime),GT_read=False,new_model_calls=0,
        outputs={str((PUB/f).relative_to(ROOT)):sha(PUB/f) for f in
                 ['REWARD_ROUND_ROWS.jsonl.gz','PER_ORDER_DIAGNOSTICS.json.gz']},time=time.time()))
    print('ORDER_ROUND_ROOT_PASS',rounds,comparisons,flush=True)


if __name__=='__main__':run()
