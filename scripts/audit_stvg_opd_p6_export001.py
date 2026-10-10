"""Portable arithmetic and anonymous bindings for all fixed P6 deployment/oracle rows.

This does not reconstruct private media, GT arrays, frozen-prefix tensors or fits.
"""
import collections
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
ARMS=['Native','ActionnessProjection','TemporalHead','SpatialOPD','Offline_GT_Head']


def read(p):return json.loads(Path(p).read_text())


def compare(a,b,count):
    if isinstance(a,dict):
        assert isinstance(b,dict) and set(a)==set(b)
        for k,v in a.items():compare(v,b[k],count)
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y,count)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isfinite(a) and isinstance(b,(float,int)) and abs(a-b)<=2e-12*(1+abs(a)),(a,b);count[0]+=1
    else:assert a==b;count[0]+=1


def recompute(rows,fields):
    source=sorted({r['source_id'] for r in rows});assert len(source)==len(rows)==32
    values=np.array([[next(r for r in rows if r['source_id']==s)[f] for f in fields] for s in source])
    rng=np.random.default_rng(20261006);boot=[]
    for _ in range(100):boot.append(values[rng.integers(0,32,(100,32))].mean(axis=1))
    lo,hi=np.quantile(np.concatenate(boot),[.025,.975],axis=0);stats={}
    for j,f in enumerate(fields):
        a=values[:,j];delta=f.startswith('delta_')
        stats[f]=dict(parent_macro=float(np.mean(a)),query_macro=float(np.mean([r[f] for r in rows])),ci95=[float(lo[j]),float(hi[j])],
            all_parent_values=a.tolist(),min=float(np.min(a)),max=float(np.max(a)),median=float(np.median(a)),
            gross_gain_pp=float(np.maximum(a,0).mean()*100) if delta else None,gross_loss_pp=float(-np.minimum(a,0).mean()*100) if delta else None,
            harm_gt5pp_parents=int((a<-.05).sum()) if delta else None,harm_gt20pp_parents=int((a<-.20).sum()) if delta else None,
            positive_parents=int((a>1e-12).sum()) if delta else None,negative_parents=int((a<-1e-12).sum()) if delta else None)
    return dict(parent_sources=32,queries=32,bootstrap_replicates=10000,seed=20261006,unit='paired original parent',multiplicity_adjusted=False,historical_cohort=True,metrics=stats)


def run(folder):
    folder=Path(folder);rows=read(folder/'ROWS.json');count=[0];assert len(rows)==64
    fields=[f for f in rows[0] if f.endswith(('_v','_t','_s'))];stats={};cases=[]
    barrier=read(folder/'receipts/P6_PREDICTION_BARRIER.json');assert barrier['status']=='sealed' and barrier['queries']==64 and barrier['logical_outputs']==256 and not barrier['GT_read']
    assert barrier['arms']==ARMS[:-1]
    rc={(r['dataset'],r['arrival']):r for r in barrier['records']};assert len(rc)==64
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds];stats[ds]=recompute(rr,fields);used=set()
        for r in rr:
            for a,b in itertools.combinations(ARMS,2):
                for m in ['v','t','s']:compare(r[b+'_'+m]-r[a+'_'+m],r[f'delta_{b}_minus_{a}_{m}'],count)
            for a in ARMS:
                for m in ['v','t','s']:assert 0<=r[a+'_'+m]<=1
            assert r['Native_t']==r['SpatialOPD_t'] and r['payload_sha256']==rc[ds,r['arrival']]['sha256']
            assert r['query_ordinal']==rc[ds,r['arrival']]['query_ordinal']
            compare(r['signal']['shrunk_temporal_tIoU'],r['TemporalHead_t'],count)
            compare(r['signal']['oracle_tIoU'],r['Offline_GT_Head_t'],count)
            assert len(r['signal']['path_tIoU'])==len(r['signal']['actual_GPU_SSL_losses'])==6
            count[0]+=6
        groups=[('temporal_gain',sorted(rr,key=lambda r:(-r['delta_TemporalHead_minus_Native_v'],r['arrival']))),
            ('temporal_harm',sorted(rr,key=lambda r:(r['delta_TemporalHead_minus_Native_v'],r['arrival']))),
            ('teacher_task_mismatch',sorted(rr,key=lambda r:(r['signal']['gradient_cosine'] if r['signal']['gradient_cosine'] is not None else 2,r['arrival'])))]
        for name,ordered in groups:
            r=next(r for r in ordered if r['query_ordinal'] not in used);used.add(r['query_ordinal']);kind=name
            if name=='temporal_gain' and r['delta_TemporalHead_minus_Native_v']<=0:kind='largest_temporal_effect_available'
            if name=='temporal_harm' and r['delta_TemporalHead_minus_Native_v']>=0:kind='least_temporal_effect_available'
            cases.append(dict(dataset=ds,kind=kind,arrival=r['arrival'],query_ordinal=r['query_ordinal'],source_id=r['source_id'],
                delta_temporal_v=r['delta_TemporalHead_minus_Native_v'],delta_temporal_t=r['delta_TemporalHead_minus_Native_t'],
                delta_spatial_v=r['delta_SpatialOPD_minus_Native_v'],gradient_cosine=r['signal']['gradient_cosine']))
    compare(stats,read(folder/'STATISTICS.json'),count);compare(cases,read(folder/'CASE_SELECTION.json')['records'],count)
    chains=read(folder/'CASE_SIGNAL_CHAINS.json')['records'];assert len(chains)==6
    for c,chain in zip(cases,chains):
        compare(c,chain['case'],count);r=next(r for r in rows if r['dataset']==c['dataset'] and r['arrival']==c['arrival'])
        compare(r['signal'],chain['temporal'],count);assert chain['temporal_payload_sha256']==r['payload_sha256']
    state=read(folder/'ROOT_MATH_STATE_DENSE_READBACK.json');n=state['counts'];assert state['status']=='pass' and state['CPU_only'] and state['new_model_calls']==0
    assert n['formal_queries']==n['complete_GPU_head_math_dicts']==n['complete_spatial_math_dicts']==n['complete_offline_GT_math_dicts']==64
    assert n['deployment_outputs']==256 and n['official_dense_scalar_checks']==64*5*3 and n['qualified_formal_complete_pairs']==4
    assert n['head_backward_rounds']==n['offline_GT_backward_rounds']==320 and n['spatial_rounds']==1600
    assert n['head_path_state_coordinates']==n['offline_GT_path_state_coordinates']==64*66306*6 and n['spatial_state_coordinates']==64*1792
    assert n['second_opaque_prediction_input_receipt_SHA_checks']==64*4 and len(state['records'])==64
    binding=read(folder/'CODE_BINDING.json')
    for rel,record in binding['public_files'].items():
        data=(folder/rel).read_bytes();assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'];count[0]+=2
    return dict(status='pass',scope='all64 anonymous P6 deployment and separately supervised offline-head arithmetic, paired parents/10000 bootstrap, negative tails/cases, original opaque bindings and public byte integrity',
        scalar_comparisons=count[0],parent_sources=64,queries=64,logical_deployment_outputs=256,offline_GT_head_fits=64,
        all_public_bytes_SHA256_verified=True,complete_negative_results_preserved=True,private_fit_GT_media_independently_reconstructed=False,
        fair_deployment_oracle=False,paper_suite_complete=False)


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),sort_keys=True))
