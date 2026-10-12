"""Independent full CPU readback of the cached eight-arm study and Q0 algebra."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import math
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_stvg_opd_temporal_spatial_v1 import BASE,OUT,O,Q,SPATIAL,GROUPS,read,write,status,sha,load,array,rows_read,verify,require_seal

def near(a,b,tol=2e-10):
    if a is None or b is None:assert a is b;return 1
    if isinstance(a,dict):
        assert set(a)==set(b);return sum(near(v,b[k],tol) for k,v in a.items())
    if isinstance(a,list):
        assert len(a)==len(b);return sum(near(x,y,tol) for x,y in zip(a,b))
    if isinstance(a,(str,bool,int)):assert a==b
    else:assert abs(a-b)<=tol*(1+abs(a)),(a,b)
    return 1

def independent_boot(values):
    x=np.asarray(values,float)
    if not len(x):return dict(count=0,mean=None,ci95=None,median=None,min=None,max=None)
    gen=np.random.default_rng(20261012)
    indices=gen.integers(0,len(x),size=(10000,len(x)))
    sample=np.sum(x[indices],axis=1)/len(x)
    return dict(count=len(x),mean=math.fsum(x)/len(x),ci95=np.percentile(sample,[2.5,97.5]).tolist(),
        median=float(np.median(x)),min=float(min(x)),max=float(max(x)),harm_5pp=sum(v<-.05 for v in x),
        harm_20pp=sum(v<-.2 for v in x),positive=sum(v>1e-12 for v in x),negative=sum(v< -1e-12 for v in x))

def temporal(z):
    from scripts.stvg_opd_p6_existing_common001 import config
    cfg=config(z['dataset']);cap=z['capture'];intervals=[];scalars=0
    for logit,action,record in zip(cap['zero']['logits'],cap['zero']['actions'],cap['records']):
        v=array(logit).reshape(-1,2);x=array(action).astype(np.float32).astype(float).reshape(-1);f=np.array(record['frame_ids'])
        ij=np.triu_indices(len(x),1);native=v[ij[0],0]+v[ij[1],1];m=max(native)
        lp=native-(m+np.log(np.sum(np.exp(native-m))))
        med=np.median(x);mad=np.median(np.abs(x-med));u=(x-cfg.center_fraction*med)/(mad+cfg.epsilon)
        edge=np.r_[f[0],(f[1:]+f[:-1])/2,f[-1]+1];w=np.diff(edge)/(edge[-1]-edge[0])
        prefix=np.r_[0.,np.cumsum(w*u)];cost=np.dot(w,np.logaddexp(0.,u))-(prefix[ij[1]+1]-prefix[ij[0]])
        at=int(np.argmax(-cost+cfg.prior_weight*lp));intervals.append([int(f[ij[0][at]]),int(f[ij[1][at]]+1)]);scalars+=len(native)+len(x)
    return [min(v[0] for v in intervals),max(v[1] for v in intervals)],scalars

def run():
    import torch
    torch.set_num_threads(2);verify();barrier=require_seal();completion=read(BASE/'CPU_COMPLETION.json')
    assert completion['GT_read'] and read(BASE/'Q0_CANCELLATION_READBACK.json')['status']=='pass'
    from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate
    activate()
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    rows=rows_read(OUT/'ROWS.jsonl.gz');design=read(BASE/'DESIGN_LOCK.json');gt={};spans={}
    for ds in ['hc2','vidstg']:
        gt[ds],spans[ds],_=truths(ds,dict(parents=sorted({r['query_ordinal'] for r in design['records'] if r['dataset']==ds})))
    checks=0;bindings=0;coordinates=0;projection=0;bytes_read=0
    for index,(r,row) in enumerate(zip(design['records'],rows)):
        p=BASE/'readouts'/f'{index:05}.pt';h=sha(p);assert h==barrier['files'][str(p.relative_to(ROOT))];z=load(p)
        ds=z['dataset'];q=z['query_ordinal'];raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q]
        p6file=ROOT/z['temporal_binding']['path'];assert sha(p6file)==z['temporal_binding']['sha256'];cap=load(p6file)
        interval,n=temporal(cap);assert interval==z['intervals']['T1'];projection+=n
        assert z['intervals']['T0']==cap['predictions']['Native']['physical_interval']
        assert torch.equal(z['boxes']['Native'],cap['predictions']['Native']['boxes']);coordinates+=z['boxes']['Native'].numel()
        for arm,b in z['spatial_bindings'].items():
            f=ROOT/b['path'];assert sha(f)==b['sha256'];old=load(f)
            assert torch.equal(z['boxes'][arm],old['fit']['final']);coordinates+=old['fit']['final'].numel();bytes_read+=f.stat().st_size
            assert sha(f)==b['sha256'];bindings+=2
        for arm,box in z['boxes'].items():
            for time_arm,interval in z['intervals'].items():
                values,ids,iou=dense_metric(array(box),raw,truth,span,interval,ds)
                for m,v in values.items():checks+=near(v,row[f'{arm}_{time_arm}_{m}'])
                for threshold in [.3,.5]:checks+=near(float(values['v']>threshold),row[f'{arm}_{time_arm}_R{int(threshold*100)}'])
        assert sha(p)==h and sha(p6file)==z['temporal_binding']['sha256'];bindings+=4
        status(BASE/'ROOT_STATUS.json',dict(status='actual_complete_source_interval_dense_readback',done=index+1,total=192,
            CPU_only=True,GT_read=True,time=time.time()))
    summary=read(OUT/'SUMMARY.json')
    for panel,data in summary.items():
        for ds,block in data.items():
            subset=[r for r in rows if r['panel']==panel and r['dataset']==ds];parents=collections.defaultdict(list)
            for row in subset:parents[row['parent_id']].append(row)
            assert len(parents)==block['parents']==32 and len(subset)==block['cells']
            for f,stat in block['metrics'].items():
                x=[math.fsum(r[f] for r in rr)/len(rr) for rr in parents.values()]
                checks+=near(independent_boot(x),stat)
    qrows=rows_read(OUT/'Q0_GROUP_ROWS.jsonl.gz');pairs=rows_read(OUT/'Q0_PAIR_ROWS.jsonl.gz');qstat=read(OUT/'Q0_DIAGNOSTIC_SUMMARY.json')
    qi={(r['parent_id'],r['group']):r for r in qrows};pair_index={(r['parent_id'],r['group'],r['direction']):r for r in pairs}
    for r in read(Q/'DESIGN_LOCK.json')['records']:
        f=Q/'formal'/f"{r['index']:05}.pt";h=sha(f);assert h==read(f.with_suffix('.json'))['sha256'];ep=load(f)['episode'];pos=ep['native_positions']
        for group in GROUPS:
            g=ep['groups'][group];rr=qi[r['index'],group];cs=g['candidates'];rewards=np.array(g['rewards']);tau=ep['tau']
            scaled=rewards/tau;m=float(scaled.max());w=np.exp(scaled-m);w/=np.sum(w);d=np.stack([array(c['delta']) for c in cs]);total=np.zeros(1792)
            for i in range(len(cs)):total+=g['weights'][i]*d[i]
            assert np.array_equal(total.astype(np.float32),g['mean_delta']);checks+=1792
            norms=np.sqrt((d*d).sum(1));norm=float(np.sqrt(np.sum(array(g['mean_delta'])**2)))
            checks+=near(norm,rr['mean_norm']);checks+=near(float(np.dot(g['weights'],norms)),rr['weighted_absolute_norm'])
            for c in cs:
                if c['sign']!=1:continue
                minus=next(x for x in cs if x['direction']==c['direction'] and x['sign']==-1)
                i,j=c['index'],minus['index'];record=pair_index[r['index'],group,c['direction']]
                checks+=near((rewards[i]-rewards[j])/tau,record['reward_margin_over_tau'])
                ratio=(w[i]-w[j])/(w[i]+w[j]);checks+=near(abs(ratio),record['tanh_retention'])
                v=w[i]*d[i]+w[j]*d[j];checks+=near(float(np.linalg.norm(v)),record['pair_mean_norm'])
            wb=np.zeros_like(array(g['mean_boxes']))
            for i,c in enumerate(cs):wb+=g['weights'][i]*array(c['boxes'])
            diff=array(g['mean_boxes'])-wb
            for field,a in [('native_Jensen_RMS',diff[pos]),('full_decoder_Jensen_RMS',diff)]:checks+=near(float(np.sqrt(np.sum(a*a)/a.size)),rr[field],tol=3e-9)
        assert sha(f)==h;bindings+=2;bytes_read+=f.stat().st_size
    for ds,data in qstat.items():
        for group,block in data.items():
            rr=[r for r in qrows if r['dataset']==ds and r['group']==group]
            for f,stat in block['statistics'].items():
                vals=[r[f] for r in rr if r[f] is not None]
                expected={**independent_boot(vals),'undefined':sum(r[f] is None for r in rr)}
                checks+=near(expected,stat)
    for f,h in barrier['source_files'].items():assert sha(ROOT/f)==h;bindings+=1
    for f,h in completion['files'].items():assert sha(ROOT/f)==h;bindings+=1
    write(BASE/'ROOT_READBACK.json',dict(status='pass',scope='all192 cached eight-way outputs/sources/independent interval/dense/paired statistics and all64 Q0 algebra',
        cells=192,logical_outputs=1536,distinct_parents=64,coordinate_checks=coordinates,
        scalar_checks=checks,projection_values=projection,second_hash_checks=bindings,source_bytes_read=bytes_read,
        Q0_queries=64,Q0_groups=192,Q0_pairs=1308,CPU_only=True,GT_read=True,new_GPU_fits=0,new_model_calls=0,
        unavailable_full_O_extension_completed=False,pending_actual_PDF_view_publication_archive=True,time=time.time()))
    print('ROOT_READBACK_PASS',checks,bindings,flush=True)

if __name__=='__main__':run()
