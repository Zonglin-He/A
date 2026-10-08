"""Independent dense formula, clustered arithmetic and public scalar audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,gzip,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_paper_baseline_scoring_common_v1 import BASE,PUB,SEED,read,write,sha

def metric(boxes,row,truth,span,interval,clip):
    b=np.asarray(boxes,dtype=np.float64);ids=np.asarray(row['frame_ids'],dtype=np.int64)
    assert b.shape==(len(ids),4) and np.isfinite(b).all() and np.all(np.diff(ids)>0)
    corners=np.empty_like(b);w,h=row['input']['width'],row['input']['height']
    corners[:,0]=(b[:,0]-b[:,2]*.5)*w;corners[:,2]=(b[:,0]+b[:,2]*.5)*w
    corners[:,1]=(b[:,1]-b[:,3]*.5)*h;corners[:,3]=(b[:,1]+b[:,3]*.5)*h
    if clip: corners[corners<0]=0
    fs=np.array(sorted(truth),dtype=np.int64);gt=np.array([truth[int(f)] for f in fs],dtype=np.float64)
    right=np.searchsorted(ids,fs,side='right');lo=np.clip(right-1,0,len(ids)-1);hi=np.clip(right,0,len(ids)-1)
    den=ids[hi]-ids[lo];ratio=np.divide(fs-ids[lo],den,out=np.zeros(len(fs),dtype=float),where=den>0)
    pred=corners[lo]+ratio[:,None]*(corners[hi]-corners[lo])
    ix=np.maximum(0,np.minimum(pred[:,2],gt[:,2])-np.maximum(pred[:,0],gt[:,0]))
    iy=np.maximum(0,np.minimum(pred[:,3],gt[:,3])-np.maximum(pred[:,1],gt[:,1]))
    inter=ix*iy
    area=np.maximum(0,pred[:,2]-pred[:,0])*np.maximum(0,pred[:,3]-pred[:,1])
    ga=np.maximum(0,gt[:,2]-gt[:,0])*np.maximum(0,gt[:,3]-gt[:,1])
    iou=np.divide(inter,area+ga-inter,out=np.zeros(len(fs)),where=area+ga-inter>0)
    iou[(fs<ids[0])|(fs>ids[-1])]=0
    a,z=map(int,interval);g,k=map(int,span);assert a<z and g<k
    length=max(0,min(z,k)-max(a,g));ti=length/(z-a+k-g-length)
    v=float(iou[(fs>=max(a,g))&(fs<min(z,k))].sum()/max(max(z,k)-min(a,g),1))
    return dict(v=v,t=float(ti),s=float(iou.sum()/len(fs)))

def independent_statistics(rows,fields,boot=True):
    sources=sorted(set(r['source_id'] for r in rows));queries=sorted(set(r['query_id'] for r in rows))
    # Build query means explicitly, rather than using producer bincounts.
    q={i:[] for i in queries}
    for r in rows:q[r['query_id']].append(r)
    sums=np.zeros((len(sources),len(fields)));counts=np.zeros(len(sources));idx={s:i for i,s in enumerate(sources)}
    for qr in q.values():
        assert len(qr)==3 and len(set(r['order'] for r in qr))==3
        assert len(set(r['source_id'] for r in qr))==1
        j=idx[qr[0]['source_id']];counts[j]+=1
        sums[j]+=np.mean([[r[f] for f in fields] for r in qr],axis=0)
    means=sums/counts[:,None];a=means.mean(0);b=sums.sum(0)/counts.sum()
    ci=None;wci=None
    if boot:
        rng=np.random.default_rng(SEED);bs=[];bq=[]
        for begin in range(0,10000,100):
            choice=rng.integers(len(sources),size=(100,len(sources)))
            bs.append(means[choice].mean(1));bq.append(sums[choice].sum(1)/counts[choice].sum(1)[:,None])
        ci=np.quantile(np.concatenate(bs),[.025,.975],axis=0)
        wci=np.quantile(np.concatenate(bq),[.025,.975],axis=0)
    return dict(sources=len(sources),queries=len(queries),source_macro=a,query_macro=b,ci=ci,query_ci=wci,source_matrix=means,source_query_counts=counts)

def run():
    start=time.time();summary=read(PUB/'SUMMARY.json');checks=0
    for ds,methods in summary['datasets'].items():
        for method,s in methods.items():
            path=PUB/ds/(s['row_file']);rr=[]
            with gzip.open(path,'rt') as f:
                for line in f:rr.append(json.loads(line))
            fields=list(s['metrics']);z=independent_statistics(rr,fields)
            assert z['sources']==s['sources'] and z['queries']==s['queries']
            for j,f in enumerate(fields):
                got=s['metrics'][f]
                for x,y in [(z['source_macro'][j],got['source_macro']),(z['query_macro'][j],got['query_macro'])]:
                    assert abs(x-y)<2e-12;checks+=1
                assert np.allclose(z['ci'][:,j],got['ci95_source'],rtol=0,atol=2e-12)
                assert np.allclose(z['query_ci'][:,j],got['ci95_query_clustered'],rtol=0,atol=2e-12);checks+=4
                if f.startswith('delta_'):
                    a=z['source_matrix'][:,j]
                    assert got['harm_gt5pp_sources']==int((a<-.05).sum()) and got['harm_gt20pp_sources']==int((a<-.20).sum())
                    assert abs(got['gross_gain_pp']-np.maximum(a,0).mean()*100)<1e-10
                    assert abs(got['gross_loss_pp']+np.minimum(a,0).mean()*100)<1e-10;checks+=4
            for r in rr:
                assert abs(r['delta_total_v']-r['delta_inherited_v']-r['delta_current_v'])<2e-12
                assert abs(r['delta_total_v']-sum(r[k] for k in ['inherited_boxes_v','inherited_time_v','current_boxes_v','current_time_v']))<2e-12
                for t in [.3,.5]:
                    assert r['After_R'+str(t)]==float(r['After_v']>t)
                checks+=4
            print('INDEPENDENT_SOURCE_STATS_VERIFIED',ds,method,len(rr),flush=True)
    from scripts.decota_paper_baseline_scoring_common_v1 import verify
    verify()
    write(PUB/'ROOT_STATISTICS_AUDIT.json',dict(status='pass',checks=checks,paired_source_bootstrap=10000,
          seed=SEED,independent_query_then_source_aggregation=True,CPU_seconds=time.time()-start,
          no_model_or_GPU_execution=True,time=time.time()))
    write(BASE/'AUDIT_COMPLETION.json',dict(status='statistics_dense_and_recorded_state_audited_pending_visual_publication',
          summary_sha256=sha(PUB/'SUMMARY.json'),root_statistics_sha256=sha(PUB/'ROOT_STATISTICS_AUDIT.json'),time=time.time()))

if __name__=='__main__':run()
