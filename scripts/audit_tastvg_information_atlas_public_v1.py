"""Independent anonymous-row aggregation audit. No private data/model imports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import sys, json, time, hashlib, collections
from pathlib import Path
import numpy as np

FIELDS=['y','y2','prediction','mse','mae','within_r2','auc','ap','prevalence','logloss']
OUTPUTS=['r2','mse','mae','within_r2','auc','ap','prevalence','logloss']
checks=collections.Counter(); errors=collections.defaultdict(float)

def read(p): return json.loads(Path(p).read_text())
def eq(actual, expected, label, tolerance=2e-8):
    if isinstance(expected,dict):
        assert set(actual)==set(expected),(label,'keys')
        for k in expected:eq(actual[k],expected[k],label,tolerance)
    elif isinstance(expected,list):
        assert len(actual)==len(expected),(label,'length')
        for a,b in zip(actual,expected):eq(a,b,label,tolerance)
    elif expected is None: assert actual is None,(label,actual)
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        assert np.isfinite(actual),(label,actual)
        error=abs(actual-expected);errors[label]=max(errors[label],error)
        assert error<=tolerance*max(1.,abs(expected)),(label,actual,expected)
    else: assert actual==expected,(label,actual,expected)
    checks[label]+=1

def avg(a):
    a=np.asarray(a,float); ok=np.isfinite(a); n=ok.sum(axis=0)
    return np.divide(np.nansum(a,axis=0),n,out=np.full(a.shape[1],np.nan),where=n>0)

def collapse(rows,key,endpoint=False):
    bins=collections.defaultdict(list)
    for r in rows:
        if endpoint:
            q=r['position_removed_endpoints'].get(key)
            v=[q['endpoint_abs_error'],q['inferred_endpoint_std']] if q else None
        else:
            q=r['metrics'].get(key)
            v=[np.nan if q[f] is None else q[f] for f in FIELDS] if q else None
        if v is not None:bins[r['source_index'],r['order'],r['condition']].append(v)
    orders=collections.defaultdict(list);sources=collections.defaultdict(list)
    for (s,o,c),v in bins.items():orders[s,o].append(avg(v))
    for (s,o),v in orders.items():sources[s].append(avg(v))
    return {s:avg(v) for s,v in sorted(sources.items())}

def resampling(src):
    a=np.array(list(src.values()),float);n=len(a)
    w=np.random.default_rng(20261003).multinomial(n,np.full(n,1/n),size=10000)/n
    ok=np.isfinite(a); count=w@ok
    boot=np.divide(w@np.nan_to_num(a),count,out=np.full(count.shape,np.nan),where=count>0)
    return a,avg(a),boot

def extract(v,f):
    if f!='r2':return v[...,FIELDS.index(f)]
    variance=v[...,1]-v[...,0]**2
    return 1-np.divide(v[...,3],variance,out=np.full(variance.shape,np.nan),where=variance>1e-12)

def finite(x): return float(x) if np.isfinite(x) else None
def interval(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return np.percentile(v,[2.5,97.5]).tolist() if len(v) else None

def summary(rows,key):
    src=collapse(rows,key)
    if not src:return dict(sources=0,metrics={})
    a,one,b=resampling(src);metrics={}
    for f in OUTPUTS:
        v=extract(b,f)
        metrics[f]=dict(mean=finite(extract(one,f)),ci95=interval(v),
            bootstrap_defined=int(np.isfinite(v).sum()),sources_defined=int(np.isfinite(extract(a,f)).sum()))
    return dict(sources=len(src),source_moments={str(k):[finite(z) for z in v] for k,v in src.items()},
                metrics=metrics,draws=10000,seed=20261003)

def pair(rows,left,right,field):
    a=collapse(rows,left);b=collapse(rows,right);ids=sorted(set(a)&set(b))
    if not ids:return dict(sources=0,mean=None,ci95=None)
    _,aone,ab=resampling({i:a[i] for i in ids});_,bone,bb=resampling({i:b[i] for i in ids})
    d=extract(ab,field)-extract(bb,field)
    return dict(sources=len(ids),mean=finite(extract(aone,field)-extract(bone,field)),
                ci95=interval(d),bootstrap_defined=int(np.isfinite(d).sum()))

def partitions(rows):
    parts={'source_validation':[r for r in rows if r['domain']=='source']}
    for mode in ['clean','corrupt']:
        rr=[r for r in rows if r['domain']=='target' and ((r['condition']=='clean')==(mode=='clean'))]
        parts['target_'+mode]=rr
        for panel in ['search','confirm']:parts['target_'+mode+'_'+panel]=[r for r in rr if r['panel']==panel]
    return parts

def endpoint_summary(rows):
    result={}
    for group,rr in partitions(rows).items():
        z={}
        for key in rr[0]['position_removed_endpoints']:
            src=collapse(rr,key,endpoint=True);a,one,b=resampling(src)
            z[key]=dict(sources=len(a),endpoint_abs_error=dict(mean=float(one[0]),ci95=interval(b[:,0])),
                        inferred_endpoint_std=dict(mean=float(one[1]),ci95=interval(b[:,1])))
        result[group]=z
    return result

def run(folder):
    tick=time.time();cfg=read(folder/'CONFIG.json');paths=read(folder/'SOURCE_PATHS.json')
    fit=read(folder/'SOURCE_FIT_SEAL.json');seal=read(folder/'GLOBAL_READOUT_SEAL.json');join=read(folder/'LABEL_JOIN.json')
    assert fit['time']<seal['time']<join['time']
    assert hashlib.sha256((folder/'SOURCE_FIT_SEAL.json').read_bytes()).hexdigest()==seal['source_fit_seal_sha256']
    assert fit['probe_hashes']==seal['probes'] and fit['target_GT_for_fit'] is False and seal['target_GT_read_in_readout'] is False
    eq(cfg['production_method_sha256'],'bd75706cf8377823ac6af488b1bd993af5fd7480e7e45a047c6fdde5e81df8d2','production_freeze')
    resource=read(folder/'RESOURCES.json')
    for f in ['new_GPU_calls','new_model_calls','new_expert_calls','new_backbone_calls','new_candidate_calls']:eq(resource[f],0,'zero_new_inference')
    eq(resource['CUDA_initialized'],False,'CPU_only')
    figure=read(folder/'FIGURE_DATA.json')
    for ds,nv,nt in [('vidstg',31,95),('hc2',16,48)]:
        rows=read(folder/ds/'ROWS.json');s=read(folder/ds/'SUMMARY.json')
        assert len(rows)==nv+144 and len({r['cell'] for r in rows})==len(rows)
        assert len(paths[ds])==68
        names=set(rows[0]['metrics']);assert len(names)==89
        for family in ['candidate','frame']:
            data=figure[ds+'/'+family]
            for g,matrix in zip(data['groups'],data['values']):
                if family=='candidate':
                    expected=[[s[g]['metrics'][f'candidate/{"Full" if v=="Full-shuffle" else v}/{t}/{"shuffle" if v=="Full-shuffle" else "real"}']['metrics']['r2']['mean']
                        for t in ['precision','recall','tiou','delta']] for v in ['Endpoint','Inside','Context','Contrast','Full','Geometry','Full-shuffle']]
                else:
                    expected=[[s[g]['metrics'][f'frame/{v}/{t}/{c}']['metrics']['auc' if t=='event' else 'r2']['mean']
                        for t in ['position','event','start_distance','end_distance','phase']] for v,c in [('Hidden','real'),('Geometry','real'),('Hidden','shuffle')]]
                eq(matrix,expected,'figure_values')
        for tag,entries in figure['paired_controls'][ds].items():
            eq(entries,[s['target_corrupt']['paired_differences'][f'candidate/Full/{t}/real/{tag}'] for t in ['precision','recall','tiou','delta']],'figure_values')
        for name,p in paths[ds].items():
            assert p['training_sources']==nt and p['validation_sources']==nv
            field='validation_logloss' if '/event/' in name else 'validation_MSE'
            i=min(range(7),key=lambda i:(p['path'][i][field],p['path'][i]['alpha']))
            eq(p['selected_index'],i,'source_validation_selection');eq(p['selected_alpha'],p['path'][i]['alpha'],'source_validation_selection')
            assert [x['alpha'] for x in p['path']]==cfg['alphas']
        for r in rows:
            assert set(r['metrics'])==names
            assert 0<=r['event_frames']<=r['frames'] and r['candidates']==32
            assert r['anchor_kind']==('native' if r['domain']=='source' else 'A8')
            for name,m in r['metrics'].items():
                if m is None:assert '/phase/' in name and r['event_frames']==0;continue
                assert m['n']==(32 if name.startswith('candidate/') else r['event_frames'] if '/phase/' in name else r['frames'])
                assert m['mse']>=0 and m['mae']>=0 and m['mse']+1e-12>=m['mae']**2
                var=m['y2']-m['y']**2;assert var>=-1e-12
                eq(m['within_r2'],1-m['mse']/var if var>1e-12 else None,'row_r2_algebra')
                if '/event/' in name:
                    eq(m['prevalence'],r['event_frames']/r['frames'],'row_event_prevalence')
                    assert (m['auc'] is not None)==r['event_both_classes']
                    assert (m['ap'] is not None)==r['event_both_classes']
                    if m['auc'] is not None:assert -1e-12<=m['auc']<=1+1e-12 and -1e-12<=m['ap']<=1+1e-12
        for group,rr in partitions(rows).items():
            coverage=dict(cells=len(rr),sources=len({r['source_index'] for r in rr}),frames=sum(r['frames'] for r in rr),
                event_frames=sum(r['event_frames'] for r in rr),no_event_support=sum(r['event_frames']==0 for r in rr),
                event_single_class=sum(not r['event_both_classes'] for r in rr))
            eq(s[group]['coverage'],coverage,'coverage')
            for key in names:eq(s[group]['metrics'][key],summary(rr,key),'anonymous_bootstrap')
            for key,value in s[group]['paired_differences'].items():
                family,view,task,control,tag=key.split('/')
                left='/'.join([family,view,task,control]);field='auc' if task=='event' else 'r2'
                right='/'.join([family,'Geometry',task,control]) if tag=='over_geometry' else '/'.join([family,view,task,'shuffle'])
                eq(value,pair(rr,left,right,field),'paired_bootstrap')
            print('ATLAS_PUBLIC_PASS',ds,group,len(rr),flush=True)
        if (folder/ds/'ENDPOINT_DIAGNOSTICS.json').exists():
            eq(read(folder/ds/'ENDPOINT_DIAGNOSTICS.json'),endpoint_summary(rows),'position_removed_bootstrap')
        if (folder/ds/'ORDER_DIAGNOSTICS.json').exists():
            for label,value in read(folder/ds/'ORDER_DIAGNOSTICS.json').items():
                mode,panel,order=label.split('/')
                rr=[r for r in rows if r['domain']=='target' and r['order']==order and (r['condition']=='clean')==(mode=='clean') and (panel=='all' or r['panel']==panel)]
                eq(value['coverage'],dict(cells=len(rr),sources=len({r['source_index'] for r in rr})),'order_coverage')
                for key,v in value['metrics'].items():eq(v,summary(rr,key),'order_bootstrap')
    receipt=dict(status='pass',checks=dict(checks),maximum_errors=dict(errors),
        evaluation_cells=335,source_queries=190,frozen_probes=136,source_cluster_bootstrap_draws=10000,
        CPU_only=True,private_assets_read=False,CPU_wall_seconds=time.time()-tick,
        scope='Independent anonymous-row coverage, alpha selections, metric algebra, source/condition/order aggregation, all bootstrap intervals and paired differences; point fitting verified separately by root audit.')
    (folder/'PUBLIC_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':run(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'results/tastvg_temporal_information_atlas/2026-10-03')
