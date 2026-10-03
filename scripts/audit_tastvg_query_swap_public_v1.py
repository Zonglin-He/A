"""Independently recompute anonymous source statistics and paired intervals."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import sys,json,hashlib,collections
from pathlib import Path
import numpy as np
FIELDS=['y','y2','prediction','mse','mae','within_r2','auc','ap','prevalence','logloss']
checks=0;maximum=0.

def read(p):return json.loads(Path(p).read_text())
def equal(a,b):
    global checks,maximum
    checks+=1
    if a is None or b is None:assert a is None and b is None,(a,b);return
    e=abs(float(a)-float(b));maximum=max(maximum,e)
    assert np.isfinite(e) and e<=2e-10*max(1,abs(float(b))),(a,b,e)
def nanmean(a,axis=0):
    a=np.asarray(a,float);mask=np.isfinite(a);n=mask.sum(axis);s=np.nansum(a,axis)
    return np.divide(s,n,out=np.full(np.shape(s),np.nan),where=n>0)
def source_arrays(rows,name):
    cells=collections.defaultdict(list)
    for r in rows:
        m=r['metrics'][name]
        cells[r['source_index'],r['order'],r['condition']].append([
            np.nan if m[x] is None else m[x] for x in FIELDS])
    orders=collections.defaultdict(list);sources=collections.defaultdict(list)
    for (s,o,c),vals in cells.items():orders[s,o].append(nanmean(vals))
    for (s,o),vals in orders.items():sources[s].append(nanmean(vals))
    return {s:nanmean(vals) for s,vals in sorted(sources.items())}
def values(a,field):
    a=np.asarray(a,float)
    if field=='r2':
        den=a[...,1]-a[...,0]**2
        return 1-np.divide(a[...,3],den,out=np.full(den.shape,np.nan),where=den>1e-12)
    return a[...,FIELDS.index(field)]
def sample(a,w):
    mask=np.isfinite(a);den=w@mask
    return np.divide(w@np.nan_to_num(a),den,out=np.full(den.shape,np.nan),where=den>0)
def finite(x):return float(x) if np.isfinite(x) else None
def interval(x):
    x=x[np.isfinite(x)];return np.quantile(x,[.025,.975]).tolist() if len(x) else None
def group(rows,key):
    mode='corrupt' if key.startswith('target_corrupt') else 'clean'
    rr=[r for r in rows if (r['condition']=='clean')==(mode=='clean')]
    tail=key[len('target_'+mode):].lstrip('_')
    if tail in ['search','confirm']:rr=[r for r in rr if r['panel']==tail]
    elif tail:rr=[r for r in rr if r['order']==tail]
    return rr

def audit(folder):
    folder=Path(folder);cfg=read(folder/'CONFIG.json');mapping=read(folder/'QUERY_MAPPING.json')
    assert cfg['probes_retrained']==0 and cfg['new_candidates']==0 and cfg['new_expert_calls']==0
    assert not cfg['GT_for_donor_or_probe_selection'] and not cfg['donor_mismatch_certified']
    for d in ['vidstg','hc2']:
        mm=[r for r in mapping if r['dataset']==d]
        assert len(mm)==48 and len({r['source_index'] for r in mm})==len({r['donor_index'] for r in mm})==48
        for r in mm:
            assert r['source_index']!=r['donor_index'] and r['true_caption_sha256']!=r['swap_caption_sha256']
            assert r['different_source'] and r['different_video_hash'] and r['different_normalized_caption']
        rows=read(folder/d/'ROWS.json');summary=read(folder/d/'SUMMARY.json')
        assert len(rows)==144 and sum(r['condition']=='clean' for r in rows)==24
        assert len({r['cell'] for r in rows})==144
        assert len({r['source_index'] for r in rows})==cfg['target_sources'][d]
        for r in rows:
            assert r['candidates']==32 and r['frames']>=9
            for name,m in r['metrics'].items():
                arm,fam,view,task,control=name.split('/')
                assert m['n']==(r['frames'] if fam=='frame' else 32)
                assert m['mse']>=-1e-12 and m['mae']>=-1e-12
                if m['auc'] is not None:assert -1e-12<=m['auc']<=1+1e-12
                if m['ap'] is not None:assert -1e-12<=m['ap']<=1+1e-12
                if arm=='true':
                    other=r['metrics']['swap/'+name[5:]]
                    for f in ['y','y2','n']:equal(m[f],other[f])
                    if view in ['Geometry','Null']:
                        for f in FIELDS:equal(m[f],other[f])
        for key,stored in summary.items():
            rr=group(rows,key);assert len(rr)==stored['coverage']['cells']
            assert len({r['source_index'] for r in rr})==stored['coverage']['sources']
            arrays={};weights={}
            for name,part in stored['metrics'].items():
                ss=source_arrays(rr,name);ids=list(ss);a=np.array(list(ss.values()));arrays[name]=a
                n=len(ids);assert part['sources']==n
                if n not in weights:
                    weights[n]=np.random.default_rng(20261003).multinomial(n,np.ones(n)/n,size=10000)/n
                draws=sample(a,weights[n]);point=nanmean(a)
                for sid,vec in ss.items():
                    for i,z in enumerate(vec):equal(finite(z),part['source_moments'][str(sid)][i])
                for f,x in part['metrics'].items():
                    p=values(point,f);b=values(draws,f);equal(finite(p),x['mean'])
                    ci=interval(b)
                    if ci is None:assert x['ci95'] is None
                    else:
                        equal(ci[0],x['ci95'][0]);equal(ci[1],x['ci95'][1])
                    assert int(np.isfinite(b).sum())==x['bootstrap_defined']
                    assert int(np.isfinite(values(a,f)).sum())==x['sources_defined']
            for name,ff in stored['paired_true_minus_swap'].items():
                a=arrays['true/'+name];b=arrays['swap/'+name];n=len(a);assert n==len(b)
                aa=sample(a,weights[n]);bb=sample(b,weights[n])
                for field,x in ff.items():
                    bs=values(aa,field)-values(bb,field)
                    point=values(nanmean(a),field)-values(nanmean(b),field)
                    equal(finite(point),x['mean']);ci=interval(bs)
                    if ci is None:assert x['ci95'] is None
                    else:equal(ci[0],x['ci95'][0]);equal(ci[1],x['ci95'][1])
                    assert int(np.isfinite(bs).sum())==x['bootstrap_defined']
    seal=read(folder/'GLOBAL_READOUT_SEAL.json');join=read(folder/'LABEL_JOIN.json')
    assert seal['time']<join['time'] and not seal['GT_read']
    assert join['only_original_query_cached_span'] and not join['donor_GT_read']
    resources=read(folder/'RESOURCES.json')
    assert resources['training_calls']==resources['new_experts']==resources['backward_calls']==resources['new_candidates']==0
    assert resources['total_new_encoder_inputs']==278 and resources['total_new_backbone_offset_forwards']==556
    return dict(status='passed',checks=checks,maximum_numeric_error=maximum,
        independent_source_aggregation=True,independent_10000_paired_bootstrap=True,
        fixed_geometry_control=True,coverage_and_seal_order=True,
        scope='anonymous metric readback; not independent GPU inference or original-annotation audit')

if __name__=='__main__':
    print(json.dumps(audit(sys.argv[1]),indent=2,allow_nan=False))
