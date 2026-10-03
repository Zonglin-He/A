"""Additional continuous geometry, motivated after the strict-sign readback.

No threshold or selector is fitted. Original rows, metrics and radii stay fixed.
"""
import json,sys,hashlib,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_local_oracle_math_v1 import summarize
F=ROOT/'results/tastvg_temporal_local_oracle/2026-10-03'

def read(f):return json.loads(Path(f).read_text())

def compute():
    records=[];groups={};inputs={}
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            f=F/split/ds/'ROWS.json';inputs[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
            rr=[]
            for r in read(f):
                s,e=r['anchor'];x,y=r['nearest']['interval'];a=abs(x-s);b=abs(y-e);d=a+b
                z={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival','capacity_gain']}
                parts=dict(start_fraction=a/d if d>1e-12 else 0.,end_fraction=b/d if d>1e-12 else 0.,
                    balance=2*min(a,b)/d if d>1e-12 else 0.,
                    centre_component=abs((x+y)-(s+e))/d if d>1e-12 else 0.,
                    extent_component=abs((y-x)-(e-s))/d if d>1e-12 else 0.)
                z.update(parts)
                for k,v in parts.items():z['gain_'+k]=r['capacity_gain']*v
                rr.append(z);records.append(z)
            groups[ds+'/'+split]={g:summarize([r for r in rr if (r['condition']=='clean')==(g=='clean')]) for g in ['corruption','clean']}
    return dict(rows=records,summaries=groups,inputs=inputs,
        scope='posthoc_continuous_diagnostic_after_strict_sign_geometry_readback',
        rules='start_fraction=abs(ds)/L1; end_fraction=abs(de)/L1; balance=2min(abs(ds),abs(de))/L1; centre/extent components divided by L1',
        zero_distance_zero_gain_convention=0.,new_decision_thresholds=0,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def independent_verify(saved):
    count=0;largest=0.
    def eq(a,b):
        nonlocal count,largest
        assert a is None and b is None or a is not None and b is not None
        if a is None:return
        er=np.max(np.abs(np.asarray(a)-np.asarray(b)));assert er<1e-10,(a,b)
        largest=max(largest,float(er));count+=np.asarray(a).size
    for name,h in saved['inputs'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    for split in ['search','confirm']:
        for ds in ['vidstg','hc2']:
            source=read(F/split/ds/'ROWS.json');out=[r for r in saved['rows'] if r['split']==split and r['dataset']==ds]
            for r,z in zip(source,out):
                a=abs(r['nearest']['signed_start']);b=abs(r['nearest']['signed_end']);d=a+b
                parts=dict(start_fraction=a/d if d>1e-12 else 0.,end_fraction=b/d if d>1e-12 else 0.,
                    balance=2*min(a,b)/d if d>1e-12 else 0.,
                    centre_component=2*abs(r['nearest']['signed_centre'])/d if d>1e-12 else 0.,
                    extent_component=abs(r['nearest']['signed_length'])/d if d>1e-12 else 0.)
                for k,v in parts.items():eq(z[k],v);eq(z['gain_'+k],r['capacity_gain']*v)
                if r['capacity_gain']>1e-12:
                    eq(parts['start_fraction']+parts['end_fraction'],1.)
                    eq(max(parts['centre_component'],parts['extent_component']),1.)
            for g,summary in saved['summaries'][ds+'/'+split].items():
                rr=[r for r in out if (r['condition']=='clean')==(g=='clean')]
                fields=sorted(summary['metrics']);sources=sorted({r['source_id'] for r in rr});mat=[]
                for s in sources:
                    os=[]
                    for order in sorted({r['order'] for r in rr}):
                        oo=[r for r in rr if r['source_id']==s and r['order']==order]
                        if oo:os.append(np.mean([np.mean([[r[f] for f in fields] for r in oo if r['condition']==c],0) for c in sorted({r['condition'] for r in oo})],0))
                    mat.append(np.mean(os,0))
                mat=np.array(mat);rng=np.random.default_rng(20261003);boots=mat[rng.integers(0,len(mat),(10000,len(mat)))].mean(1)
                at={f:i for i,f in enumerate(fields)};bd=boots[:,at['capacity_gain']];ok=bd>1e-12
                for f,z in summary['metrics'].items():
                    eq(z['mean'],mat[:,at[f]].mean());eq(z['ci95'],np.percentile(boots[:,at[f]],[2.5,97.5]))
                for f,z in summary['capacity_gain_shares'].items():
                    den=mat[:,at['capacity_gain']].mean()
                    eq(z['mean'],mat[:,at[f]].mean()/den if den>1e-12 else None)
                    eq(z['ci95'],np.percentile(boots[ok,at[f]]/bd[ok],[2.5,97.5]) if ok.any() else None)
                    eq(z['bootstrap_zero_denominator_draws'],int((~ok).sum()))
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==saved['script_sha256']
    return dict(status='pass',independent_continuous_geometry_source_bootstrap_checks=int(count),max_error=largest,
        new_predictions=0,new_GT_annotation_reads=0,decision_thresholds_fit=0)

if __name__=='__main__':
    file=F/'ENDPOINT_BALANCE.json'
    if '--verify' in sys.argv:print(json.dumps(independent_verify(read(file)),indent=2))
    else:
        assert not file.exists();file.write_text(json.dumps(compute(),indent=2,allow_nan=False)+'\n')
        print('CONTINUOUS_ENDPOINT_BALANCE_RECORDED')
