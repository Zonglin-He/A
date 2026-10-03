"""Paired comparisons of prelocked readout controls; no new fitted model."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_structured_separability_math_v1 import EPS,FIELDS,SIGNS,SEED,DRAWS
OUT=ROOT/'results/tastvg_structured_separability/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def matrix(rows,recipe,field,outcome):
    rr=[r for r in rows if r['eligible'] and r['label_'+outcome]!='neutral'];ids=sorted({r['source_id'] for r in rr})
    groups=[[r for r in rr if r['source_id']==s] for s in ids];h=[];f=[]
    for g in groups:
        h.append(np.array([SIGNS[field]*r['readouts'][recipe][field] for r in g if r['label_'+outcome]=='helpful']))
        f.append(np.array([SIGNS[field]*r['readouts'][recipe][field] for r in g if r['label_'+outcome]=='harmful']))
    n=len(ids);a=np.zeros((n,n));p=np.array([len(z)>0 for z in h],float);q=np.array([len(z)>0 for z in f],float)
    for i in range(n):
        for j in range(n):
            if len(h[i]) and len(f[j]):
                x=h[i][:,None]-f[j][None,:];a[i,j]=np.mean((x>0)+.5*(x==0))
    return ids,a,p,q
def contrast(rows,left,right,field,outcome):
    ids,a,p,q=matrix(rows,left,field,outcome);ids0,b,p0,q0=matrix(rows,right,field,outcome)
    assert ids==ids0 and np.array_equal(p,p0) and np.array_equal(q,q0);n=len(ids)
    w=np.random.default_rng(SEED).multinomial(n,np.full(n,1/n),size=DRAWS).astype(float) if n else np.empty((DRAWS,0))
    result={}
    for kind in ['class_source_auc','within_source_auc']:
        if kind=='class_source_auc':
            den=(w@p)*(w@q);num=np.einsum('bi,ij,bj->b',w,a-b,w);d0=p.sum()*q.sum();point=(a-b).sum()/d0 if d0 else None
        else:
            den=w@(p*q);num=w@np.diag(a-b);d0=(p*q).sum();point=np.trace(a-b)/d0 if d0 else None
        valid=den>0;vals=num[valid]/den[valid]
        result[kind]=dict(mean=float(point) if point is not None else None,ci95=np.quantile(vals,[.025,.975]).tolist() if len(vals) else None,
            bootstrap_defined=int(valid.sum()),bootstrap_undefined=int((~valid).sum()))
    return dict(left=left,right=right,field=field,outcome=outcome,sources=n,metrics=result)
def groups(rows,source,ds):
    out={'source_validation':[r for r in source if r['dataset']==ds]}
    for sp in ['search','confirm']:
        for mode in ['corrupt','clean']:
            out[sp+'_'+mode]=[r for r in rows if r['dataset']==ds and r['split']==sp and (r['condition']=='clean')==(mode=='clean')]
    return out
def main():
    start=time.time();rows=read(OUT/'ROWS.json');src=read(OUT/'SOURCE_ROWS.json');out={}
    for ds in ['vidstg','hc2']:
        out[ds]={}
        for name,rr in groups(rows,src,ds).items():
            values=[]
            for recipe,sh in [('Inside_Endpoint','Shuffle_Inside_Endpoint'),('Inside_Context','Shuffle_Inside_Context'),('Full',None)]:
                for control in ['Geometry']+([sh] if sh else []):
                    for outcome in (['t'] if name=='source_validation' else ['t','v']):
                        for field in FIELDS:values.append(contrast(rr,recipe,control,field,outcome))
            out[ds][name]=values
    p=OUT/'PAIRED_CONTROLS.json';assert not p.exists();p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    receipt=dict(time=time.time(),CPU_wall_seconds=time.time()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unchanged_locked_measurements=True,prelocked_control_recipes_only=True,no_new_fit_or_selection=True)
    (OUT/'SUPPLEMENT_BINDING.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
