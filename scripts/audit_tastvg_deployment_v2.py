"""Independent saved-state audit, including unchanged-input controls."""
import sys,json,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
OUT=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2';OLD=ROOT/'artifacts/tastvg_corruption_c0c1_v1'

def check_stats(values,summary):
    a=np.array(values,float);assert len(a)==summary['n']
    assert abs(float(a.mean())-summary['mean'])<1e-12
    rng=np.random.default_rng(20260929);boot=np.mean(a[rng.integers(len(a),size=(10000,len(a)))],axis=1)
    assert np.allclose(np.quantile(boot,[.025,.975]),summary['ci95'],rtol=0,atol=1e-12)

def run():
    p=read(OUT/'LOCK.json');panel=read(OUT/'GENERATION.json');assert not panel['interval_GT_for_generation'] and not panel['query_used']
    refs=read(OLD/'PREDICTION_BARRIER.json')['files'];nohit=0;count=0
    for r in p['rows']:
        rel=f"capture/clean/{r['ordinal']:03}.pt";assert sha(OLD/rel)==refs[rel];clean=load(OLD/rel)
        seed=int(hashlib.sha256(('DeploymentBurst-20260929|'+r['source']).encode()).hexdigest()[:16],16);u=np.random.default_rng(seed).random()
        for pct in p['levels']:
            z=panel['panel'][str(r['ordinal'])][str(pct)];T=r['input']['frame_count'];L=(T*pct+99)//100;s=int(u*(T-L+1))
            assert z['physical_start']==s and z['physical_end']==s+L
            selected=[i for i,f in enumerate(r['frame_ids']) if s<=f<s+L];assert selected==z['positions']
            for fam in p['families']:
                x=load(OUT/'c05'/f'{fam}_{pct}'/f"{r['ordinal']:03}.pt");assert x['generation']==z
                assert set(x['actual_changed_positions'])<=set(selected);count+=1
                if not selected:
                    assert x['pixel_sha256']==clean['pixel_sha'];assert x['batch_sha256']==clean['batch_pixels_sha'];a,b=x['native'],clean['native']
                    assert a['indices']==b['indices'] and torch.equal(a['boxes'],b['boxes']) and all(torch.equal(v,w) for v,w in zip(a['logits'],b['logits']));nohit+=1
    rows=read(OUT/'analysis/C05_ROWS.json');summary=read(OUT/'analysis/C05_SUMMARY.json');n=0
    for name,v in summary.items():
        rr=rows if name=='panel' else [r for r in rows if r['family']==name or r['condition']==name]
        for m in ['sIoU','tIoU','vIoU_corrected']:
            d=collections.defaultdict(list)
            for r in rr:d[r['parent']].append(r['delta'][m]);assert abs(r['delta'][m]-(r['native'][m]-r['clean'][m]))<1e-12
            check_stats([np.mean(d[k]) for k in sorted(d)],v['delta'][m]);n+=1
    write(OUT/'INDEPENDENT_C05_AUDIT.json',dict(status='pass',cells=count,unchanged_input_exact_controls=nohit,scalar_bootstrap_checks=n,generator_independent_of_query_and_GT=True))
    print('PASS',count,nohit,n)

if __name__=='__main__':run()
