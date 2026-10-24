"""Independent CG calibration and published TTA follow-up. No production edits."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
import sys,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
S=Path(__file__).parent
sys.path[:0]=[str(R),str(R/'tools'),str(R/'studies/paper_review_suite_v1')]
from full import RUN,PROVIDER,read,sha,atomic,identity,check
D=R/'outputs/paper_followup_v1';P=R/'results/paper_followup_v1';OLD=R/'outputs/paper_review_suite_v1'
SEED=20261025

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def freeze(name,p,code):
    p.update(code_sha256={n:sha(S/n) for n in code},production_plan_sha256=sha(RUN/'plan.json'))
    p['fingerprint']=digest(p)
    assert not(D/(name+'_plan.json')).exists()
    atomic(D/(name+'_plan.json'),p);atomic(P/(name+'_protocol.json'),p)
    for n in code:
        dst=D/'source_snapshot'/(name+'_'+n);assert not dst.exists();dst.write_bytes((S/n).read_bytes())
    return p

def validate(name):
    p=read(D/(name+'_plan.json'));assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
    for n,h in p['code_sha256'].items():assert sha(S/n)==h,n
    assert sha(RUN/'plan.json')==p['production_plan_sha256'];check()
    return p

def records(ds):
    job=read(RUN/'plan.json')['jobs'][ds];path=PROVIDER/job['manifest'];assert sha(path)==job['manifest_sha256']
    return list(map(json.loads,path.read_text().splitlines()))

def labels(ds):
    job=read(RUN/'plan.json')['jobs'][ds];path=PROVIDER/job['labels'];assert sha(path)==job['labels_sha256']
    return {x['id']:x for x in map(json.loads,path.read_text().splitlines()) if x['dataset']==ds}

def report(rows,arms,pairs,folder):
    import csv,numpy as np
    from full_score import cluster
    means={a:{f:float(np.mean([r['values'][a][f]for r in rows if r['values'][a][f] is not None])) if any(r['values'][a][f]is not None for r in rows)else None for f in rows[0]['values'][a]}for a in arms}
    severe={a:dict(t=sum(r['values'][a]['t']-r['values']['Source']['t']<-.05 for r in rows),v=sum(r['values'][a]['v']-r['values']['Source']['v']<-.05 for r in rows),union=sum(any(r['values'][a][f]-r['values']['Source'][f]<-.05 for f in ['t','v'])for r in rows))for a in arms}
    result=dict(queries=len(rows),parents=len({r['parent']for r in rows}),means=means,severe_vs_Source=severe,paired={a+'-'+b:cluster(rows,a,b)for a,b in pairs},CI='pointwise parent bootstrap; no familywise correction')
    result['timegroups']={group:dict(queries=len(rr),paired={a+'-'+b:cluster(rr,a,b)for a,b in pairs})for group in ['short_le5s','long_gt5s']if(rr:=[r for r in rows if r['duration_group']==group])}
    folder.mkdir(parents=True,exist_ok=True);atomic(folder/'report.json',result)
    with(folder/'all_signed.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','duration_group']+[a+'_'+k for a in arms for k in ['t','v','s','observed_actual_sIoU','unobserved_actual_sIoU']]);w.writerows([[f'Q{i:03}',r['duration_group']]+[r['values'][a][k]for a in arms for k in ['t','v','s','observed_actual_sIoU','unobserved_actual_sIoU']]for i,r in enumerate(rows,1)])
    return result
