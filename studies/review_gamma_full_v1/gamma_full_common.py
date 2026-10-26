"""Independent full gamma sensitivity; all predecessor namespaces read-only."""
import os,sys,time,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[2]
REVIEW=R/'studies/review_strengthening_v1'
sys.path.insert(0,str(REVIEW))
import review_common as review
from review_common import read,sha,atomic,identity,digest,records,binding,RUN,PROVIDER,COUNTS,selected
S=Path(__file__).parent
D=R/'outputs/review_gamma_full_v1';P=R/'results/review_gamma_full_v1'
RETAINED=review.D/'gamma_retained_v1'
ARMS=['Source','SourceJointMAP','Gamma0','Gamma025','Gamma05','Gamma075','Gamma1']
def default_arm(ds):
    gamma=read(R/'configs/default.json')['datasets'][ds]['initialization_gamma']
    return dict(zip([0,.25,.5,.75,1],ARMS[2:]))[gamma]
def pairs(ds):
    default=default_arm(ds)
    return [(a,'Source')for a in ARMS[2:]]+[(a,default)for a in ARMS[2:]if a!=default]+[(a,'SourceJointMAP')for a in ARMS[2:]]
def validate():
    p=read(D/'plan.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    review.validate();return p
def resources(start):
    p=read(D/'plan.json')['limits']
    assert time.perf_counter()-start<p['stage_seconds']
    assert shutil.disk_usage(R).free>=p['disk_free_bytes']
    assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=p['RAM_available_bytes']
def save(path,value):
    import torch
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)
def checked_current(ds,key):
    import numpy as np
    e=read(selected.D/f'components_{ds}_sealed.json')['files'][key]
    for field,h in [('path','sha256'),('receipt_path','receipt_sha256')]:assert sha(selected.D/e[field])==e[h]
    with np.load(selected.D/e['path'])as z:a={k:z[k].copy()for k in ['Source','Source_interval','SourceJointMAP','SourceJointMAP_interval','FullMAP','FullMAP_interval']}
    return a,e
def complete_arrays(compact,current):
    import numpy as np
    from gamma_cached_core_v1 import replay
    a=replay(compact)
    for arm in ['Source','SourceJointMAP']:
        a[arm]=current[arm].copy();a[arm+'_interval']=current[arm+'_interval'].copy()
    return a
def committed(ds,key):
    rp=D/ds/'receipts'/(key+'.json');e=read(rp)
    for field,h in [('path','sha256'),('compact_path','compact_sha256')]:assert sha(D/e[field])==e[h]
    return {**e,'receipt_path':str(rp.relative_to(D)),'receipt_sha256':sha(rp)}
