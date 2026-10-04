"""New opposite-checkpoint qualification, separate from every historical queue."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import BASE,PUB,DATASETS,budget,archive,r1
from scripts.decota_matrix_common_v1 import read,write,status,save,load,sha
CROSS=BASE/'cross_domain'

def verify():
    from scripts.decota_actuation_scope_common_v1 import verify as parent_verify
    parent_verify();p=read(CROSS/'RUNTIME_LOCK.json');pins=dict(p['pins'])
    for f in sorted((CROSS/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    for f,h in {**pins,**p['inputs'],**p['protected']}.items():assert sha(ROOT/f)==h,f
    return p

def commit(f,x):
    save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f),GT_read=False,cross_lock_sha256=sha(CROSS/'RUNTIME_LOCK.json'),time=time.time()))

def checked(f):
    r=read(f.with_suffix('.json'));assert sha(f)==r['sha256'] and not r['GT_read'] and r['cross_lock_sha256']==sha(CROSS/'RUNTIME_LOCK.json');return load(f)

def cached(ds,parent,cond):
    from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
    f=CROSS/ds/'capture'/cond/f'{parent:05}.pt';return device_tree(checked(f),'cuda'),read(f.with_suffix('.json'))

def model_for(ds):
    from scripts.run_spatial_regression_alignment_v1 import model_load
    # Opposite source checkpoint, not the earlier within-domain model_for helper.
    return model_load('vidstg_test' if ds=='vidstg' else 'hcstvg1_test').eval().requires_grad_(False)
