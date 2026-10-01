import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_best_full_common_v1 import *
def run():
 assert not (BASE/'RUNTIME_LOCK.json').exists()
 old=ROOT/'artifacts/tastvg_extended_sensitivity_v3';lock=read(old/'RUNTIME_LOCK.json');pins=dict(lock['pins'])
 for p in sorted((old/'revisions').glob('*.json')):pins.update(read(p)['pins'])
 for rel,h in pins.items():assert sha(ROOT/rel)==h,rel
 # Frozen old modules remain dependencies. New modules are independently pinned.
 paths=list((ROOT/'scripts').glob('*best_full*v1.py'))+[ROOT/'vg_tta/tastvg_best_full_method_v1.py',ROOT/'protocols/tastvg_best_full_v1.md']
 for p in paths:pins[str(p.relative_to(ROOT))]=sha(p)
 design=read(BASE/'DESIGN_LOCK.json');metadata=dict(design['metadata']);metadata['DESIGN_LOCK.json']=sha(BASE/'DESIGN_LOCK.json')
 for d in DATASETS:
  f=old/d/'SELECTION.json';pins[str(f.relative_to(ROOT))]=sha(f)
 write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_model_execution',pins=pins,metadata=metadata,time=time.time()))
 verify();print('RUNTIME_LOCKED',len(pins),flush=True)
if __name__=='__main__':run()
