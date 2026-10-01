"""Pin the exact unchanged method plus isolated quick-run entrypoints."""
from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_best_quick_common_v1 import *

def run():
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    previous = ROOT/'artifacts/tastvg_best_full_v1'
    lock = read(previous/'RUNTIME_LOCK.json')
    pins = dict(lock['pins'])
    for rev in sorted((previous/'revisions').glob('*.json')):
        pins.update(read(rev)['pins'])
    for rel, h in pins.items():
        assert sha(ROOT/rel) == h, rel
    for f in list((ROOT/'scripts').glob('*best_quick*v1.py')) + [ROOT/'protocols/tastvg_best_quick_v1.md']:
        pins[str(f.relative_to(ROOT))] = sha(f)
    design = read(BASE/'DESIGN_LOCK.json')
    metadata = dict(design['metadata'])
    metadata['DESIGN_LOCK.json'] = sha(BASE/'DESIGN_LOCK.json')
    write(BASE/'RUNTIME_LOCK.json', dict(status='locked_before_model_execution',
          pins=pins, metadata=metadata, time=time.time()))
    verify()
    print('QUICK_RUNTIME_LOCKED',len(pins),flush=True)

if __name__ == '__main__':
    run()
