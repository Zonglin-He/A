"""O2 cached critic only: budgeted evidence first, full reference after online seal."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,load,save,sha
from scripts.run_tastvg_regime_online_capture_v1 import OUT,verify
PREV=ROOT/'artifacts/tastvg_temporal_fourarm_v1'


def run(phase):
    tick=time.monotonic();p=verify();assert phase in ['online','full']
    cb=read(OUT/'CAPTURE_BARRIER.json');old=read(PREV/'C2_BARRIER.json');pins=read(PREV/'C2_LOCK.json')['pins']
    pins={k:v for k,v in pins.items() if k.endswith(('.pth','.pt','opt.yaml','universal_vtg_inference.py'))}
    for f,h in pins.items():assert sha(f)==h,f
    if phase=='full':assert (OUT/'ONLINE_BARRIER.json').exists()
    else:assert not (OUT/'ONLINE_BARRIER.json').exists()
    rows=[r for r in p['rows'] if r['expert']==(phase=='online')];assert len(rows)==(20 if phase=='online' else 60)
    for r in rows:
        pos=r['position'];rel=f'capture/{pos:03}.pt';assert sha(OUT/rel)==cb['files'][rel];x=load(OUT/rel)
        oldrel=f"c2/{r['condition']}/{r['ordinal']:03}.pt";assert sha(PREV/oldrel)==old['files'][oldrel];z=load(PREV/oldrel)
        assert z['pixel_sha256']==x['pixel_sha256'] and z['candidate_intervals']==[c['physical_interval'] for c in x['candidates']]
        f=OUT/'teacher'/phase/f'{pos:03}.pt';assert not f.exists()
        save(f,dict(position=pos,phase=phase,scores=z['scores'],selected=z['selected'],candidate_intervals=z['candidate_intervals'],proposals=z['proposals'],proposal_confidence=z['proposal_confidence'],pixel_sha256=z['pixel_sha256'],reused=True,reused_sha256=old['files'][oldrel],GT_read=False))
    write(OUT/f'TEACHER_{phase.upper()}_BARRIER.json',dict(phase=phase,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'teacher'/phase).glob('*.pt')},logical_calls=len(rows),new_evidence=0,reused=len(rows),pins=pins,online_barrier_sha256=sha(OUT/'ONLINE_BARRIER.json') if phase=='full' else None,seconds=time.monotonic()-tick,time=time.time()))
    print('CACHED TEACHER SEALED',phase,len(rows))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['online','full']);run(p.parse_args().phase)
