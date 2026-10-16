"""Bounded final simplification study; inert imports and original source custody."""
import copy
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha,guard
from scripts.stvg_soft_ep_coordinate_common_v1 import BASE as SEARCH,roster,load_case,save_pt
BASE=ROOT/'artifacts/stvg_lite_event_support_v1'
COMP=ROOT/'artifacts/stvg_component_ablation128_v1'
RUNTIME=BASE/'EXECUTION_RUNTIME.json'


def verify():
    r=read(RUNTIME)
    assert r['status']=='locked' and r['GT_read'] is False
    for rel,digest in {**r['pins'],**r['protected']}.items():
        assert sha(ROOT/rel)==digest,rel
    return r


def event_output_positions(frame_ids,interval):
    """GT-free sparse positions used by integer-frame final Tube interpolation.

    Matches the pinned official metric's positive-coordinate int() interval
    convention and np.interp support. Includes nonzero boundary dependencies;
    outside the sparse grid there is no supported geometric output.
    """
    import bisect
    ids=list(map(int,frame_ids))
    if len(ids)<2 or any(b<=a for a,b in zip(ids,ids[1:])):
        raise ValueError('strictly increasing source grid required')
    if len(interval)!=2 or not all(math.isfinite(float(v)) for v in interval) or interval[1]<=interval[0]:
        raise ValueError('finite increasing half-open predicted interval required')
    a,z=int(interval[0]),int(interval[1])
    first,last=max(a,ids[0]),min(z-1,ids[-1])
    if last<first:return []
    lo=bisect.bisect_right(ids,first)-1
    hi=bisect.bisect_left(ids,last)
    return list(range(max(0,lo),min(len(ids)-1,hi)+1))


def contracts():
    import numpy as np
    rng=np.random.default_rng(2410);n=0
    for ids in ([0,10,20,30],[3,5,9,20,99],list(range(17))):
        for interval in ([0,1],[11,14],[10,21],[4.9,9.9],[0,100],[31,40],[1,3]):
            support=event_output_positions(ids,interval)
            boxes=rng.random((len(ids),4));changed=boxes.copy()
            outside=sorted(set(range(len(ids)))-set(support));changed[outside]+=100
            frames=np.arange(max(int(interval[0]),ids[0]),min(int(interval[1]),ids[-1]+1))
            if len(frames):
                x=np.column_stack([np.interp(frames,ids,boxes[:,k]) for k in range(4)])
                y=np.column_stack([np.interp(frames,ids,changed[:,k]) for k in range(4)])
                assert np.array_equal(x,y)
            n+=1
    assert event_output_positions([0,10,20,30],[11,14])==[1,2]
    assert event_output_positions([0,10,20,30],[10,11])==[1]
    return dict(status='pass',scope='synthetic exact event interpolation dependency, no GT/model',count=n+2,CPU_only=True,GT_read=False)


def make_trial(tid,recipe,response_scope='full_grid',multiplier=1.,temporal_path=None,reference_path=None):
    verify()
    assert recipe in {'full','lite_backtrack','lite_fullstep'}
    assert response_scope in {'full_grid','event_support'}
    b=BASE/'trials'/tid;b.mkdir(parents=True,exist_ok=False)
    final=read(COMP/'FINAL_PARAMETER_LOCK.json')
    configs={ds:{'M2':dict(final['configs'][ds],lr=final['configs'][ds]['lr']*multiplier)} for ds in ('hc2','vidstg')}
    settings={ds:{'M1':copy.deepcopy(final['response_settings'][ds]),'M2':copy.deepcopy(final['response_settings'][ds])}for ds in configs}
    for name,source in [('REFERENCE_LOCK.json',reference_path or SEARCH/'REFERENCE_LOCK.json'),('TEMPORAL_LOCK.json',temporal_path or SEARCH/'TEMPORAL_LOCK.json')]:
        source=Path(source);doc=read(source);assert doc['status']=='locked' and doc['count']==128 and not doc['GT_read'];(b/name).write_bytes(source.read_bytes())
    design=dict(status='locked',trial_id=tid,scope='finite 128 exposed-development full/lite event study',recipe=recipe,response_scope=response_scope,lr_multiplier=multiplier,configs=configs,response_settings=settings,family_reuse_from={ds:{'M2':None}for ds in configs},GT_read=False)
    write(b/'DESIGN_LOCK.json',design)
    runtime=read(RUNTIME);pins=dict(runtime['pins']);protected=dict(runtime['protected'])
    for p in [b/'DESIGN_LOCK.json',b/'REFERENCE_LOCK.json',b/'TEMPORAL_LOCK.json',SEARCH/'ROSTER_LOCK.json',SEARCH/'SOURCE_CAPTURE_BARRIER.json']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    write(b/'RUNTIME.json',dict(status='locked',trial_id=tid,pins=pins,protected=protected,GT_read=False))
    return b
