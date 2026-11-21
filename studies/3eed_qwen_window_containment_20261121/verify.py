"""Recount containment with the original 3EED corner implementation on CPU."""
import csv
import sys
from pathlib import Path
import numpy as np
from projection import (BASE, HERE, PUBLIC, PRIVATE, TOL, camera, digest, inputs, raw_window,
                        read, rigid_xyz, write)
from geometry import sensor_pixels
from summarize import PLATFORMS, parse

UPSTREAM = HERE.parents[1]/'crossplatform/upstream/utils'
sys.path.insert(0,str(UPSTREAM))
from eval_det import bbox_7d9d_to_corners


def reference(box, spec, size, mat, window):
    if box is None or window is None:
        return None
    corners=bbox_7d9d_to_corners(np.array(box,dtype=np.float32))
    uv=sensor_pixels(rigid_xyz(corners,mat),spec)
    if uv is None:
        return None
    lo=np.maximum(uv.min(axis=0),0)
    hi=np.minimum(uv.max(axis=0),size)
    if np.any(hi<=lo):
        return None
    return bool(np.all(window[:2]>=lo-TOL) and np.all(window[2:]<=hi+TOL))


def main():
    seal=read(PUBLIC/'scoring_seal.json')
    assert seal['outcome_rows']==digest(PUBLIC/'rows.csv') and seal['score_code']==digest(HERE/'score_projection.py')
    panel, frozen=inputs()
    with (PUBLIC/'rows.csv').open() as f:
        data=[{k:parse(v) for k,v in r.items()} for r in csv.DictReader(f)]
    assert len(data)==len(panel)==len({r['case'] for r in data})==416
    equal=unknown=0
    for f,r in zip(sorted(panel,key=lambda x:x['uid']),data,strict=True):
        assert f['target']==r['target'] and f['separation']==r['separation']
        p=frozen[f['uid']]
        spec,size,mat=camera(f)
        w=raw_window(p['pixel'],size)
        raw=read(f['calibration'])['ground_info'][f['slot']]['bbox_3d']
        for tag,b in [('native',raw),('truth',raw[:7]),('frozen',p['boxes']['H'])]:
            v=reference(b,spec,size,mat,w)
            assert v==r[tag+'_pass'],(r['case'],tag,v,r[tag+'_pass'])
            equal+=1
            unknown+=v is None
    sel=[r for r in data if r['selected']]
    rates=[]
    for target in PLATFORMS:
        tt=[r for r in sel if r['target']==target]
        group=[[r for r in tt if r['frozen_pass'] is k] for k in (False,True)]
        rates.append([sum(r['iou']<=.5 for r in g)/len(g) for g in group])
    delta=100*np.mean(np.subtract(*np.array(rates).T))
    d=read(HERE/'DECISION.json')
    assert abs(delta-d['platform_stratified_recording_bootstrap']['selected_failure_difference_pp']['estimate'])<1e-10
    assert (sum(r['truth_pass'] is True for r in data)>sum(r['truth_pass'] is False for r in data))==d['majority_of_true_rectangles_cover']
    assert d['run_correction']==(delta>0 and d['majority_of_true_rectangles_cover'])
    files=('summary.csv','rows.csv','sides.csv','edge_diagnostics.csv','observation_seal.json','scoring_seal.json')
    write(PUBLIC/'verification.json', dict(frames=len(panel), recordings=len({r['recording'] for r in data}),
          exact_containment_decisions_including_unknown=equal,unknown_decisions=unknown,decision_verified=True,
          official_native_corner_code_sha=digest(UPSTREAM/'eval_det.py'), protocol_sha256=digest(HERE/'PROTOCOL.json'),
          decision_sha256=digest(HERE/'DECISION.json'),result_sha256={p:digest(PUBLIC/p) for p in files},
          implementation={p.name:digest(p) for p in HERE.glob('*.py')},new_qwen_calls=0,GPU_used=False,FINAL_access=False))
    print('Verified',equal,'projection flags, including',unknown,'unavailable comparisons.')


if __name__=='__main__':
    main()
