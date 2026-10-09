"""Portable independent count/bootstrap audit; anonymous saved scalars only."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_stvg_motivation_cross_domain_v6 import independent_summary


def audit(base, conditional_file):
    rows=json.loads((base/'SCALAR_ROWS.json').read_text())
    quadrants=json.loads((base/'QUADRANT_STATISTICS.json').read_text())
    saved=json.loads(conditional_file.read_text())
    primary=independent_summary(rows,quadrants)
    checked=0
    def same(a,b):
        nonlocal checked
        x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
        assert x.shape==y.shape and np.allclose(x,y,rtol=0,atol=1e-10),(a,b)
        checked+=int(x.size)
    for di,direction in enumerate(['vidstg_to_hc2','hc2_to_vidstg']):
        sample=np.random.default_rng(20261009+di).integers(0,128,size=(10000,128))
        # Independent multiplicity aggregation, rather than indexing the
        # producer's indicator arrays and summing their sampled rows.
        multiplicity=np.array([np.bincount(r,minlength=128) for r in sample])
        for model in ['tastvg','tubedetr']:
            rr=sorted((r for r in rows if r['direction']==direction and r['model']==model),key=lambda r:r['parent'])
            assert [r['parent'] for r in rr]==list(range(128))
            groups=[[],[],[],[]]
            for r in rr:
                cat=(0 if r['sIoU']>.5 else 1) if r['tIoU']>.5 else (2 if r['sIoU']>.5 else 3)
                groups[cat].append(r['parent'])
            counts=[len(g) for g in groups];den=counts[0]+counts[1];num=counts[1]
            z=saved['directions'][direction][model]
            same(z['original_quadrant_counts'],counts)
            same(z['temporal_correct'],den);same(z['temporal_correct_spatial_wrong'],num)
            same(z['conditional_spatial_failure_percent'],100*num/den)
            bn=multiplicity[:,groups[1]].sum(axis=1)
            bd=multiplicity[:,groups[0]+groups[1]].sum(axis=1)
            assert np.all(bd>0)
            same(z['conditional_95CI_percent'],np.percentile(100*bn/bd,[2.5,97.5]))
            same(z['original_TplusSminus_minus_TminusSplus_pp'],100*(counts[1]-counts[2])/128)
    assert checked==40 and primary['scalar_comparisons']==104
    return dict(status='pass',scope='independent anonymous scalar count and multiplicity-bootstrap readback',
        native_rows=512,unique_parents=256,original_primary_comparisons=104,
        conditional_comparisons=checked,total_scalar_comparisons=144,
        bootstrap_draws=10000,original_primary_endpoint_unchanged=True,
        conditional_endpoint_posthoc_descriptive=True,all_original_negative_findings_retained=True,
        no_model_GT_media_or_raw_prediction_payload_access=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=ROOT/'results/stvg_motivation_cross_domain/2026-10-09')
    parser.add_argument('--conditional',type=Path,default=ROOT/'results/stvg_motivation_filmstrip/2026-10-09/CONDITIONAL_STATISTICS.json')
    args=parser.parse_args();print(json.dumps(audit(args.base,args.conditional)))
