"""Conditional one-plane expansion backed by separate fit and validation returns."""
from collections import Counter
import sys

import numpy as np

from scope164 import HERE, PLATFORMS, WORK, atomic_json, read, sha, source_scope, verify_seal, write_seal
from fit164 import ARMS, outside_cost
from geometry162 import axes, local_points, to_parameters
from run162 import array_file
from score162 import put_table
from score164 import audited_ious, changed, ground_truth, paired_metrics

ARMS_FACE = ('r158','initial','split_outside_s0','one_face')


def weighted_median(values, weights):
    order = np.argsort(values,kind='stable')
    c = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(c,c[-1]/2)]])


def one_plane(points, weights, held, initial, scale):
    """Use supported external returns as a lower bound on enclosure."""
    info = dict(reason='insufficient_fit_extent',side_fit_n=0,side_held_n=0,held_cost_delta=0.,accepted=False)
    if not (~held).any() or not (peak:=weights[~held].max())>0:
        return initial.copy(),info
    local = local_points(points,initial)
    dims = initial[3:6]
    trusted = weights>=.5*peak
    choices = []
    for k in (0,1):
        tangent = np.arange(3)!=k
        near = (np.abs(local[:,tangent])<=dims[tangent]/2).all(1)
        for sign in (-1,1):
            out = sign*local[:,k]-dims[k]/2
            witnesses = trusted & near & (out>.03)
            fit = ~held & witnesses
            if fit.sum()<6:
                continue
            extent = min(weighted_median(out[fit],weights[fit]),.20*dims[k])
            choices.append((extent/dims[k],extent,k,sign,witnesses))
    if not choices:
        return initial.copy(),info
    _,delta,k,sign,witnesses = max(choices,key=lambda x:x[0])
    info.update(side_fit_n=int((~held&witnesses).sum()),side_held_n=int((held&witnesses).sum()))
    if info['side_held_n']<3:
        info['reason']='insufficient_holdout_extent'
        return initial.copy(),info
    if delta<=.03:
        info['reason']='bounded_change_below_detection_limit'
        return initial.copy(),info
    box = initial.copy()
    local_shift=np.zeros(3)
    local_shift[k]=sign*delta/2
    box[:3]+=local_shift@axes(initial[6]).T
    box[3+k]+=delta
    # The opposite face stays fixed. No other direction gains free parameters.
    applied = (box[:3]-initial[:3])@axes(initial[6])
    np.testing.assert_allclose(applied,local_shift,atol=1e-10,rtol=0.)
    np.testing.assert_allclose(-sign*applied[k]+box[3+k]/2,dims[k]/2,atol=1e-10,rtol=0.)
    mass = weights[held]
    old,new = [float(mass@outside_cost(points[held],to_parameters(b),scale)/mass.sum()) for b in (initial,box)]
    if old-new<=1e-12:
        info['reason']='no_heldout_improvement'
        return initial.copy(),info
    info.update(reason='accepted',held_cost_delta=new-old,accepted=True)
    return box,info


def predict():
    assert not (HERE/'FACE_SEAL.json').exists()
    summary = read(HERE/'results/GEOMETRY_SUMMARY.json')
    harm = summary['Acc50_harms']
    assert summary['conditional_face_gate'] and all(harm[f'outside_s{j}']<harm[f'surface_s{j}'] for j in (0,1))
    files = verify_seal('GEOMETRY',WORK/'geometry')
    assert sha(HERE/'GEOMETRY_SEAL.json')==read(HERE/'GEOMETRY_AUDIT.json')['geometry_seal_sha256']
    out = WORK/'face'
    out.mkdir(parents=True,exist_ok=True)
    for name in files:
        with np.load(WORK/'geometry'/name,allow_pickle=False) as p:
            old,initial=p['full'][:2]
            outside=p['split'][ARMS.index('outside_s0')]
            result,info=one_plane(p['points'],p['weights'][0],p['held'],initial,float(p['scale']))
            if not bool(p['has_winner']):
                info['reason']='no_initial_winner'
            assert changed(result,initial)==info['accepted']
        array_file(out/name,boxes=np.stack((old,initial,outside,result)),**info)
    write_seal('FACE',out,range(192),parent_geometry_audit=sha(HERE/'GEOMETRY_AUDIT.json'),
               passed_predeclared_stage_gate=True, no_label_in_per_case_decision=True)
    print('Sealed one-face experiment',len(files),flush=True)


def score():
    verify_seal('FACE',WORK/'face')
    verify_seal('GEOMETRY',WORK/'geometry')
    output=HERE/'results'
    rows,metrics,contrasts=[],[],[]
    greatest=0.
    for i,r in enumerate(source_scope()):
        name=f'{i:05d}.npz'
        with np.load(WORK/'face'/name,allow_pickle=False) as p, np.load(WORK/'geometry'/name,allow_pickle=False) as g:
            boxes=p['boxes']
            np.testing.assert_array_equal(boxes[:2],g['full'][:2])
            np.testing.assert_array_equal(boxes[2],g['split'][ARMS.index('outside_s0')])
            x,e=audited_ious(boxes,r,ground_truth(r))
            greatest=max(greatest,e)
            rows.append(dict(row=i+1,source=r['source'],**{k:p[k].item() for k in
                ('reason','side_fit_n','side_held_n','held_cost_delta','accepted')},**{a:float(b) for a,b in zip(ARMS_FACE,x)}))
    for key in (*PLATFORMS,'all'):
        group=[r for r in rows if key in (r['source'],'all')]
        values={a:np.array([r[a] for r in group]) for a in ARMS_FACE}
        for arm,a in values.items():
            metrics.append(dict(source=key,arm=arm,n=len(a),mIoU=100*a.mean(),Acc25=100*(a>.25).mean(),Acc50=100*(a>.50).mean(),
                delivered_face_update=sum(r['accepted'] for r in group) if arm=='one_face' else 0))
        for base in ARMS_FACE[:-1]:
            contrasts.append(dict(source=key,arm='one_face',control=base,n=len(group),
                **paired_metrics(values[base],values['one_face'])))
    put_table(output/'face_rows.csv',rows)
    put_table(output/'face_metrics.csv',metrics)
    put_table(output/'face_contrasts.csv',contrasts)
    atomic_json(HERE/'FACE_AUDIT.json',dict(rows=len(rows),endpoint_checks=len(rows)*len(ARMS_FACE),
                max_abs_iou_error=greatest,face_seal_sha256=sha(HERE/'FACE_SEAL.json'),
                reasons=dict(Counter(r['reason'] for r in rows)),scalars_sha256=sha(output/'face_rows.csv'),FINAL_access=False))
    print('One-face experiment scored',Counter(r['reason'] for r in rows),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['score']:
        score()
    else:
        predict()
