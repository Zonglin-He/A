"""Outcome-only ceiling for arbitrary translation along a frozen box's width axis."""
import csv
from collections import Counter
import numpy as np
from scipy.optimize import minimize_scalar
from common import BASE, HERE, OUT, PRIVATE, SPEC, digest, inputs, read, write
from score import canonical_box, many_iou, score_boxes, truth_for_slot
from batch_yaw_iou import corners

def strip_vertices(vertices, half_length):
    pts=[p for p in vertices if abs(p[0])<=half_length]
    for p,q in zip(vertices,np.roll(vertices,-1,axis=0)):
        if p[0]==q[0]:
            continue
        for x in (-half_length,half_length):
            t=(x-p[0])/(q[0]-p[0])
            if 0<t<1:
                pts.append(p+t*(q-p))
    return np.asarray(pts)

def maximize(box, gt):
    baseline=many_iou(box[None],gt)[0]
    c,s=np.cos(box[6]),np.sin(box[6])
    normal=np.array([-s,c])
    local=(corners(gt[None])[0]-box[:2])@np.array([[c,-s],[s,c]])
    poly=strip_vertices(local,box[3]/2)
    if len(poly)<3 or abs(box[2]-gt[2])>=(box[5]+gt[5])/2:
        return float(baseline),0.,0.
    def values(offsets):
        rows=np.repeat(box[None],len(offsets),axis=0)
        rows[:,:2]+=np.asarray(offsets)[:,None]*normal
        return many_iou(rows,gt)
    knots=np.unique(np.r_[poly[:,1]-box[4]/2,poly[:,1]+box[4]/2])
    left,right=knots[:-1],knots[1:]
    mid=(left+right)/2
    nodes=np.r_[knots,mid]
    iou=values(nodes)
    # Recover a quantity proportional to intersection volume; it is quadratic
    # between consecutive vertex crossings. IoU itself is a rational function.
    v=iou/(1+iou)
    v0,v1,vm=v[:len(knots)-1],v[1:len(knots)],v[len(knots):]
    quadratic=2*(v1+v0-2*vm)
    linear=v1-v0-quadratic
    live=quadratic < -1.e-15
    vertex=-linear[live]/(2*quadratic[live])
    extra=(left[live]+(right-left)[live]*vertex)[(vertex>0)&(vertex<1)]
    offsets=np.r_[0.,nodes,extra]
    vals=values(offsets)
    j=int(np.argmax(vals))
    lower,upper=knots[0],knots[-1]
    alternative=minimize_scalar(lambda t:-values([t])[0],bounds=(lower,upper),method='bounded',
                                options={'xatol':1.e-9})
    error=abs(vals[j]-max(baseline,-alternative.fun))
    checked=box.copy()
    checked[:2]+=normal*offsets[j]
    independent,gap=score_boxes(checked[None],gt)
    return float(vals[j]),float(offsets[j]),max(error,float(gap),abs(float(independent[0])-vals[j]))

def run():
    panel, frozen=inputs()
    with (BASE/'results/rows.csv').open(newline='') as f:
        history={r['uid']:float(r['H']) for r in csv.DictReader(f)}
    data,counts={},Counter()
    max_gap=parity=0.
    for row in panel:
        uid=row['uid']
        entry=frozen[uid]
        val=shift=base=0.
        if entry['boxes']['H'] is not None:
            box=canonical_box(entry['boxes']['H'])
            gt=canonical_box(truth_for_slot(row,read(row['calibration'])['ground_info']))
            base=float(many_iou(box[None],gt)[0])
            val,shift,gap=maximize(box,gt)
            max_gap=max(max_gap,gap)
        parity=max(parity,abs(base-history[uid]))
        assert val+1.e-10>=base
        opportunity=bool(base<=.5 and val>.5)
        counts['opportunities_all']+=opportunity
        counts['opportunities_selected']+=opportunity and entry['selected']
        counts['null_boxes']+=entry['boxes']['H'] is None
        data[uid]=dict(baseline=base,oracle=val,delta=shift,recoverable50=opportunity)
    assert max_gap<1.e-6 and parity<1.e-8, (max_gap,parity)
    write(PRIVATE/'oracle.json',data)
    receipt=dict(**counts,frames=len(panel),maximum_solver_gap=max_gap,baseline_parity=parity,
                 code=digest(__file__),protocol=digest(SPEC),private_sha=digest(PRIVATE/'oracle.json'),
                 diagnostics_only=True,FINAL_access=False)
    write(OUT/'oracle_seal.json',receipt)
    print(receipt,flush=True)

if __name__=='__main__':
    run()
