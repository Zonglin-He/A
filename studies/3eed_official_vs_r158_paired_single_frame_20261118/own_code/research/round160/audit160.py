"""Independently read raw annotations, validate box geometry, paired counts and six-direction results."""
import csv
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import ijson
import numpy as np
import torch
from common160 import ROOT, HERE, WORK, REC, EXTRA, DEV, ORDER, rows, sha, read, canonical, json_new

sys.path[:0]=[str(ROOT/'research/round5/audit'),str(ROOT/'tools'),str(ROOT/'crossplatform/round3/p0/upstream')]
from independent_core_v2 import transform_gt, yaw_matrix
from obb_iou import oriented_iou
from utils.eval_det import iou3d_rotated_vs_aligned

ARMS=('official_span','native_text_only','prepared_axis','prepared_pca','r158_full')
METRICS=('mIoU','Acc25','Acc50')


def table(name):
    with (HERE/'results'/name).open(newline='') as f:return list(csv.DictReader(f))


@lru_cache(maxsize=8192)
def raw_annotations(path):
    # Read the original sensor-coordinate box and verify the expression's annotation index.
    with Path(path).open('rb') as f:
        objects=ijson.items(f,'ground_info.item',use_float=True)
        return [(o['caption'],np.asarray(o['bbox_3d'],np.float64)) for o in objects]


def sensor_frame(box,target,suffix):
    b=box.copy()
    if target=='drone':b[2]-=1.8
    if target=='waymo':
        a=math.radians((0,-45,45,-90,90)[suffix])
        b[:3]=yaw_matrix(-a)@b[:3];b[6]-=a
    return b


def published_geometry(box,gt):
    # Use the authors' float32 polyhedron metric. Rotate the common frame if the prediction has yaw.
    bg=gt.astype(np.float64).copy(); pred=box[:6].copy()
    if box[6]!=0.:
        bg[:3]=yaw_matrix(-box[6])@(bg[:3]-box[:3]); bg[6]-=box[6];pred[:3]=0
    result,_=iou3d_rotated_vs_aligned(torch.tensor(bg[None],dtype=torch.float32),
                                     torch.tensor(pred[None],dtype=torch.float32))
    return float(result[0,0])


def scalar(vals,weights):
    n=sum(weights)
    return [100*sum(w*f(x) for x,w in zip(vals,weights))/n
            for f in (lambda x:x,lambda x:x>.25,lambda x:x>.50)]


def main():
    from eval_lock import verify
    verify()
    torch.set_num_threads(1)
    seal=read(HERE/'PREDICTION_SEAL.json')
    assert read(HERE/'EVALUATION.json')['prediction_seal']==sha(HERE/'PREDICTION_SEAL.json')
    dest=WORK/'BOXES_PRIVATE.npz';assert sha(dest)==seal['sha256']
    with np.load(dest,allow_pickle=False) as data:
        boxes=data['boxes'].astype(np.float64);assert tuple(data['arms'])==ARMS
    registry=rows();rr=table('all_records.csv')
    assert len(rr)==len(registry)==len(boxes)==seal['n']
    lookup={r['id']:r['object_index'] for r in read(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json')}
    lookup.update({r['id']:r['object_index'] for r in read(ROOT/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json')})
    geom_max=0.;native_max=0.;native_disagreements=defaultdict(int);generic_calls=0
    grouped=defaultdict(list);recordings=defaultdict(set);native_values=defaultdict(list)
    for i,(p,e,bb) in enumerate(zip(rr,registry,boxes)):
        row=e['r'];t=e['target'];c=e['cohort'];k=e['ordinal'];s=e['source'];view=int(row['suffix'] or 0)
        assert (c,k,s,t)==(p['cohort'],int(p['ordinal']),p['source'],p['target'])
        assert float(p['sampling_weight'])==e['sampling_weight']
        pos=int(lookup[row['id']] if c==REC else row['object_index'])
        label,raw=raw_annotations(row['metapath'])[pos];assert canonical(label)==canonical(row['caption'])
        g=transform_gt(raw,t,view).astype(np.float64);sensor_gt=sensor_frame(g,t,view)
        for j,arm in enumerate(ARMS):
            b=bb[j];bs=sensor_frame(b,t,view)
            v=float(oriented_iou(bs[:3],bs[3:6],yaw_matrix(bs[6]),sensor_gt[:3],sensor_gt[3:6],yaw_matrix(sensor_gt[6])))
            actual=float(p[arm]);geom_max=max(geom_max,abs(v-actual));generic_calls+=1
            assert (v>.25,v>.5)==(actual>.25,actual>.5)
            if arm in ('official_span','r158_full'):
                literal=published_geometry(b,g)
                if not math.isfinite(literal):raise ValueError('Nonfinite literal geometry')
                native_max=max(native_max,abs(literal-actual))
                native_values[c,s,t,arm].append((literal,e['sampling_weight']))
                for th in (.25,.5):native_disagreements[arm,th]+=int((actual>th)!=(literal>th))
        grouped[c,s,t].append(p);recordings[c,t].add(Path(row['metapath']).resolve().parent.parent)
        if (i+1)%2500==0:print('audit',i+1,flush=True)
    if geom_max>1e-7:raise ValueError('Different sensor-frame IoU')
    expected={};errors=[]
    for d in table('directions.csv'):
        group=grouped[d['cohort'],d['source'],d['target']];a=d['arm']
        assert len(group)==int(d['n'])
        z=scalar([float(r[a]) for r in group],[float(r['sampling_weight']) for r in group])
        expected[d['cohort'],d['source'],d['target'],a]=z
        errors.extend(abs(x-float(d[m])) for m,x in zip(METRICS,z))
    for d in table('overall.csv'):
        z=np.array([expected[d['cohort'],s,t,d['arm']] for s,t in ORDER]).mean(0)
        errors.extend(abs(x-float(d[m])) for m,x in zip(METRICS,z))
    for p in table('paired.csv'):
        c,s,t,a,b=(p[k] for k in ('cohort','source','target','before','after'))
        directions=ORDER if s=='six_equal' else [(s,t)]
        vals=[];up=np.zeros(3,int);down=np.zeros(3,int)
        for s,t in directions:
            group=grouped[c,s,t]
            va=np.array([float(r[a]) for r in group]);vb=np.array([float(r[b]) for r in group])
            diff=np.stack([vb-va,(vb>.25).astype(int)-(va>.25),(vb>.5).astype(int)-(va>.5)])
            vals.append(np.subtract(expected[c,s,t,b],expected[c,s,t,a]));up+=(diff>1e-12).sum(1);down+=(diff< -1e-12).sum(1)
        for j,m in enumerate(METRICS):
            errors.append(abs(float(p[m])-np.mean(vals,axis=0)[j]))
            assert (int(p[m+'_up']),int(p[m+'_down']))==(up[j],down[j])
    assert max(errors)<1e-9
    native_rows=[]
    for (c,s,t,a),items in native_values.items():
        v,w=zip(*items);z=scalar(v,w)
        native_rows.append(dict(cohort=c,source=s,target=t,arm=a,n=len(items),**dict(zip(METRICS,z))))
    with (HERE/'results/native_geometry_check.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(native_rows[0]));w.writeheader();w.writerows(native_rows)
    for t in ('waymo','drone','quad'):
        assert recordings[REC,t].isdisjoint(recordings[DEV,t]|recordings[EXTRA,t])
    json_new(HERE/'AUDIT.json',dict(status='PASS',paired_rows=len(rr),all_arm_sensor_frame_IoU_checks=generic_calls,
       maximum_independent_geometry_abs_error=geom_max,original_float32_geometry_calls=2*len(rr),
       original_float32_max_abs_error=native_max,original_float32_threshold_differences={
           a:{str(th):native_disagreements[a,th] for th in (.25,.50)} for a in ('official_span','r158_full')},
       weighted_scalar_checks=len(errors),maximum_scalar_error_pp=max(errors),negative_pairs_and_threshold_counts_verified=True,
       recording_count={c:{t:len(recordings[c,t]) for t in ('waymo','drone','quad')} for c in (EXTRA,DEV,REC)},
       target_blind_selection=False,FINAL_access=False))
    print('audit passed',len(rr),max(errors),dict(native_disagreements),flush=True)


if __name__=='__main__':main()
