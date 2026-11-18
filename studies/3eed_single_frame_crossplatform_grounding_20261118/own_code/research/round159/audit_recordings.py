"""Independently join raw annotations, recover sensor coordinates, and check each published outcome."""
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import ijson
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT/'crossplatform/round159'
sys.path[:0] = [str(ROOT/'research/round5/audit'),str(ROOT/'tools')]
from independent_core_v2 import transform_gt, yaw_matrix
from obb_iou import oriented_iou


def readj(path):
    return json.loads(path.read_text())


def read_table(name):
    with (HERE/'results'/name).open(newline='') as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def raw_gt(meta,index,caption,target,suffix):
    with Path(meta).open('rb') as stream:
        item=next(obj for i,obj in enumerate(ijson.items(stream,'ground_info.item',use_float=True)) if i==index)
    def caption_tokens(value):
        return value.replace(',', ' ,').split()

    if caption_tokens(item['caption']) != caption_tokens(caption):
        raise ValueError('Independent expression/object join failed')
    return transform_gt(np.asarray(item['bbox_3d'],dtype=np.float64),target,suffix).astype(np.float64)


def unframe(box,target,suffix):
    box=box.copy().astype(np.float64)
    if target=='drone':box[2]-=1.8
    if target=='waymo':
        a=math.radians((0,-45,45,-90,90)[suffix])
        box[:3]=yaw_matrix(-a)@box[:3]
        box[6]-=a
    return box


def main():
    seal=readj(HERE/'RECORDING_PREDICTION_SEAL.json')
    validation=readj(HERE/'RECORDING_AUDIT.json')
    assert validation['status']=='PASS' and not seal['target_grounding_labels_read']
    manifest=readj(WORK/'RECORDING_INPUTS_PRIVATE.json')
    index={r['id']:r['object_index'] for r in readj(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json')}
    index.update({r['id']:r['object_index'] for r in readj(ROOT/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json')})
    earlier=readj(ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json')
    extra=readj(ROOT/'crossplatform/round158/EXTRA_REGISTRY_PRIVATE.json')
    folder=lambda r:Path(r['metapath']).resolve().parent.parent
    prior_dirs={folder(e['row']) for e in earlier+extra}
    prior_dirs.update(folder(e['pair_PRIVATE']['B']) for e in earlier if e['has_B'])
    used_dirs={folder(r) for r in manifest}
    assert used_dirs.isdisjoint(prior_dirs)
    tasks=read_table('recording_all_rows.csv')
    offset=0;maximum=0.; geometry_calls=0; mode_totals=defaultdict(list); per_scene=defaultdict(list)
    for s in ('waymo','drone','quad'):
        path=WORK/(s+'_PRIVATE.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['sources'][s]['sha256']
        with np.load(path,allow_pickle=False) as f:
            ids,coords,winners=f['rows'],f['boxes'],f['winners']
        for i,three,who in zip(ids,coords,winners):
            r=manifest[int(i)]; target=r['platform']; v=int(r['suffix'] or 0); record=tasks[offset];offset+=1
            if (record['source'],record['target'],int(record['ordinal']),int(record['scene_cluster']),int(record['point_content_cluster']))!=(s,target,offset,r['scene_cluster'],r['point_content_cluster']):
                raise ValueError('Independent row ordering or anonymous cluster mismatch')
            assert int(record['changed_query'])==int(who[0]!=who[1]) and np.array_equal(three[0,:6],three[1,:6])
            gt=unframe(raw_gt(r['metapath'],index[r['id']],r['caption'],target,v),target,v)
            expected=[]
            for col,box in zip(('old_axis','old_pca','r158_pca'),three):
                b=unframe(box,target,v)
                y=oriented_iou(b[:3],b[3:6],yaw_matrix(b[6]),gt[:3],gt[3:6],yaw_matrix(gt[6]))
                observed=float(record[col]); maximum=max(maximum,abs(y-observed));geometry_calls+=1
                assert (y>.25)==(observed>.25) and (y>.5)==(observed>.5)
                mode_totals[s,target,col].append(observed)
                expected.append(observed)
            per_scene[r['scene_cluster']].append(expected[2]-expected[1])
    assert offset==len(tasks)==seal['total'] and maximum<1e-7
    metrics=['mIoU','Acc25','Acc50'];err=0.;pooled=defaultdict(list)
    for row in read_table('recording_directions.csv'):
        k=row['source'],row['target'],row['arm']
        a=mode_totals[k]
        actual=[100*math.fsum(fn(x) for x in a)/len(a) for fn in (float,lambda y:y>.25,lambda y:y>.5)]
        assert int(row['n'])==len(a)
        err=max(err,*(abs(float(row[m])-v) for m,v in zip(metrics,actual)))
        pooled[k[2]].append(actual)
    for row in read_table('recording_overall.csv'):
        for j,m in enumerate(metrics):
            err=max(err,abs(float(row[m])-math.fsum(x[j] for x in pooled[row['arm']])/6))
    assert err<1e-10
    delta=[float(r['r158_pca'])-float(r['old_pca']) for r in tasks]
    a=next(r for r in read_table('recording_changes.csv') if r['source']=='all' and r['after']=='r158_pca')
    assert int(a['improved'])==sum(d>1e-12 for d in delta) and int(a['worsened'])==sum(d< -1e-12 for d in delta)
    result=dict(status='PASS',all_rows=offset,recording_directories_disjoint_from_R158=len(used_dirs),
                fields_read_after_prediction_seal=True,raw_sensor_frame_oriented_iou_checks=geometry_calls,
                maximum_independent_iou_error=maximum,max_percentage_point_error=err,all_thresholds_agree=True,
                improved=int(a['improved']),worsened=int(a['worsened']),tied=int(a['tied']),FINAL_access=False)
    (HERE/'RECORDING_SCALAR_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
