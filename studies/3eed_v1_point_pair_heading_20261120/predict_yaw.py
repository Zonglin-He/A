"""Reproduce the locked single-frame winner and seal yaw-only predictions for all cohorts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from orientation import ARMS, estimate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAMERA=HERE.parent/'3eed_v1_camera_word_replacement_20261120'
PRIVATE=HERE/'run_PRIVATE'
RESULTS=HERE/'results'
sys.path[:0]=[str(CAMERA),str(ROOT),str(ROOT/'crossplatform/round7'),
              str(ROOT/'crossplatform/round7/experiment'),str(ROOT/'crossplatform/round3/p0/upstream')]
from predict import inputs, ordinary_baseline, archive_npz
from calibrate import ORD, digest, read, save
from runner_helpers import sample_points
from utils.transform_waymo import transform_to_front_view
from camera_projection import WAYMO_ANGLES, z_rotation
from crossplatform.candidate_frontend import WAYMO_VIEWS
from scope183 import check_input

STAGES=('source','pilot','ordinary')
SIZES=dict(source=384,pilot=3072,ordinary=18620)


def roster_digest(rows):
    return hashlib.sha256('\n'.join(f"{r['pointpath']}|{r['metapath']}|{r['caption']}|{r.get('seed',0)}"
                                    for r in rows).encode()).hexdigest()


def sample_native(row):
    target=row['platform']
    view=int(row['suffix'] or 0)
    points,_,_=sample_points(dict(platform=target,pointpath=row['pointpath'],seed=int(row.get('seed',0))))
    if target=='waymo':
        points,_=transform_to_front_view(points,np.empty((0,7)),WAYMO_VIEWS[view])
        points=np.ascontiguousarray(points,dtype=np.float32)
    assert points.shape==(16384,3) and points.dtype==np.float32
    return points


def measure_row(row, base):
    native=base.copy()
    if row['platform']=='drone':
        native[2]+=1.8
    elif row['platform']=='waymo':
        angle=WAYMO_ANGLES[int(row['suffix'] or 0)]
        native[:3]=base[:3]@z_rotation(angle).T
        native[6]=(base[6]+np.deg2rad(angle)+np.pi/2)%np.pi-np.pi/2
    angles,counts,error=estimate(sample_native(row),native[:6],native[6])
    sensor=np.full(len(angles),base[6])
    changed=angles!=native[6]
    sensor[changed]=(base[6]+angles[changed]-native[6]+np.pi/2)%np.pi-np.pi/2
    assert sensor[0]==base[6]
    return sensor,counts,error


def direction_job(job):
    stage,direction,rows,lock=job
    dest=PRIVATE/stage
    path=dest/(direction+'.npz')
    receipt=path.with_suffix('.json')
    if receipt.exists():
        item=read(receipt)
        assert item['lock']==lock and item['sha256']==digest(path) and item['n']==len(rows)
        return direction,item
    source=CAMERA/'run_PRIVATE'/stage
    previous={} if stage=='ordinary' else read(source/'SEALED.json')['files']
    bases, yaws, counts, errors=[],[],[],[]
    for first in range(0,len(rows),64):
        block=rows[first:first+64]
        if stage=='ordinary':
            all_boxes,picks=ordinary_baseline(direction,first,len(block))
        else:
            rel=f'{direction}/{first:05d}.npz'
            assert digest(source/rel)==previous[rel]['sha256']
            with np.load(source/rel,allow_pickle=False) as old:
                all_boxes,picks=old['boxes'],old['choices'][:,0]
        for j,row in enumerate(block):
            if stage=='ordinary':
                check_input(row)
            base=all_boxes[j,picks[j]].astype(np.float64,copy=True)
            angles,nums,err=measure_row(row,base)
            bases.append(base)
            yaws.append(angles)
            counts.append(nums)
            errors.append(err)
    archive_npz(path,base=np.stack(bases),yaws=np.stack(yaws),counts=np.array(counts,dtype=np.int32))
    out=dict(lock=lock,sha256=digest(path),n=len(rows),eligible=sum(int(c[0]) for c in counts),
             native_yaw_max_error=max(errors))
    save(receipt,out)
    print(stage,direction,len(rows),'eligible',out['eligible'],flush=True)
    return direction,out


def run(workers):
    cfg=read(HERE/'PROTOCOL.json')
    assert tuple(cfg['arms'])==ARMS
    assert digest(ROOT/'crossplatform/single_frame_sources.json')==cfg['source_assets']
    assert read(RESULTS/'geometry_audit.json')['orientation_sha256']==digest(HERE/'orientation.py')
    assert digest(ORD/'ordinary_PRIVATE/SEALED.json')==cfg['ordinary_roster']
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    for stage in STAGES:
        start=perf_counter()
        parent=(ORD/'predictions_PRIVATE' if stage=='ordinary' else CAMERA/'run_PRIVATE'/stage)
        assert digest(parent/'SEALED.json')==cfg['baseline_seals'][stage]
        rows=inputs(stage)
        rosters={f'{s}_to_{t}':roster_digest(rs) for (s,t),rs in rows.items()}
        if stage!='ordinary':
            old=read(parent/'SEALED.json')
            assert old['lock']==digest(parent/'LOCK.json')
            assert rosters==read(parent/'LOCK.json')['roster']
        else:
            assert read(parent/'SEALED.json')['roster_sha256']==cfg['ordinary_roster']
        dest=PRIVATE/stage
        dest.mkdir(exist_ok=True)
        lock=dict(stage=stage,rosters=rosters,baseline=cfg['baseline_seals'][stage],arms=list(ARMS),
                  code={f:digest(HERE/f) for f in ('PROTOCOL.json','orientation.py','predict_yaw.py','check_orientation.py')})
        if (dest/'LOCK.json').exists():
            assert lock==read(dest/'LOCK.json')
        else:
            save(dest/'LOCK.json',lock)
        lock_hash=digest(dest/'LOCK.json')
        jobs=[(stage,f'{s}_to_{t}',rs,lock_hash) for (s,t),rs in rows.items()]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            files=dict(pool.map(direction_job,jobs))
        assert sum(x['n'] for x in files.values())==SIZES[stage]
        seal=dict(n=SIZES[stage],lock=lock_hash,files=files,task_labels_opened=False,FINAL_access=False)
        save(dest/'SEALED.json',seal)
        print('sealed',stage,round(perf_counter()-start,2),'s',flush=True)
    save(PRIVATE/'SEALED_ALL.json',{stage:digest(PRIVATE/stage/'SEALED.json') for stage in STAGES})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args().workers)
