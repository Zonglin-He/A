"""Fix a source-only training cohort before starting GPU work."""
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from bridge import CFG, DATA, HERE, RUN, UPSTREAM, digest, publish, read, setup, source_guard


def stable(key):
    from hashlib import sha256
    return sha256(f'posttrain:{CFG["seed"]}:{key}'.encode()).hexdigest()


def choose(groups, count, key, used):
    groups = {r:sorted(rows, key=lambda f:stable(key+f.parent.name+'/'+f.name)) for r,rows in groups.items() if rows}
    order = sorted(groups, key=lambda r:stable(key+r))
    members = []
    for index in range(max(map(len, groups.values()))):
        for rec in order:
            if index >= len(groups[rec]):
                continue
            path = groups[rec][index]
            sha = digest(path/'image.jpg')
            if sha not in used:
                used.add(sha)
                members.append((path,sha))
            if len(members) == count:
                return members
    raise ValueError(f'Only {len(members)} eligible inputs for {key}')


def camera_raw(box, mat):
    c = np.asarray(box[:3])@mat[:3,:3].T+mat[:3,3]
    rot = mat[:3,:3] @ Rotation.from_euler('z',box[6]).as_matrix()
    return np.r_[c,box[3:6],Rotation.from_matrix(rot).as_euler('xyz')/np.pi]


def row_for(item):
    from geometry import camera
    from score import canonical_box
    from crossplatform.round7.camera_projection import read_camera_allowlist

    folder, rgb_sha = item
    metadata = folder/'meta_info.json'
    cal = read_camera_allowlist(metadata)
    camera_path = RUN/'waymo-camera'/f'{stable(str(folder))[:24]}.json'
    if not camera_path.exists():
        publish(camera_path,{k:np.asarray(v, dtype=float).tolist() for k,v in cal.items() if v is not None})
    with Image.open(folder/'image.jpg') as image:
        scale = np.tile(image.size,2)
    annotations = read(metadata)['ground_info']
    for slot in sorted(range(len(annotations)), key=lambda k:stable(str(folder)+str(k))):
        ann=annotations[slot]
        box=np.asarray(ann['bbox_3d'],float)
        window=np.clip(np.asarray(ann.get('bbox_2d_proj'),float)/scale*1000,0,1000)
        if box.shape!=(7,) or not np.isfinite(box).all() or min(box[3:6])<=0:
            continue
        if window.shape!=(4,) or not np.isfinite(window).all() or min(window[2:]-window[:2])<4:
            continue
        uid=stable(f'{folder.parent.name}/{folder.name}/{slot}')[:24]
        frame=dict(uid=uid,target='waymo',text=ann['caption'],suffix=int(folder.name.split('_')[-1]),
                   rgb=str(folder/'image.jpg'),points=str(folder/'lidar.npy'),calibration=str(camera_path),seed=int(uid[:16],16))
        gt = canonical_box(box)
        cam = camera_raw(gt,camera(frame)[2])
        if cam[2] <= .2:
            continue
        return dict(uid=uid,frame=frame,rgb_sha=rgb_sha,recording=folder.parent.name,
                    box=gt.tolist(),window=np.round(window).astype(int).tolist(),camera_box=cam.round(2).tolist(),
                    label=ann['class'],source_slot=slot)
    return None


def main():
    source_guard()
    setup()
    import geometry
    import score
    RUN.mkdir(parents=True,exist_ok=True)
    (RUN/'waymo-camera').mkdir(exist_ok=True)
    used=set()
    split_dir=UPSTREAM/'data/splits'
    splits={key:split_dir/f'waymo_{key}.txt' for key in ('train','val')}
    source={key:set(p.read_text().split()) for key,p in splits.items()}
    assert not source['train'] & source['val']
    result={}
    for key,requested in [('train',CFG['train_queries']),('val',CFG['validation_queries'])]:
        groups={rec:[p.parent for p in (DATA/'waymo'/rec).glob('*/meta_info.json')] for rec in source[key]}
        subset=choose(groups,requested+40,key,used)
        with ThreadPoolExecutor(max_workers=8) as threads:
            jobs=[v for v in threads.map(row_for,subset) if v is not None][:requested]
        assert len(jobs)==requested
        path=RUN/f'source-{key}.json'
        publish(path,jobs)
        result[key]=dict(file=digest(path),n=len(jobs),recordings=len({v['recording'] for v in jobs}),split=digest(splits[key]))
        print(key,result[key],flush=True)
    assert not {v['recording'] for v in read(RUN/'source-train.json')} & {v['recording'] for v in read(RUN/'source-val.json')}
    assert len({v['rgb_sha'] for p in ('train','val') for v in read(RUN/f'source-{p}.json')})==sum(x['n'] for x in result.values())
    publish(RUN/'source-seal.json',dict(protocol=digest(HERE/'protocol.json'),script=digest(Path(__file__)),target_read=False,**result))


if __name__=='__main__':
    main()
