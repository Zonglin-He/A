"""Audit same-class competitors using complete KITTI frame annotations offline."""
from contextlib import closing
from hashlib import sha256
from io import BytesIO
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np

from source_geometry import HERE,PRIOR,ROOT,RUN,digest,read,write
from evaluate_relative import separation
from score_paradigms import ious


def parents(path,seal,key):
    assert digest(path)==seal[key]
    return read(path)


def summarized(rows,stat_panel):
    if not rows:
        return dict(n=0)
    score=stat_panel(rows,('single','competitive'),'single')
    v=score['metrics']['competitive']
    return dict(n=len(rows),old=score['metrics']['single']['miou']/100,competitive=v['miou']/100,
                paired_delta=v['delta_miou']/100,paired_ci=[c/100 for c in v['miou_ci']],
                recordings=score['recordings'])


def main():
    from shared import setup
    setup()
    from evaluate_structure import stat_panel
    from metric_image_common import EXTERNAL,inputs
    from truth_score_inputs import truth_data
    from kitti_data import small_zip
    from kitti_score import labels_for,upright
    from score import canonical_box

    out=RUN/'target'
    predict=read(out/'prediction-seal.json')
    assert not predict['labels_read']
    previous=read(RUN/'results.json')
    assert digest(out/'scores-private.json')==previous['scores_sha']
    assert digest(out/'boxes-private.json')==predict['payload']
    assert digest(out/'visual-seal.json')==predict['visual']
    pins=read(out/'visual-seal.json')
    view=parents(out/'visual-private.json',pins,'visual_sha')
    regions=parents(out/'regions-private.json',pins,'regions_sha')
    cases={r['uid']:r for r in parents(PRIOR/'inputs-private.json',read(PRIOR/'input-seal.json'),'evidence')}
    private={r['uid']:r for r in read(out/'scores-private.json')}
    eligible={r['uid'] for r in read(out/'boxes-private.json') if r['compact'] and r['uid'] in private}

    kitti=EXTERNAL/'members.json'
    by_kitti={r['frame']['uid']:r for r in read(kitti)}
    use=[u for u in eligible if u in by_kitti]
    gt,_=truth_data({u:r for u,r in inputs().items() if u in use})
    output=[]
    with small_zip('data_tracking_label_2.zip') as archive:
        banks={}
        for u in use:
            item=by_kitti[u]
            drive=item['drive']
            if drive not in banks:
                banks[drive]=labels_for(archive,drive)
            target=gt[u]['box']
            ortho=[upright(obj,item['rect_from_lidar']) for obj in banks[drive][item['index']]]
            negative=[b for b,x in zip(ortho,ious(ortho,target),strict=True) if x<.50]
            result=separation(cases[u],regions[u],view[u],target,negative)
            if result and 'same_class' in result:
                output.append(dict(uid=u,target='kitti',recording=private[u]['recording'],
                                   iou=dict(zip(('single','competitive'),result['same_class'],strict=True))))

    mono=ROOT/'runs/2d-pose-finalization-20261128'
    sys.path.insert(0,str(HERE.parent/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel
    target_bank,reference_sha=labels_for_panel(read(mono/'mono-input-seal.json'))
    meta={r['queries'][0]['uid']:r for r in read(mono/'mono-paired-index-private.json')}
    url='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
    with closing(urlopen(url,timeout=30)) as response:
        raw=response.read()
    assert sha256(raw).hexdigest()==reference_sha==read(mono/'mono-results.json')['official_kitti_label_sha256']
    with ZipFile(BytesIO(raw)) as archive:
        for u in sorted(eligible & target_bank.keys()):
            target=target_bank[u]
            ref=target['box']
            expected=np.r_[target['Camera_hwl'],target['Camera_center'],target['Camera_yaw']]
            path=f"training/label_2/{target['image_id']:06d}.txt"
            candidates=[line.split() for line in archive.read(path).decode('ascii').splitlines()]
            data=[(v[0],np.asarray(v[8:15],float)) for v in candidates if len(v)>=15 and v[0]!='DontCare']
            kind=next(category for category,b in data if np.allclose(b,expected,atol=1.e-4,rtol=0))
            negative=[]
            inv=np.linalg.inv(meta[u]['rect_from_velodyne'])
            for category,b in data:
                if category!=kind or np.allclose(b,expected,atol=1.e-4,rtol=0) or min(b[:3])<=0:
                    continue
                h,w,l=b[:3]
                center=inv[:3,:3]@(b[3:6]-[0,h/2,0])+inv[:3,3]
                axis=inv[:3,:3]@np.array([np.cos(b[6]),0.,-np.sin(b[6])])
                negative.append(canonical_box(np.r_[center,l,w,h,np.arctan2(axis[1],axis[0])]))
            negative=[b for b,x in zip(negative,ious(negative,ref),strict=True) if x<.50]
            result=separation(cases[u],regions[u],view[u],ref,negative)
            if result and 'same_class' in result:
                output.append(dict(uid=u,target='mono',recording=private[u]['recording'],
                                   iou=dict(zip(('single','competitive'),result['same_class'],strict=True))))
    for uid,row in private.items():
        x=row.get('separation') or {}
        if 'same_class' in x:
            output.append(dict(uid=uid,target=row['target'],recording=row['recording'],
                               iou=dict(zip(('single','competitive'),x['same_class'],strict=True))))
    details={k:summarized([r for r in output if r['target']==k],stat_panel) for k in ('waymo','drone','quad','kitti','mono')}
    summary=dict(total=summarized(output,stat_panel),platforms=details,official_label_hash=reference_sha,
                 evaluated_with_prior_pred_seal=digest(out/'prediction-seal.json'),labels_used_for_diagnostic_only=True)
    write(out/'same-class-audit.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()
