"""Find and evaluate previously unused Mono3DRefer recordings after the source gate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from io import BytesIO
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from compact_pool import HERE, ROOT, digest, read, write
from prepare_blind import prepare as observe
from source import BATCH, MODEL, SOURCE, runtime
from transfer import TARGET, batches, infer, paired_summary, training

sys.path.insert(0,str(HERE.parent/'3eed_2d_pose_finalization_20261128'))
import prepare_mono as archive
from kitti_archives import save_member

OUT = TARGET/'mono'
HISTORICAL = ROOT/'runs/2d-pose-finalization-20261128'
MONO_REV = '5bd11c9bd3e0174f8aced071b8661129f92a4ca9'
MONO_FILES = {
    'val': '1dab7d29f3236611937423db7f6e8905629bb07ca2404c58a3e1a075c1de9c5f',
    'train': 'cea5d17e9ed752de0b5ada4231439f590010296b712a4c5c7816db830a232875',
}


def sensors(task):
    image_id,row,record,lidar,calib,rgb,split = task
    key = f'{image_id:06d}'
    path = OUT/'sensors'/key
    path.mkdir(parents=True,exist_ok=True)
    image, points, original, camera = (path/n for n in ('image.png','returns.bin','calib.txt','camera.json'))
    for package,name,suffix,p in ((rgb,'image_2','png',image),(lidar,'velodyne','bin',points),
                                  (calib,'calib','txt',original)):
        url,index = package
        save_member(url,index[f'training/{name}/{key}.{suffix}'],p)
    spec, transform, (proj,rect,vel) = archive.calibration(original.read_text())
    for field,expected in zip(('P2','R0','Tr_velo2cam'),(proj,rect,vel),strict=True):
        np.testing.assert_allclose(row['calib'][field],expected,rtol=2.e-7,atol=2.e-7)
    arrow = row['image']['bytes']
    if image.read_bytes() != arrow:
        with Image.open(BytesIO(arrow)) as a, Image.open(image) as b:
            np.testing.assert_array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB')))
    write(camera,spec)
    query = archive.questions(row['info'],image_id)[0]
    uid = f'mono-{split}-'+query['uid']
    # The registered Mono adapter uses the front-camera geometry entry with
    # the official KITTI extrinsic encoded in the exported CameraSpec.
    frame = dict(uid=uid,text=query['text'],target='waymo',suffix=0,waymo_view='F',seed=int(query['uid'][:16],16),
                 rgb=str(image),points=str(points),calibration=str(camera))
    blobs = {str(p):digest(p) for p in (image,points,camera)}
    return (frame,dict(uid=uid,image_id=image_id,recording=record,slot=query['slot'],
                      instance=query['instance'],transform=transform),blobs)


def prepare():
    training()
    OUT.mkdir(parents=True,exist_ok=True)
    assert digest(archive.PARQUET) == archive.TEST_HASH
    assert digest(archive.ASSETS/'devkit_object.zip') == archive.DEVKIT_HASH
    id_to_record = archive.raw_recordings()
    test_ids = {int(r['info']['img_id'][0]) for r in pq.read_table(archive.PARQUET,columns=['info']).to_pylist()}
    test_records = {id_to_record[i][0] for i in test_ids}
    old = HISTORICAL/'mono-input-seal.json'
    union = HISTORICAL/'tracking-join-private.json'
    assert digest(union) == read(old)['overlap_join']
    tracking = {r['raw_recording'] for r in read(union)}
    blocked = test_records | tracking
    examined = {}
    for split,expect in MONO_FILES.items():
        url = f'https://huggingface.co/datasets/JadeRay-42/Mono3DRefer/resolve/{MONO_REV}/data/{split}-00000-of-00001.parquet'
        parquet = archive.ASSETS/f'{split}.parquet'
        archive.fetch_asset(parquet,url,expect)
        targets = pq.read_table(parquet,columns=['info']).to_pylist()
        examined[split] = len(targets)
        choice = []
        for row in targets:
            image_id = int(row['info']['img_id'][0])
            record,_ = id_to_record[image_id]
            if record not in blocked and image_id not in test_ids and archive.questions(row['info'],image_id):
                choice.append((image_id,record))
        if choice:
            break
    assert len({v[0] for v in choice}) == len(choice)

    frames,labels,sensor_hashes = [],[],{}
    if choice:
        by_id = {int(r['info']['img_id'][0]):r for r in
                 pq.read_table(parquet,columns=['info','image','calib']).to_pylist()}
        with ThreadPoolExecutor(12) as workers:
            lidar,calib,rgb = workers.map(archive.index_archive,('data_object_velodyne','data_object_calib','data_object_image_2'))
            for f,y,h in workers.map(sensors,((i,by_id[i],s,lidar,calib,rgb,split) for i,s in choice)):
                frames.append(f)
                labels.append(y)
                sensor_hashes.update(h)
    write(OUT/'inputs-private.json',frames)
    write(OUT/'label-join-private.json',labels)
    write(OUT/'sensor-hashes-private.json',sensor_hashes)
    seal = dict(outcome='eligible' if frames else 'no_new_recording', benchmark='Mono3DRefer',
                split=split, examined=examined, n=len(frames), recordings=len({r['recording'] for r in labels}),
                excluded_prior_recordings=len(blocked), prior_mono_recordings=len(test_records),
                prior_tracking_recordings=len(tracking), labels_read=False, revision=MONO_REV,
                parquet=expect, prior=digest(old), training=digest(SOURCE/'training-seal.json'),
                inputs=digest(OUT/'inputs-private.json'), joins=digest(OUT/'label-join-private.json'),
                sensor_hashes=digest(OUT/'sensor-hashes-private.json'),code=digest(__file__))
    write(OUT/'input-seal.json',seal)
    print('Mono novelty audit',seal,flush=True)


def predict():
    training()
    def audit(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes,Path)):
            import os
            p = Path(os.fsdecode(args[0]))
            if p.name in ('val.parquet','test.parquet','train.parquet','label-join-private.json','mono-paired-index-private.json',
                          'data_object_label_2.zip'):
                raise PermissionError(f'Frozen external run denied {p.name}')
    sys.addaudithook(audit)
    runtime()
    from run_full_qwen import batch_read, load_model
    import minimal

    seal = read(OUT/'input-seal.json')
    assert seal['training'] == digest(SOURCE/'training-seal.json') and seal['code'] == digest(__file__)
    assert seal['inputs'] == digest(OUT/'inputs-private.json')
    assert seal['sensor_hashes'] == digest(OUT/'sensor-hashes-private.json')
    assert all(digest(p)==s for p,s in read(OUT/'sensor-hashes-private.json').items())
    frames = read(OUT/'inputs-private.json')
    if not frames:
        return
    torch.set_num_threads(1)
    processor,model,limit = load_model(MODEL)
    data = []
    with torch.inference_mode():
        for group in batches(frames,BATCH):
            reads = {k:batch_read(processor,model,limit,group,k,Counter()) for k in ('2d','3d')}
            for row,raw2,raw3 in zip(group,reads['2d'],reads['3d'],strict=True):
                raw = {'2d':raw2,'3d':raw3}
                base = minimal.predict(row,raw)
                task = dict(frame=row,raw=raw,target='mono')
                data.append((observe((row['uid'],task,dict(outputs=dict(axis_2d=base)))),task,None))
    del model
    torch.cuda.empty_cache()
    boxes = infer(batches(data))
    path = OUT/'predictions-private.json'
    write(path,boxes)
    write(OUT/'prediction-seal.json',dict(payload=digest(path),n=len(boxes),inputs=digest(OUT/'input-seal.json'),
          train=digest(SOURCE/'training-seal.json'), transfer=digest(HERE/'transfer.py'),labels_read=False))


def score():
    training()
    seal = read(OUT/'input-seal.json')
    if seal['outcome'] == 'no_new_recording':
        return
    stamp = read(OUT/'prediction-seal.json')
    predictions = OUT/'predictions-private.json'
    assert stamp['payload'] == digest(predictions) and not stamp['labels_read']
    assert stamp['train']==digest(SOURCE/'training-seal.json') and stamp['transfer']==digest(HERE/'transfer.py')
    parquet = archive.ASSETS/(seal['split']+'.parquet')
    assert stamp['inputs'] == digest(OUT/'input-seal.json') and seal['parquet'] == digest(parquet)
    runtime()
    from shared import setup
    setup()
    from score import canonical_box

    joins = OUT/'label-join-private.json'
    assert seal['joins'] == digest(joins)
    refs = {r['uid']:r for r in read(joins)}
    target = {int(r['info']['img_id'][0]):r for r in pq.read_table(parquet,columns=['info','objects']).to_pylist()}
    with urlopen(archive.BASE+'data_object_label_2.zip',timeout=60) as response:
        payload = response.read()
    actual = {}
    with ZipFile(BytesIO(payload)) as official:
        for uid,item in refs.items():
            raw = target[item['image_id']]
            slot = item['slot']
            assert raw['info']['instance_id'][slot] == item['instance']
            gt = raw['objects']
            h,w,l = (float(gt[n][slot]) for n in ('h','w','l'))
            pos,yaw = np.asarray(gt['pos'][slot],float),float(gt['ry'][slot])
            filename = f"training/label_2/{item['image_id']:06d}.txt"
            official_boxes = [np.array(v.split()[8:15],float) for v in official.read(filename).decode('ascii').splitlines()
                              if len(v.split())>=15 and v.split()[0]!='DontCare']
            assert any(np.allclose(b,np.r_[h,w,l,pos,yaw],atol=1.e-4,rtol=0) for b in official_boxes)
            inv = np.linalg.inv(item['transform'])
            center = inv[:3,:3]@(pos-np.array([0,h/2,0]))+inv[:3,3]
            direction = inv[:3,:3]@np.array([np.cos(yaw),0.,-np.sin(yaw)])
            actual[uid] = dict(box=canonical_box(np.r_[center,l,w,h,np.arctan2(direction[1],direction[0])]))
    summary, _ = paired_summary(read(predictions),actual,refs)
    write(OUT/'results.json',dict(panel=summary,predictions=stamp,inputs=seal,labels_after_seal=True))
    print('New Mono panel',summary['overall'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('prepare','predict','score'))
    dict(prepare=prepare,predict=predict,score=score)[parser.parse_args().phase]()
