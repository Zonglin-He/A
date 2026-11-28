"""Build a recording-isolated Mono3DRefer/KITTI sensor and language panel."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import pickle
import sys
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO.parent
ASSETS = ROOT / 'assets/archives/mono3drefer'
RUN = ROOT / 'runs/2d-pose-finalization-20261128'
SENSORS = RUN / 'mono-input'
PARQUET = ASSETS / 'test.parquet'
TEST_HASH = '2c6fcc9bf64cad397962e4f5ca675c10b87e13736b5fe7b5168931c66700922e'
DEVKIT_HASH = 'ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1'
PARQUET_URL = ('https://huggingface.co/datasets/JadeRay-42/Mono3DRefer/resolve/main/'
               'data/test-00000-of-00001.parquet')

sys.path.insert(0, str(REPO/'tools/3eed_recovery'))
from run_qwen_3eed import digest, publish, read
from kitti_archives import BASE, index, member, save_member


def fetch_asset(path, url, want):
    if not path.exists():
        staged = path.with_suffix('.partial')
        with urlopen(url, timeout=120) as response, staged.open('wb') as output:
            while part := response.read(4 << 20):
                output.write(part)
        if digest(staged) != want:
            raise ValueError(f'Download hash mismatch for {path.name}')
        staged.replace(path)
    if digest(path) != want:
        raise ValueError(f'Cached hash mismatch for {path.name}')


def index_archive(key):
    path = ASSETS / (key + '.pkl')
    if not path.exists():
        index_items = index(key+'.zip')
        path.write_bytes(pickle.dumps(index_items))
    return pickle.loads(path.read_bytes())


def raw_recordings():
    with ZipFile(ASSETS/'devkit_object.zip') as packed:
        def data(suffix):
            k = next(n for n in packed.namelist() if n.endswith('/'+suffix))
            return packed.read(k).decode('ascii')
        indices = [int(x)-1 for x in data('train_rand.txt').strip().split(',')]
        physical = [x.split() for x in data('train_mapping.txt').splitlines() if x.strip()]
    assert len(indices)==7481 and len(set(indices))==7481 and min(indices)>=0
    return {i: (physical[j][1],int(physical[j][2])) for i,j in enumerate(indices)}


def overlap_verify(object_lidar, tracking_lidar, lookup):
    object_url, objects = object_lidar
    tracking_url, tracking = tracking_lidar
    by_fingerprint = defaultdict(list)
    for name, z in objects.items():
        if name.startswith('training/velodyne/') and name.endswith('.bin'):
            by_fingerprint[z.CRC,z.file_size].append((int(Path(name).stem),z))
    matches = defaultdict(list)
    for name,z in tracking.items():
        if name.startswith('training/velodyne/') and name.endswith('.bin'):
            tracks = Path(name).parent.name
            for code,o in by_fingerprint[z.CRC,z.file_size]:
                matches[tracks].append((code,o,z))
    assert set(matches)=={f'{i:04d}' for i in range(21)}
    # SHA-256 equality of actual member bodies checks the archive index join.
    def inspect(track):
        rows = matches[track]
        raw = {lookup[i][0] for i,_,_ in rows}
        assert len(raw)==1,(track,raw)
        code,a,b = rows[len(rows)//2]
        first = member(object_url,a)
        second = member(tracking_url,b)
        assert first == second
        return dict(tracking_sequence=track,raw_recording=next(iter(raw)),
                    joined_frames=len(rows), verified_sample=code, byte_sha256=sha256(first).hexdigest())
    with ThreadPoolExecutor(10) as threads:
        return list(threads.map(inspect,sorted(matches)))


def calibration(text):
    entries = {k.rstrip(':'):np.fromstring(v,sep=' ') for k,v in
               [ln.split(maxsplit=1) for ln in text.splitlines() if ln.strip()]}
    p = entries['P2'].reshape(3,4)
    vel = np.eye(4)
    vel[:3] = entries['Tr_velo_to_cam'].reshape(3,4)
    rect = np.eye(4)
    rect[:3,:3] = entries['R0_rect'].reshape(3,3)
    rect_from_lidar = rect @ vel
    optical = rect_from_lidar.copy()
    optical[:3,3] += np.linalg.solve(p[:,:3],p[:,3])
    axis = np.eye(4)
    axis[:3,:3] = [[0,-1,0],[0,0,-1],[1,0,0]]
    spec = dict(image_intrinsic=p[:,:3].tolist(), image_extrinsic=(np.linalg.inv(optical)@axis).tolist())
    return spec, rect_from_lidar.tolist(), (p,rect[:3,:3],vel[:3])


def questions(info, image_id):
    entries=[]
    assert set(info['img_id'])=={image_id}
    for slot,(instance,phrases) in enumerate(zip(info['instance_id'],info['description'],strict=True)):
        for sentence in phrases.split('//'):
            text=sentence.strip()
            if not text:
                continue
            query=sha256((f'Mono3DRefer-Test-v1|{image_id}|{instance}|'+text).encode()).hexdigest()
            entries.append(dict(uid=query[:24],text=text,slot=slot,instance=instance))
    return sorted({e['uid']:e for e in entries}.values(),key=lambda e:e['uid'])


def main():
    RUN.mkdir(exist_ok=True,parents=True)
    ASSETS.mkdir(exist_ok=True,parents=True)
    fetch_asset(PARQUET,PARQUET_URL,TEST_HASH)
    fetch_asset(ASSETS/'devkit_object.zip',BASE+'devkit_object.zip',DEVKIT_HASH)
    source = pq.read_table(PARQUET,columns=['info','image','calib']).to_pylist()
    assert len(source)==262
    lookup = raw_recordings()
    with ThreadPoolExecutor(4) as threads:
        rgb, lidar, tracking, calib = threads.map(index_archive,
           ('data_object_image_2','data_object_velodyne','data_tracking_velodyne','data_object_calib'))
    verify = RUN/'tracking-join-private.json'
    if not verify.exists():
        publish(verify,overlap_verify(lidar,tracking,lookup))
    history = read(verify)
    contaminated = {r['raw_recording'] for r in history}
    assert len(history)==21
    clean=[]
    test_ids=set()
    total_descriptions=0
    test_image_index={}
    for row in source:
        image_id = int(row['info']['img_id'][0])
        assert image_id not in test_ids
        test_ids.add(image_id)
        image_key = f'training/image_2/{image_id:06d}.png'
        has_rgb = image_key in rgb[1]
        assert has_rgb and image_id in lookup
        record, _ = lookup[image_id]
        queries = questions(row['info'],image_id)
        total_descriptions += len(queries)
        test_image_index[image_id]=(record,len(queries))
        if record not in contaminated:
            clean.append((image_id,row,record,queries))
    assert clean and total_descriptions>=5000

    def prepare(task):
        image_id,row,record,queries=task
        key=f'{image_id:06d}'
        dst=SENSORS/key
        dst.mkdir(exist_ok=True,parents=True)
        rgb_file=dst/'image.png'
        points=dst/'returns.bin'
        spec_file=dst/'camera.json'
        cam_source=dst/'camera-kitti.txt'
        save_member(rgb[0],rgb[1][f'training/image_2/{key}.png'],rgb_file)
        save_member(lidar[0],lidar[1][f'training/velodyne/{key}.bin'],points)
        save_member(calib[0],calib[1][f'training/calib/{key}.txt'],cam_source)
        spec, rect, expected = calibration(cam_source.read_text())
        for k,x in zip(('P2','R0','Tr_velo2cam'),expected,strict=True):
            np.testing.assert_allclose(row['calib'][k],x,rtol=2e-7,atol=2e-7,err_msg=f'calibration {key} {k}')
        arrow_rgb = row['image']['bytes']
        if rgb_file.read_bytes() != arrow_rgb:
            with Image.open(BytesIO(arrow_rgb)) as a, Image.open(rgb_file) as b:
                assert np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB'))),key
        publish(spec_file,spec)
        return dict(image_id=image_id,recording=record,rect_from_velodyne=rect,queries=queries,
                    rgb=str(rgb_file),points=str(points),calibration=str(spec_file),seed=int(sha256(key.encode()).hexdigest()[:16],16),
                    sensor_sha256={p.name:digest(p) for p in (rgb_file,points,spec_file)})

    result=[]
    with ThreadPoolExecutor(16) as threads:
        for i,row in enumerate(threads.map(prepare,clean),1):
            result.append(row)
            if i%24==0 or i==len(clean):
                print('Verified aligned RGB + LiDAR frames',i,'/',len(clean),flush=True)
    result.sort(key=lambda x:x['image_id'])
    index_path=RUN/'mono-paired-index-private.json'
    publish(index_path,result)
    frames=[]
    for row in result:
        q=row['queries'][0]
        frames.append(dict(uid=q['uid'],target='waymo',suffix=0,text=q['text'],
                    **{k:row[k] for k in ('rgb','points','calibration','seed')}))
    manifest=RUN/'mono-inputs.json'
    publish(manifest,frames)
    sensor_bank=RUN/'mono-sensor-hashes.json'
    publish(sensor_bank,{row[k]:row['sensor_sha256'][Path(row[k]).name]
                        for row in result for k in ('rgb','points','calibration')})
    seal=RUN/'mono-input-seal.json'
    publish(seal,dict(benchmark='Mono3DRefer Test + official KITTI Object Velodyne',parquet=TEST_HASH, devkit=DEVKIT_HASH,
        protocol=digest(HERE/'PROTOCOL_ZH.md'), implementation=digest(Path(__file__)),
        overlap_join=digest(verify),physical_recordings_all=len({x[0] for x in test_image_index.values()}),
        test_frames=len(test_ids),test_unique_descriptions=total_descriptions,
        eligible_frames=len(frames), eligible_descriptions=sum(len(r['queries']) for r in result),
        excluded_by_recording=len(test_ids)-len(frames), direct_recordings_overlap=0,
        eligible_recordings=len({r['recording'] for r in result}),exposed_tracking_sequences=len(history),
        historical_lidar_joins=sum(r['joined_frames'] for r in history),
        manifest=digest(manifest), sensor_bank=digest(sensor_bank), private_index=digest(index_path),target_labels_read=False))
    print(read(seal),flush=True)


if __name__=='__main__':
    main()
