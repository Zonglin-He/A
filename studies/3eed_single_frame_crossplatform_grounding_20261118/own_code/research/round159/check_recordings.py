"""Select by sensor and recording metadata, then freeze raw single-frame predictions."""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT/'crossplatform/round159'
sys.path.insert(0,str(ROOT))
from crossplatform.ground_single_frame import SingleFrameGrounder, WAYMO_VIEWS, file_sha256, orient_box

LOCK = HERE/'experiment/RECORDING_CHECK.json'
INPUT = WORK/'RECORDING_INPUTS_PRIVATE.json'


def read(path):
    return json.loads(path.read_text())


def write_new(path, data):
    with path.open('x') as f:
        json.dump(data,f,indent=2,allow_nan=False)
        f.write('\n')


def check_lock():
    spec=read(LOCK)
    for name,pin in {**spec['source_weights_and_inference_code_sha256'],**spec['input_files_sha256']}.items():
        if file_sha256(ROOT/name)!=pin:
            raise ValueError('Registered file changed: '+name)


def select():
    WORK.mkdir(exist_ok=True)
    check_lock()
    old=read(ROOT/'research/round145/INPUT_REGISTRY_PRIVATE.json')
    extra=read(ROOT/'crossplatform/round158/EXTRA_REGISTRY_PRIVATE.json')
    used={(e['target'],e['row']['scene']) for e in old+extra}
    used.update((e['target'],e['pair_PRIVATE']['B']['scene_PRIVATE']) for e in old if e.get('has_B'))
    groups=read(ROOT/'research/round131/CONTENT_GROUPS_PRIVATE.json')
    all_frames=[f for g in groups for f in g['frames']]
    val_paths={str(Path(f[k]).resolve()) for f in all_frames for k in ('imagepath','pointpath')}
    val_points={f['point_sha256'] for f in all_frames if f['point_sha256']}
    val_images={f['image_sha256'] for f in all_frames if f['image_sha256']}
    all_rows=(read(ROOT/'research/round69/input/TRAIN1024_PRIVATE.json')+
              read(ROOT/'research/round76/input/SOURCE_TRAIN512_PRIVATE.json'))
    group_index={};scene_index={};items=[];skip=Counter()
    hash_sensor=lru_cache(None)(file_sha256)
    for r in all_rows:
        sk=(r['platform'],r['scene'])
        if sk in used:
            skip['recording_in_R158']+=1;continue
        if any(str(Path(r[k]).resolve()) in val_paths for k in ('pointpath','imagepath')):
            skip['registered_VAL_path']+=1;continue
        ph,ih=(hash_sensor(r[k]) for k in ('pointpath','imagepath'))
        if ph in val_points or ih in val_images:
            skip['shared_VAL_sensor_content']+=1;continue
        fields={k:r[k] for k in ('id','platform','pointpath','imagepath','metapath','caption','suffix','scene')}
        fields.update(scene_cluster=scene_index.setdefault(sk,len(scene_index)+1),
                      point_content_cluster=group_index.setdefault(ph,len(group_index)+1),
                      point_sha256=ph,image_sha256=ih)
        items.append(fields)
    assert len(items)+sum(skip.values())==len(all_rows) and len({r['id'] for r in items})==len(items)
    write_new(INPUT,items)
    info=dict(lock_sha256=file_sha256(LOCK),input_sha256=file_sha256(INPUT),expressions=len(items),
              eligible_by_platform=dict(Counter(r['platform'] for r in items)),excluded=dict(skip),
              point_contents=len(group_index),recording_ids=len(scene_index),
              target_labels_used_for_selection=False,FINAL_access=False)
    write_new(HERE/'RECORDING_INPUT_SEAL.json',info)
    print('selected',info['eligible_by_platform'],info['excluded'],flush=True)


def run():
    import gc
    import torch
    check_lock()
    seal=read(HERE/'RECORDING_INPUT_SEAL.json')
    assert file_sha256(INPUT)==seal['input_sha256'] and file_sha256(LOCK)==seal['lock_sha256']
    rows=read(INPUT)
    screens={}
    tic=time.monotonic()
    for source in ('waymo','drone','quad'):
        ids=[i for i,row in enumerate(rows) if row['platform']!=source]
        final=WORK/(source+'_PRIVATE.npz')
        receipt=WORK/(source+'_SEALED.json')
        if receipt.exists():
            s=read(receipt)
            assert s['lock_sha256']==file_sha256(LOCK) and s['input_sha256']==seal['input_sha256'] and s['sha256']==file_sha256(final)
            screens[source]=s
            continue
        model=SingleFrameGrounder(source)
        boxes=np.empty((len(ids),3,7),np.float32)
        winners=np.empty((len(ids),2),np.int16)
        # Per-row receipts allow a recoverable interruption without replacing any sealed output.
        scratch=WORK/'partial'/source
        scratch.mkdir(parents=True,exist_ok=True)
        for n,i in enumerate(ids):
            r=rows[i]; f=scratch/f'{i}.npz'
            if f.exists():
                with np.load(f,allow_pickle=False) as saved:
                    assert str(saved['lock'])==seal['lock_sha256'] and str(saved['manifest'])==seal['input_sha256']
                    boxes[n],winners[n]=saved['boxes'],saved['winners']
            else:
                if file_sha256(r['pointpath'])!=r['point_sha256'] or file_sha256(r['imagepath'])!=r['image_sha256']:
                    raise ValueError('Input sensor bytes changed')
                result=model.score_frame(target=r['platform'],points=r['pointpath'],rgb=r['imagepath'],calibration=r['metapath'],
                                         text=r['caption'],waymo_view=WAYMO_VIEWS[int(r['suffix'] or 0)],seed=0)
                old,new=int(result.probability.argmax()),result.winner();winners[n]=old,new
                old_pca=orient_box(result.sampled_points,result.boxes[old])
                new_pca=old_pca if old==new else orient_box(result.sampled_points,result.boxes[new])
                boxes[n]=np.stack([np.r_[old_pca[:6],np.float32(0)],old_pca,new_pca])
                if not np.isfinite(boxes[n]).all():raise ValueError('Invalid prediction')
                tmp=f.with_suffix('.tmp')
                with tmp.open('wb') as stream:
                    np.savez(stream,lock=seal['lock_sha256'],manifest=seal['input_sha256'],boxes=boxes[n],winners=winners[n])
                os.replace(tmp,f)
            if (n+1)%128==0:print('predict',source,n+1,len(ids),round(time.monotonic()-tic),flush=True)
        tmp=final.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,rows=ids,boxes=boxes,winners=winners)
        os.replace(tmp,final)
        screens[source]=dict(n=len(ids),sha256=file_sha256(final),input_sha256=seal['input_sha256'],lock_sha256=seal['lock_sha256'])
        write_new(receipt,screens[source])
        del model
        gc.collect();torch.cuda.empty_cache()
    write_new(HERE/'RECORDING_PREDICTION_SEAL.json',dict(sources=screens,directions=6,
              total=sum(x['n'] for x in screens.values()),lock_sha256=seal['lock_sha256'],
              target_grounding_labels_read=False,FINAL_access=False))
    print('sealed predictions',flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['select']:select()
    elif sys.argv[1:]==['predict']:run()
    else:raise SystemExit('use select or predict')
