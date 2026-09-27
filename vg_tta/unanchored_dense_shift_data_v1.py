"""Immutable HC/Vid evaluation roster, raw decode and offline metric bridge.

Preparation may read annotation-bearing parent records to extract metadata and
Frozen predictions. The runtime query whitelist contains no GT. Scoring uses
the parent's saved normalized labels, only after predictions have completed.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/baseline_expansion_v1'
FFMPEG=ROOT/'.conda/tubedetr/bin/ffmpeg'
FFMPEG_SHA='bfec0e82c8497b0b979b1a7812ba1101da3646e8a3330bcb5537badb840a527e'
GROUPS={'hc_to_vid':('hc','vid'),'vid_to_hc':('vid','hc')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sampled_ids(meta,kind):
    if kind=='vidstg':
        from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _sample_frame_ids
        return _sample_frame_ids(meta)
    if kind!='hcstvg':
        raise ValueError(kind)
    rate=5.0/(int(meta['frame_count'])/20.)
    if not 0<rate<=1:
        raise ValueError('HC native loader requires downsampling')
    ids=[0]
    for index in range(int(meta['frame_count'])-1):
        if int(ids[-1]*rate)<int(index*rate):
            ids.append(index)
    if len(ids)>200:
        ids=[ids[(j*len(ids))//200] for j in range(200)]
    return ids


def prepare_group(group):
    source,target=GROUPS[group]
    lp=BASE/'lock.json';lock=read(lp);spec=lock['datasets'][target]
    assert lock['cells'][group+'_clean']=={'source':source,'target':target,
            'condition':'clean','shift':'compound_cross_dataset'}
    annotation=Path(spec['annotation'])
    assert sha(annotation)==spec['annotation_sha256']
    contents=read(annotation)
    rows=contents['videos'] if target=='vid' else contents
    checkpoint=Path(lock['checkpoints'][source])
    assert sha(checkpoint)==lock['checkpoint_sha256'][source]
    pins={str(lp):sha(lp),str(annotation):sha(annotation),str(checkpoint):sha(checkpoint),
          str(FFMPEG):sha(FFMPEG),str(Path(__file__)):sha(__file__),
          str(ROOT/'vg_tta/metrics.py'):sha(ROOT/'vg_tta/metrics.py'),
          str(ROOT/'scripts/evaluate_fullspan_scale_shift_corruptions_v1.py'):
          sha(ROOT/'scripts/evaluate_fullspan_scale_shift_corruptions_v1.py')}
    assert pins[str(FFMPEG)]==FFMPEG_SHA
    queries=[]
    for index in spec['indices']:
        row=rows[index]
        meta={k:row[k] for k in ('caption','video_id','frame_count','width','height','fps',
                               'start_frame','end_frame','original_video_id') if k in row}
        path=Path(row['video_path'])
        if not path.is_absolute():
            path=Path(spec['root'])/'video'/path
        assert str(path) in spec['video_sha256']
        assert sha(path)==spec['video_sha256'][str(path)]
        parent=BASE/(group+'_clean')/'episodes'/f'{index:06d}.json'
        ep=read(parent)
        assert ep['index']==index and ep['source']==spec['sources'][str(index)]
        frozen=[r for r in ep['records'] if r['method']=='frozen']
        assert len(frozen)==1
        replay=frozen[0]['metric_replay']
        ids=sampled_ids(meta,spec['kind'])
        assert ids==replay['frame_ids']
        # Only these fields cross the prepared runtime query boundary.
        query={'index':index,'source':ep['source'],'kind':spec['kind'],**meta,
               'video_path':str(path),'video_sha256':sha(path),'frame_ids':ids,
               'parent_episode':str(parent),'parent_episode_sha256':sha(parent),
               'raw_pixel_sha256':ep['raw_pixel_sha256'],
               'reference_frozen':{k:replay[k] for k in ('pred_boxes','pred_sted')}}
        queries.append(query);pins[str(parent)]=sha(parent);pins[str(path)]=sha(path)
    return {'group':group,'source':source,'target':target,'checkpoint':str(checkpoint),
            'checkpoint_sha256':sha(checkpoint),'queries':queries,'pins':pins,
            'metric_name':'vIoU_corrected','source_count':len({q['source'] for q in queries}),
            'historical_untouched':False,'prepare_annotation_qa':True,'fit_GT_used':False}


def raw_digest(raw):
    # Rich successor receipt. The historical baseline used bytes only.
    h=hashlib.sha256();h.update(str(raw.dtype).encode());h.update(str(raw.shape).encode())
    h.update(np.ascontiguousarray(raw).tobytes());return h.hexdigest()


def decode_raw(query):
    assert sha(query['video_path'])==query['video_sha256']
    assert sha(FFMPEG)==FFMPEG_SHA
    if query['kind']=='vidstg':
        from scripts.evaluate_fullspan_scale_shift_corruptions_v1 import _decode_raw_metadata_only
        raw,ids=_decode_raw_metadata_only(query,Path(query['video_path']))
    else:
        import ffmpeg
        ids=sampled_ids(query,'hcstvg')
        data,_=(ffmpeg.input(query['video_path'],ss=0,t=20).filter('fps',fps=len(ids)/20.)
                .output('pipe:',format='rawvideo',pix_fmt='rgb24')
                .run(capture_stdout=True,quiet=True,cmd=str(FFMPEG)))
        raw=np.frombuffer(data,np.uint8).reshape(-1,query['height'],query['width'],3).copy()
    assert ids==query['frame_ids'] and len(raw)==len(ids)
    assert hashlib.sha256(np.ascontiguousarray(raw)).hexdigest()==query['raw_pixel_sha256'],'Native pixel receipt mismatch'
    return raw,ids


def score(query,prediction):
    from vg_tta.metrics import compute_stvg_metrics,interval_from_logits
    assert sha(query['parent_episode'])==query['parent_episode_sha256']
    ep=read(query['parent_episode'])
    replay=next(r['metric_replay'] for r in ep['records'] if r['method']=='frozen')
    assert replay['frame_ids']==query['frame_ids']
    targets=[{'boxes':torch.tensor(b,dtype=torch.float32).reshape(-1,4)} for b in replay['target_boxes']]
    values=compute_stvg_metrics(prediction['pred_boxes'],targets,interval_from_logits(prediction['pred_sted']),
                tuple(replay['gt_indices']),frame_ids=query['frame_ids'],
                gt_frame_interval=tuple(replay['gt_frame_interval']))
    return {'index':query['index'],'source':query['source'],'metric':values['vIoU_corrected'],'metrics':values}
