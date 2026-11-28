"""Run and score the frozen paired external test in separate processes."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO.parent
RUN = ROOT/'runs/2d-pose-finalization-20261128'
MODELDIR = ROOT/'assets/models/Qwen3-VL-4B-Instruct/ebb281ec70b05090aa6165b016eac8ec08e71b17'
UPSTREAM = ROOT/'assets/upstream-3eed'
META = REPO/'QWEN_CURRENT_METHOD.json'
ARMS = ('native','old','promoted')

sys.path.insert(0,str(REPO/'tools/3eed_recovery'))
from run_qwen_3eed import check_geometry, digest, input_frames, publish, read, source_paths


def checked():
    reg=read(META)
    lock=reg['method_lock']
    entry=reg['prediction_entry']
    assert all(digest(REPO/r['file'])==r['sha256'] for r in (lock,entry))
    dependencies=read(REPO/lock['file'])['config']['historical_geometry']
    assert all(digest(REPO/f)==a for f,a in dependencies.items())
    seal_path=RUN/'mono-input-seal.json'
    inputs=read(seal_path)
    assert digest(RUN/'mono-inputs.json')==inputs['manifest'] and not inputs['target_labels_read']
    assert digest(HERE/'PROTOCOL_ZH.md')==inputs['protocol']
    sensors=RUN/'mono-sensor-hashes.json'
    assert digest(sensors)==inputs['sensor_bank']
    assert all(digest(Path(path))==checksum for path,checksum in read(sensors).items())
    source_paths(UPSTREAM)
    check_geometry()
    return reg,inputs


def block_annotations(event, args):
    if event=='open' and isinstance(args[0],(str,bytes,Path)):
        filename=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).name
        if filename in ('test.parquet','mono-paired-index-private.json','mono-scores-private.json',
                         'tracking-join-private.json','data_object_label_2.zip'):
            raise PermissionError(f'Annotation or evaluator key unavailable to frozen inference: {filename}')


def predict():
    import torch
    import transformers

    sys.addaudithook(block_annotations)
    reg, inputs=checked()
    from run_original_schedule import load_model, predict as decode
    from live_2d import geometry, paired

    frames=list(input_frames(RUN/'mono-inputs.json').values())
    assert len(frames)==inputs['eligible_frames']
    for frame in frames:
        frame['rgb_sha']=digest(Path(frame['rgb']))
    plan=dict(input_seal=digest(RUN/'mono-input-seal.json'), registered_method=digest(META),
              method=reg['method_lock']['sha256'], entry=reg['prediction_entry']['sha256'],
              script=digest(Path(__file__)), model_revision=reg['model']['revision'], descriptions=len(frames),
              batch_max_tasks=4, new_model_calls_per_comparison=0, checkpoint_training=False)
    out=RUN/'mono-raw-predictions-private.json'
    if out.exists():
        state=read(out)
        assert state['plan']==plan
    else:
        state=dict(plan=plan, batches=[], predictions=[])
    start=len(state['predictions'])
    assert start==0 or start==len(frames) or start%2==0
    assert [r['uid'] for r in state['predictions']]==[f['uid'] for f in frames[:start]]
    q=perf_counter()
    processor,model,limit=load_model(MODELDIR)
    model_load=perf_counter()-q
    with torch.inference_mode():
        for i in range(start,len(frames),2):
            subset=frames[i:i+2]
            answer_started=perf_counter()
            flat=decode(processor,model,limit,[(f,t) for f in subset for t in ('3d','2d')])
            torch.cuda.synchronize()
            elapsed=perf_counter()-answer_started
            state['batches'].append(dict(queries=len(subset),seconds=elapsed))
            for j,frame in enumerate(subset):
                replies=dict(zip(('3d','2d'),flat[2*j:2*j+2],strict=True))
                old,promoted=paired(frame,replies,diagnostics=True)
                qraw=replies['3d']['value']
                native=geometry.to_sensor(np.asarray(qraw,float),geometry.camera(frame)[2])[0].tolist() if qraw else None
                state['predictions'].append(dict(uid=frame['uid'], replies=replies,
                    outputs=dict(native=native,old=old,promoted=promoted)))
            publish(out,state)
            if i%40==0 or i+2>=len(frames):
                print('External',len(state['predictions']),'/',len(frames),'recent batch s',round(elapsed,2),flush=True)
    publish(RUN/'mono-prediction-seal.json',dict(**plan, payload=digest(out), target_labels_read=False, no_target_tuning=True,
        native_and_geometry_share_answers=True, original_image_requests=len(frames)*2,
        gpu=torch.cuda.get_device_name(), peak_bytes=torch.cuda.max_memory_allocated(),
        pytorch=torch.__version__,transformers=transformers.__version__,model_load_seconds=model_load,
        total_inference_seconds=sum(v['seconds'] for v in state['batches'])))


def labels_for_panel(seal):
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile

    import pyarrow.parquet as pq
    from prepare_mono import PARQUET
    from score import canonical_box

    assert digest(RUN/'mono-paired-index-private.json')==seal['private_index'] and digest(PARQUET)==seal['parquet']
    bank=pq.read_table(PARQUET,columns=['info','objects']).to_pylist()
    matched={int(x['info']['img_id'][0]):x for x in bank}
    labels={}
    for x in read(RUN/'mono-paired-index-private.json'):
        query=x['queries'][0]
        i=query['slot']
        meta=matched[x['image_id']]
        assert meta['info']['instance_id'][i]==query['instance']
        assert query['text'] in [s.strip() for s in meta['info']['description'][i].split('//')]
        gt=meta['objects']
        h,w,l=map(float,(gt['h'][i],gt['w'][i],gt['l'][i]))
        pos=np.asarray(gt['pos'][i],float)
        yaw=float(gt['ry'][i])
        assert min(h,w,l)>0 and np.isfinite(pos).all() and np.isfinite(yaw)
        inv=np.linalg.inv(x['rect_from_velodyne'])
        center=inv[:3,:3]@(pos-np.array([0,h/2,0]))+inv[:3,3]
        axis=inv[:3,:3]@np.array([np.cos(yaw),0,-np.sin(yaw)])
        labels[query['uid']]=dict(recording=x['recording'], box=canonical_box(
            np.r_[center,l,w,h,np.arctan2(axis[1],axis[0])]),category=str(meta['info']['category'][i]),
            image_id=x['image_id'], Camera_hwl=[h,w,l], Camera_center=pos.tolist(), Camera_yaw=yaw)
    # Cross-check target camera boxes against the official KITTI Object labels
    # only after opening the sealed predictions in the offline process.
    url='https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip'
    with urlopen(url,timeout=30) as response:
        archive_bytes=response.read()
    from hashlib import sha256
    with ZipFile(BytesIO(archive_bytes)) as release:
        for item in labels.values():
            filename=f"training/label_2/{item['image_id']:06d}.txt"
            expected=np.array([*item['Camera_hwl'],*item['Camera_center'],item['Camera_yaw']])
            official=[np.array(v.split()[8:15],float) for v in release.read(filename).decode('ascii').splitlines()
                      if len(v.split())>=15 and v.split()[0]!='DontCare']
            assert any(np.allclose(b,expected,atol=1e-4,rtol=0) for b in official),item['image_id']
    return labels,sha256(archive_bytes).hexdigest()


def score():
    reg, inputs=checked()
    from score import score_boxes
    sys.path.insert(0,str(HERE.parent/'3eed_hypothesis_verification_20261128'))
    from inference_stats import cluster_weights, intervals

    predictions=RUN/'mono-raw-predictions-private.json'
    stamp=read(RUN/'mono-prediction-seal.json')
    assert stamp['payload']==digest(predictions) and stamp['input_seal']==digest(RUN/'mono-input-seal.json')
    assert not stamp['target_labels_read'] and not stamp['checkpoint_training'] and stamp['registered_method']==digest(META)
    assert stamp['entry']==reg['prediction_entry']['sha256']
    out=read(predictions)['predictions']
    labels,official_label_hash=labels_for_panel(inputs)
    assert len(labels)==len(out)==inputs['eligible_frames'] and set(labels)=={r['uid'] for r in out}

    checks=[]
    discrepancy=0.
    for row in out:
        truth=labels[row['uid']]
        p=row['outputs']
        boxes = dict(native=p['native'],old=p['old']['box'],promoted=p['promoted']['box'])
        full={k:b for k,b in boxes.items() if b is not None}
        iou=dict.fromkeys(ARMS,0.)
        if full:
            values,err=score_boxes(np.array(list(full.values())),truth['box'])
            iou.update({k:float(v) for k,v in zip(full,values,strict=True)})
            discrepancy=max(discrepancy,err)
        extra=p['promoted'].get('candidate_details',{})
        checks.append(dict(uid=row['uid'],target='kitti',recording=truth['recording'],iou=iou,
                           route=p['old']['readout'],category=truth['category'],
                           switched=extra.get('switched',False),seconds=extra.get('extra_geometry_seconds'),
                           valid_2d=row['replies']['2d']['value'] is not None,
                           valid_3d=row['replies']['3d']['value'] is not None))

    def panel(values):
        if not values:
            return dict(n=0)
        w=cluster_weights(values)
        data={k:np.array([r['iou'][k] for r in values]) for k in ARMS}
        acc={k:x>.50 for k,x in data.items()}
        metrics={k:dict(miou=100*float(x.mean()),acc50=100*float(acc[k].mean())) for k,x in data.items()}
        pairs={}
        for new,old in (('old','native'),('promoted','old'),('promoted','native')):
            m=100*(data[new]-data[old])
            a=100*(acc[new].astype(float)-acc[old])
            pairs[new+'_vs_'+old]=dict(delta_miou=float(m.mean()),miou_95ci=intervals(m,w),
                 delta_acc50=float(a.mean()),acc50_95ci=intervals(a,w),rescue=int((a==100).sum()),harm=int((a==-100).sum()))
        return dict(n=len(values),recordings=len({r['recording'] for r in values}),metrics=metrics,
                    paired=pairs,switched=sum(r['switched'] for r in values), routes=dict(Counter(r['route'] for r in values)),
                    valid_2d=sum(r['valid_2d'] for r in values),valid_3d=sum(r['valid_3d'] for r in values))

    path=RUN/'mono-scores-private.json'
    publish(path,checks)
    whole=panel(checks)
    active=[r for r in checks if r['route']=='C_ray']
    performance=dict(all=whole, C_ray=panel(active),
        by_class={k:panel([r for r in checks if r['category']==k]) for k in sorted({r['category'] for r in checks})},
        extra_geometry_median_ms=1000*float(np.median([r['seconds'] for r in active])) if active else None,
        max_iou_independent_error=discrepancy)
    publish(RUN/'mono-results.json',dict(input=inputs, prediction=stamp, score_payload=digest(path),
        only_offline_gt=True,official_kitti_label_sha256=official_label_hash,
        mono_boxes_directly_matched_to_kitti=len(labels),missing_or_invalid_label_count=0,result=performance))
    print(whole,flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','score'))
    args=p.parse_args()
    globals()[args.stage]()
