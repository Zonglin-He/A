"""One full-prompt replay per original source row. Predict every arm before opening labels."""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from seed_code import ARMS, FIELDS, label_frame_boxes, mask_in_label_frame, native_codes, readout

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAY = HERE.parent / '3eed_v1_language_displacement_rays_20261120'
sys.path[:0] = [str(ROOT), str(RAY)]
import capture as prior
from scope183 import atomic, digest, read
from features import observation
from train import prepare_device
from crossplatform.candidate_frontend import CandidateFrontend, WAYMO_VIEWS

OUT = HERE/'run_PRIVATE/predictions'
PROTOCOL = read(HERE/'PROTOCOL.json')
PINS = PROTOCOL['pins']
NATIVE = ROOT/'crossplatform/round3/p0/upstream'
PARENT = RAY/'run_PRIVATE/source'


def pinned():
    sources = {'registered_model':ROOT/'crossplatform/CURRENT_METHOD.json',
               'registered_sources':ROOT/'crossplatform/single_frame_sources.json',
               'native_loss':NATIVE/'models/losses.py', 'native_membership':NATIVE/'utils/box_util.py',
               'native_loader':NATIVE/'src/joint_det_dataset.py',
               'parent_manifest':PARENT/'SEALED.json','parent_lock':PARENT/'LOCK.json'}
    assert all(digest(p)==PINS[k] for k,p in sources.items())
    old = read(PARENT/'SEALED.json')
    assert old['n']==192 and not old['annotations_used'] and old['lock']==PINS['parent_lock']
    return old,read(PARENT/'LOCK.json')['roster']


@torch.inference_mode()
def infer(model,row,previous,platform,ordinal):
    from model51 import SENTINEL
    from runner_helpers import sample_points
    from utils.transform_waymo import transform_to_front_view
    frame = observation(row)
    sampled,raw_ids,_ = sample_points(dict(platform=platform,seed=frame['seed'],pointpath=frame['points']))
    pre_view = sampled
    if platform=='waymo':
        sampled,_=transform_to_front_view(sampled,np.empty((0,7)),frame['waymo_view'])
    model_cloud = np.ascontiguousarray(sampled,dtype=np.float32)
    start=perf_counter()
    output = model(dict(point_clouds=torch.as_tensor(model_cloud,device='cuda')[None],
                        text=[frame['text'].strip()+SENTINEL]))
    def cpu(t):return t[0].detach().cpu().numpy()
    logits = cpu(output['seeds_obj_cls_logits'])[0]
    idx = cpu(output['seed_inds']).astype(np.int64)
    xyz = cpu(output['seed_xyz'])
    query_ids = cpu(output['query_points_sample_inds']).astype(np.int64)
    stages = output['sa_ins']
    composed = cpu(stages[1]).astype(np.int64)[cpu(stages[2]).astype(np.int64)]
    np.testing.assert_array_equal(model_cloud[composed],xyz)
    assert logits.shape==(1024,) and len(idx)==1024 and np.isfinite(logits).all()
    assert logits.dtype==np.float32 and xyz.shape==(1024,3)
    np.testing.assert_array_equal(previous['ids'],query_ids)
    np.testing.assert_array_equal(previous['anchors'],xyz[query_ids])
    np.testing.assert_array_equal(previous['seed'],logits[query_ids])
    np.testing.assert_array_equal(previous['full'][-1,:,:3],cpu(output['last_center']))
    np.testing.assert_array_equal(previous['full'][-1,:,3:],cpu(output['last_pred_size']))
    boxes = previous['boxes']
    joint = previous['joint']
    view = int(previous['view'])
    assert view==WAYMO_VIEWS.index(frame['waymo_view'])
    slate=previous['slate']
    np.testing.assert_array_equal(boxes[:,:3],cpu(output['last_center'])[slate])
    assert list(joint.shape)==[16] and boxes.shape==(16,7)
    single_seconds = perf_counter()-start
    start=perf_counter()
    labelboxes=label_frame_boxes(boxes,platform,view)
    # Official membership gathers the original input's labels through fp2_inds.
    inside=mask_in_label_frame(pre_view[idx],labelboxes)
    physical_inside=mask_in_label_frame(pre_view[composed],labelboxes)
    codes=native_codes(xyz,boxes,inside,'cuda')
    physical_codes=native_codes(xyz,boxes,physical_inside,'cuda')
    birth=query_ids[slate]
    value=readout(codes,inside,logits,birth,joint,previous['choices'][0],platform,ordinal)
    extra_seconds=perf_counter()-start
    np.testing.assert_array_equal(value['choices'][:3],previous['choices'][:3])
    assert value['fields'].shape==(len(FIELDS),16) and len(value['choices'])==len(ARMS)
    return dict(**value,boxes=boxes,joint=joint,slate=slate,birth=birth,logits=logits,xyz=xyz,
                input_index=idx,composed=composed,raw_ids=raw_ids,pre_view=pre_view,
                codes=codes,inside=inside,view=view,
                index_xyz_diff=(xyz!=model_cloud[idx]).any(1),
                index_distance=np.linalg.norm(xyz.astype(float)-model_cloud[idx],axis=1),
                physical_inside=physical_inside,physical_codes=physical_codes,
                native_seconds=single_seconds,readout_and_checks_seconds=extra_seconds)


def main():
    prepare_device()
    old,roster=pinned()
    assert tuple(PROTOCOL['arms'])==ARMS
    OUT.mkdir(mode=0o700,parents=True,exist_ok=True)
    assert not (OUT/'SEALED.json').exists()
    inputs=[HERE/'PROTOCOL.json',HERE/'predict.py',HERE/'seed_code.py',RAY/'capture.py']
    lock=dict(files={str(p.relative_to(ROOT)):digest(p) for p in inputs},parent_seal=PINS['parent_manifest'],
              roster_parent_lock=PINS['parent_lock'],arms=list(ARMS),fields=list(FIELDS),labels_read=False)
    if (OUT/'LOCK.json').exists():
        assert read(OUT/'LOCK.json')==lock
    else:
        atomic(OUT/'LOCK.json',lock)
    files,states={},{}
    start=perf_counter()
    for platform in PROTOCOL['source']:
        front=CandidateFrontend(platform)
        model=front.base
        from model51 import fingerprint
        assert not any(p.requires_grad for p in model.parameters()) and not any(m.training for m in model.modules())
        inputs = prior.source_inputs(platform)
        assert len(roster[platform])==64
        before=fingerprint(model)
        assert before==old['native_state_hashes'][platform]
        directory=OUT/platform
        directory.mkdir(mode=0o700,exist_ok=True)
        for i in roster[platform]:
            row=inputs[i]
            assert digest(Path(row['pointpath']))==row['point_sha256']
            rel=f'{platform}/{i:05}.npz'
            assert digest(PARENT/rel)==old['files'][rel]
            with np.load(PARENT/rel,allow_pickle=False) as prior_npz:
                values=infer(model,row,prior_npz,platform,i)
            path=OUT/rel
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f,**values)
            path.with_suffix('.tmp').replace(path)
            files[rel]=digest(path)
            if len(files)%32==0:
                print('Source predictions',len(files),'of 192',round(perf_counter()-start,1),'s',flush=True)
        assert fingerprint(model)==before
        states[platform]=before
        del model,front
        torch.cuda.empty_cache()
    assert len(files)==192
    atomic(OUT/'SEALED.json',dict(n=len(files),files=files,lock=digest(OUT/'LOCK.json'),
           original_native_forward_count=len(files),deployment_additional_network_forwards=0,
           native_state_sha256=states,seconds=perf_counter()-start,labels_read=False,
           target_rows=0,FINAL_access=False))
    print('Sealed',len(files),digest(OUT/'SEALED.json'),flush=True)


if __name__=='__main__':
    main()
