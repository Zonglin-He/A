"""Seal the four registered scene/text forwards without grounding labels."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent/'3eed_v1_reciprocal_physical_roles_20261120'
sys.path.insert(0, str(PARENT))
from capture_roles import Grounder, SOURCES, observation_list, original_pool, compute_joint
from calibrate import digest, read, save
from predict import archive_npz
from mirror import Mirror, conditions
from crossplatform.candidate_frontend import WAYMO_VIEWS
from crossplatform import v1_evidence
from factor import camera
from crossplatform.round7.camera_projection import model_to_camera_matrix

WORK = HERE/'private'
RESULTS = HERE/'results'
CONFIG = HERE/'PROTOCOL.json'


def lock():
    config = read(CONFIG)
    assert digest(ROOT/config['panel_parent_receipt']) == config['parent_sha256']
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == config['source_assets_sha256']
    return dict(protocol=digest(CONFIG), source_weights=config['source_assets_sha256'],
                old_seal=config['parent_sha256'], code={n:digest(HERE/n) for n in ('mirror.py','observe.py')})


def corrected_scene(model, inp, mirror):
    """PointNet, candidate generation, PCA and decoder all see the same reflected model points."""
    import interface66
    frontend = model._frontend
    original_score, native_capture = frontend.score_frame, interface66.capture
    native_pixels = v1_evidence.visible_pixels
    recorded = {}
    def sample_forward(base, points, ids, caption):
        if mirror is None:
            next_points = points
        else:
            next_points = np.ascontiguousarray(mirror.points(points),dtype=np.float32)
        recorded.update(points=next_points, indices=hashlib.sha256(ids.tobytes()).hexdigest())
        return native_capture(base, next_points, ids, caption)
    def score_frame(**kw):
        scene = original_score(**kw)
        return replace(scene,sampled_points=recorded['points'])
    def project(points,sampled,target,view,image,calibration):
        if mirror is None:
            return native_pixels(points,sampled,target,view,image,calibration)
        uv, visible = native_pixels(mirror.points(points),mirror.points(sampled),target,view,image,calibration)
        uv[:,0] = 1. - uv[:,0]
        return uv, visible
    with ExitStack() as stack:
        stack.enter_context(patch.object(interface66,'capture',sample_forward))
        stack.enter_context(patch.object(frontend,'score_frame',score_frame))
        stack.enter_context(patch.object(v1_evidence,'visible_pixels',project))
        captured = {}
        scene,ev = model._read_observation(inp,capture=captured)
    return scene,ev,captured,recorded['indices']


def mask_spans(capture, spans):
    start,end = capture['offsets'].T
    return np.stack([(end>start)&(end>lo)&(start<hi)&capture['valid'] for lo,hi in spans])


def directional_evidence(model, frame, mirror, spec):
    text,spans = spec
    obs,ev,info,ids = corrected_scene(model,dict(frame,text=text),mirror)
    masks = mask_spans(info,spans)
    if not masks.any(1).all():
        raise RuntimeError('Mapped original language role lost all of its tokens')
    native = info['native']
    boxes, logit = v1_evidence.reference_anchors(obs.boxes,native['queries'],native['tokens'],
                                                masks,True,obs.sampled_points)
    dev=native['xyz'].device
    ev=dict(ev,role_mask=torch.as_tensor(masks[None],device=dev),
            has_reference=torch.ones(1,dtype=torch.bool,device=dev),
            reference_boxes=torch.as_tensor(boxes[None],device=dev),ref_logit=logit)
    return obs,ev,ids


def pack(model,scene,ev,mirror):
    candidates=original_pool(scene)
    axis=candidates.copy()
    axis[:,6]=0.
    s,r=[ev[k][0].double().cpu().numpy() for k in ('subject_boxes','reference_boxes')]
    if mirror is not None:
        s,r,candidates,axis=[mirror.boxes(g) for g in (s,r,candidates,axis)]
    return dict(subject=s,reference=r,pool=candidates,axis=axis,
                joint=compute_joint(model._decoder,ev) if bool(ev['has_reference'][0]) else np.zeros((4,4)))


@torch.inference_mode()
def observe(model, rec, historical):
    inp=rec['frame']
    scene,ev,cap,point_ids=corrected_scene(model,inp,None)
    baseline=model._decoder(ev)
    orig=pack(model,scene,ev,None)
    owner,ref=map(int,baseline['choices'][0].cpu().tolist())
    np.testing.assert_array_equal(orig['subject'],historical['slate'])
    assert owner==int(historical['choices'][0])
    np.testing.assert_array_equal(orig['pool'],historical['pool'])
    valid_ref=bool(ev['has_reference'][0])
    poses=conditions(inp['text'].strip(),cap['graph'],cap['document'],valid_ref)
    geom,_=camera(inp['target'],WAYMO_VIEWS.index(inp['waymo_view']),str(inp['calibration']),str(inp['rgb']))
    reflection=Mirror.camera(model_to_camera_matrix(geom))
    use=poses is not None and reflection is not None
    all_data=[orig]
    max_error=0.
    if use:
        original_text,spans=poses[0]
        assert original_text==inp['text'].strip()
        masks=mask_spans(cap,spans)
        np.testing.assert_array_equal(masks,ev['role_mask'][0].cpu().numpy())
        for pos in (1,2,3):
            m=reflection if pos in (1,3) else None
            observed,changed,ids=directional_evidence(model,inp,m,poses[pos])
            assert ids==point_ids
            if m is None:
                np.testing.assert_array_equal(observed.sampled_points,scene.sampled_points)
            else:
                delta=np.max(abs(m.points(observed.sampled_points)-scene.sampled_points))
                max_error=max(max_error,float(delta))
                assert delta<5e-5
            all_data.append(pack(model,observed,changed,m))
    return dict(**{k:np.stack([v[k] for v in all_data]) for k in orig},
                original_choice=np.array([owner,ref]),original_ref_valid=valid_ref,
                active=use,point_inverse_error=max_error)


def key(rec):
    return f"{rec['panel']}_{rec['case']:03d}_{rec['source']}.npz"


def collect(source):
    WORK.mkdir(mode=0o700,exist_ok=True)
    pin=lock()
    old=read(PARENT/'run_PRIVATE/SEAL.json')
    batch=[r for r in observation_list() if r['source']==source]
    receipts={}
    model=None
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    start=perf_counter()
    for count,r in enumerate(batch,1):
        name=key(r)
        target=WORK/name
        receipt=target.with_suffix('.json')
        if receipt.exists():
            proof=read(receipt)
            assert proof['inputs']==pin and digest(target)==proof['sha256']
        else:
            orig=PARENT/'run_PRIVATE'/name
            assert digest(orig)==old['files'][name]
            if model is None:
                model=Grounder(source)
                model._frontend.parser=lru_cache(maxsize=1024)(model._frontend.parser)
            with np.load(orig,allow_pickle=False) as prior:
                result=observe(model,r,prior)
            archive_npz(target,**result)
            proof=dict(inputs=pin,sha256=digest(target))
            save(receipt,proof)
        receipts[name]=proof['sha256']
        if count%25==0 or count==len(batch):
            print(source,count,len(batch),round(perf_counter()-start,1),flush=True)
    save(WORK/f'{source}_seal.json',dict(inputs=pin,files=receipts,n=len(batch),seconds=perf_counter()-start))


def seal():
    pin=lock()
    combined={}
    for source in SOURCES:
        item=read(WORK/f'{source}_seal.json')
        assert item['inputs']==pin and not combined.keys()&item['files'].keys()
        combined.update(item['files'])
    assert set(combined)=={key(r) for r in observation_list()} and len(combined)==501
    assert all(digest(WORK/name)==v for name,v in combined.items())
    save(WORK/'SEALED.json',dict(inputs=pin,n=501,files=combined,labels_used=False,FINAL_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=(*SOURCES,'seal'))
    phase=parser.parse_args().phase
    seal() if phase=='seal' else collect(phase)
