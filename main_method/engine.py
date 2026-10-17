"""Original Student replay and exact Temporal fit; no new learning rule."""
import collections
import copy
import hashlib
import gc
from .runtime import ROOT, read, sha, PinVerifier

_plans = {}
_source_hashes = {}
_videos = {}

def row_for(rec):
    from pathlib import Path
    key=(rec['plan_path'],rec['plan_sha256'])
    if key not in _plans:
        checker=PinVerifier({rec['plan_path']:rec['plan_sha256']})
        checker.verify()
        _plans[key]=(read(ROOT/rec['plan_path']),checker)
    plan,checker=_plans[key];checker.verify()
    subject=ROOT/rec['subject_path']
    assert sha(subject)==rec['subject_sha256']
    parse=read(subject);row=plan['rows'][int(rec['query_ordinal'])]
    assert parse['ordinal']==rec['query_ordinal'] and not parse['GT_read']
    assert parse['caption_sha256']==hashlib.sha256(row['input']['caption'].encode()).hexdigest()
    assert list(row['frame_ids'])==list(rec['frame_ids'])
    result={**row,'parses':parse['parses'],'subject_sha256':rec['subject_sha256']}
    if not getattr(checker,'qualified',False):
        from scripts.run_stvg_table1_source_v1 import _validate_roster_record
        assert result==_validate_roster_record(rec)[1]
        checker.qualified=True
    return result

def checkpoint_hash(relative):
    if relative not in _source_hashes:
        digest=sha(ROOT/relative)
        _source_hashes[relative]=(digest,PinVerifier({relative:digest}))
    digest,checker=_source_hashes[relative];checker.verify()
    return digest

def model_for(dataset):
    from scripts.run_decota_paper_main_v1 import model_for as load
    from .runtime import BASE
    spec=read(BASE/'METHOD_LOCK.json')['source_checkpoints'][dataset]
    assert checkpoint_hash(spec['path'])==spec['sha256']
    return load('vidstg' if dataset == 'hc2' else 'hcstvg2')

def capture(model, rec, frame_cache):
    """Source capture without computing any external Actionness projection."""
    import numpy as np
    import torch
    from scripts.run_decota_paper_main_v1 import frames_for
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward
    from methods.decota_final_simplified_v1.objectives import prediction
    from methods.decota_final_simplified_v1.tensors import detached
    row = row_for(rec)
    video_key=(row['input']['video_path'],row['input']['video_sha256'])
    if video_key not in _videos:_videos[video_key]=PinVerifier({video_key[0]:video_key[1]})
    _videos[video_key].verify()
    frames, ids, reused = frames_for(rec['dataset'], row, frame_cache)
    assert list(ids) == list(rec['frame_ids'])
    batch, records, base = frozen_forward(model, frames, row)
    native = prediction(base.zero['logits'], base.zero['boxes'], records, ids)
    boxes = native['boxes'].detach().cpu().numpy().copy()
    chart = boxes.astype(np.float64)
    xyxy = np.concatenate((chart[:, :2] - chart[:, 2:] / 2, chart[:, :2] + chart[:, 2:] / 2), 1)
    cp = 'checkpoints/TASTVG_VidSTG.pth' if rec['dataset'] == 'hc2' else 'checkpoints/TASTVG_HCSTVG2.pth'
    case = dict(case_id=rec['probe_alias'], dataset=rec['dataset'],
        query_ordinal=rec['query_ordinal'], query_key_sha256=rec['query_key_sha256'],
        parent_sha256=rec['parent_sha256'], subject_sha256=rec['subject_sha256'],
        input={k:v for k,v in row['input'].items() if k != 'caption'},
        width=rec['width'], height=rec['height'], video_bounds=rec['video_bounds'],
        frame_ids=np.asarray(ids), Native_interval=list(native['physical_interval']),
        native_boxes_cxcywh=boxes, native_boxes_xyxy_normalized=xyxy,
        zero=detached({k:base.zero[k] for k in ['boxes','logits','actions']}, 'cpu'),
        records=detached(records, 'cpu'), temporal_hidden=[z[-1,0].detach().cpu().clone() for z in base.temporal_inputs],
        source_model_state_sha256=model._fixed_full_state_hash,
        source_checkpoint_sha256=checkpoint_hash(cp),
        pixel_sha256=hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest(),
        GT_read=False, external_Actionness=False)
    # Immutable source identity is independent of storage paths and worker speed.
    case['source_case_sha256'] = hashlib.sha256((rec['safe_record_sha256'] + case['pixel_sha256'] + case['source_model_state_sha256']).encode()).hexdigest()
    return case, base, records, batch, frames, row

def temporal(case, hypotheses, rewards):
    from scripts import stvg_temporal_marginal_train_fit256_v1 as fitter
    from scripts.stvg_temporal_marginal_decode_util256_v1 import decode
    off = fitter.native.offsets(case, case['temporal_hidden'])
    result = fitter.fit(off, hypotheses, rewards)
    # Same complete optimizer path and independent audit as the frozen method.
    result['independent_checks'] = fitter.audit(off, result)
    result['marginal_interval'] = decode(case['records'], result['final_logits'])[0]
    return result

def original_spatial_config(dataset):
    from .runtime import BASE
    locked=BASE/'METHOD_LOCK.json'
    if locked.exists():
        cfg,settings=read(locked)['spatial']['settings'][dataset]
        return copy.deepcopy(cfg),copy.deepcopy(settings)
    d = read(ROOT/'artifacts/stvg_qwen35_geometry32_v1/arms/QWEN_OPD/trials/G32/DESIGN_LOCK.json')
    return copy.deepcopy(d['configs'][dataset]['M2']), copy.deepcopy(d['response_settings'][dataset]['M2'])

def spatial(base, case, anchors, key, checkpoint=None, qualification=False):
    from scripts.stvg_spatial_lite_fit_v1 import fit
    cfg, settings = original_spatial_config(case['dataset'])
    return fit(base, base.initial, anchors, list(map(int, case['frame_ids'])), key,
        cfg, 'M2', settings, qualification=qualification, checkpoint=checkpoint,
        recipe='lite_fullstep', response_scope='full_grid')
