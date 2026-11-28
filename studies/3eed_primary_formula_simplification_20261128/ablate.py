"""Apply one specified formula change to the registered three-axis readout."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import lru_cache
from inspect import getsource
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
sys.path.insert(0,str(STUDIES/'3eed_extent_2d_factorial_20261128'))
from replay_factorial import MONO, REGISTRATION, RELOC, ROOT, check, compare_boxes, digest, load_blind, publish, ray_changes, read

REFERENCE = ROOT/'runs/extent-2d-factorial-20261128'
RUN = ROOT/'runs/primary-formula-simplification-20261128'
EPS = 1.e-8
UPSTREAM = ('O_no_ground','O_two_scales','O_no_region_center','O_h','O_fraction','O_no_vehicle_gate',
            'C_midpoint','C_no_terrain','C_no_pca_swap','R_uncapped','R_no_miss','R_no_tau_floor','R_default_pca')
FIT = ('F_off','F_height','F_horizontal','F_no_prior','F_no_measured','F_no_entries','F_no_bounds',
       'F_no_upper','F_keep_center','F_fixed_raw_z','F_no_pad')
RANK = ('R_ray_only','R_intersection','R_center','R_unclipped','R_dice','C_only_pca','C_only_qwen')
FALLBACK = ('H_visible_only','H_prior_only','A_no_front','A_no_lidar')
ARMS = ('formal',*UPSTREAM,*FIT,*RANK,*FALLBACK,'E_algebra')
EDITS = {
    'O_no_ground':('region_catalog','sample[:, 2] - sample[:, :2] @ terrain[:2] - terrain[2] > .20','np.ones(len(sample),bool)'),
    'O_no_region_center':('choose_region','(center >= MIN_RETURNS) & (overlap > 0)','overlap > 0'),
    'O_h':('choose_region','overlap[candidates].astype(float)**2 / exposure[candidates]','overlap[candidates].astype(float)'),
    'O_fraction':('choose_region','overlap[candidates].astype(float)**2 / exposure[candidates]','overlap[candidates].astype(float) / exposure[candidates]'),
    'C_midpoint':('anchor','middle + face * (dimensions[:2] - span) / 2','middle'),
    'C_no_terrain':('anchor','if terrain is not None and np.isfinite(terrain).all():','if False:'),
    'R_uncapped':('ray_score','np.minimum(residual, cap)','residual'),
    'R_no_miss':('ray_score','np.minimum(residual, cap), cap','np.minimum(residual, cap), 0.'),
    'H_visible_only':('completed_dimensions','spans + np.maximum(aligned - spans, 0.)','np.maximum(spans,1.e-6)'),
    'H_prior_only':('completed_dimensions','spans + np.maximum(aligned - spans, 0.)','aligned'),
    'A_no_front':('a_box',' + front_offset(raw, guide)',''),
    'A_no_lidar':('a_box','if len(central) >= MIN_RETURNS:','if False:'),
    'E_algebra':('completed_dimensions','spans + np.maximum(aligned - spans, 0.)','np.maximum(spans, aligned)'),
}
PAIR_EDITS = {
    'C_no_pca_swap':(('measured[6] + (np.pi/2 if spans[1] > spans[0] else 0.)','measured[6]'),),
    'R_no_tau_floor':(('max(.1, .5 * min(dimensions[:2]))','.5 * min(dimensions[:2])'),),
    'R_default_pca':(('pca_cost, qwen_cost = (ray_score(box, selection, transform, cap) for box in pair)\n    ray_winner = int(qwen_cost < pca_cost - 1.e-12 * len(selection) * cap)','ray_winner = 0'),),
}


@lru_cache(maxsize=None)
def edited(fn, replacements):
    script = dedent(getsource(fn))
    for old, new in replacements:
        assert script.count(old)==1, (fn.__name__,old)
        script = script.replace(old,new,1)
    scope = {}
    exec(compile(script,fn.__code__.co_filename,'exec'),fn.__globals__,scope)
    return scope[fn.__name__]


@contextmanager
def changed(tag, memo):
    import axiswise_2d as trial
    import cray_simple as physics

    old_choose = physics.choose_region
    with ExitStack() as st:
        if tag in EDITS:
            name,old,new = EDITS[tag]
            active = edited(getattr(physics,name),((old,new),))
            st.enter_context(patch.object(physics,name,active))
            if name in ('ray_score','completed_dimensions'):
                st.enter_context(patch.object(trial.official,name,active))
        if tag in PAIR_EDITS:
            st.enter_context(patch.object(trial.official,'prepare_pair',edited(trial.official.prepare_pair,PAIR_EDITS[tag])))
        if tag=='O_no_vehicle_gate':
            for m in (physics,trial.live):
                st.enter_context(patch.object(m,'vehicle_reply',lambda _:True))
        choose = physics.choose_region
        signature = tag if tag in ('O_no_ground','O_no_region_center','O_h','O_fraction') else 'default'

        def choose_cached(sample, terrain, scales, fit_returns, in_window, selection=None):
            scales = physics.TWO_SCALES if tag=='O_two_scales' else scales
            # Official variants share identical projected geometry. Fallbacks may
            # supply a different collection of witness rays, keyed by their bytes.
            key = signature, tuple(scales), None if selection is None else selection.tobytes()
            if key not in memo:
                memo[key] = choose(sample,terrain,scales,fit_returns,in_window,selection)
            return memo[key]
        assert callable(old_choose)
        for m in (trial.live,physics):
            st.enter_context(patch.object(m,'choose_region',choose_cached))
        yield


def modified_clip(ctx, box, pack, tag):
    """Mirror each noted formula while retaining the original clip gate."""
    import axiswise_2d as trial
    from center_sets import axes

    if tag=='F_off':
        return box
    if tag=='F_height':
        return trial.fit(ctx,box,pack,height_only=True)
    if tag=='formal':
        return trial.fit(ctx,box,pack,height_only=False)
    if tag=='F_horizontal':
        clipped = trial.fit(ctx,box,pack,height_only=False).copy()
        clipped[[2,5]]=box[[2,5]]
        return clipped if np.max(abs(clipped-box))>EPS else box
    if tag=='F_keep_center':
        clipped=trial.fit(ctx,box,pack,height_only=False).copy()
        clipped[:3]=box[:3]
        return clipped if np.max(abs(clipped-box))>EPS else box

    eye, measured, entries = pack
    size = box[3:6]
    half = size/2
    if not len(entries) or (abs(eye)<=half+EPS).all():
        return box
    groups = (entries,) if tag=='F_no_measured' else (measured,) if tag=='F_no_entries' else (measured,entries)
    pad = 0. if tag=='F_no_pad' else EPS
    vertical = 2 if tag=='F_fixed_raw_z' else int(np.argmax(abs(ctx.rotation[2])))
    prior = np.r_[np.sort(np.delete(ctx.dimensions,vertical))[::-1],ctx.dimensions[vertical]]
    lower, upper = eye[:2]<-half[:2], eye[:2]>half[:2]
    direction = lower.astype(float)-upper.astype(float)
    valid = [w for w in groups if len(w)]
    required = np.zeros(3)
    if valid and tag!='F_no_bounds':
        if tag=='E_algebra':
            low=np.minimum.reduce([w.min(0) for w in valid]); high=np.maximum.reduce([w.max(0) for w in valid])
        else:
            witness=np.concatenate(valid)
            low,high=witness.min(0),witness.max(0)
        required[:2] = np.where(lower,half[:2]+high[:2]+pad,
            np.where(upper,half[:2]-low[:2]+pad,2*(np.maximum(-low[:2],high[:2])+pad)))
        required[2] = half[2]+high[2]+pad
    prior = np.zeros(3) if tag=='F_no_prior' else prior
    dimensions = np.maximum(np.maximum(prior,required),EPS)
    if tag!='F_no_upper':
        dimensions=np.minimum(size,dimensions)
    result=box.copy()
    result[2]+=(dimensions[2]-size[2])/2
    # Use the same sign and multiplication order as the frozen implementation.
    offset=(size[:2]-dimensions[:2])*(-direction)/2
    result[:2]+=axes(box[6])[:2,:2]@offset
    result[3:6]=dimensions
    return result if np.max(abs(result-box))>EPS else box


def rank_variant(ctx, boxes, tag):
    import axiswise_2d as trial

    if tag in ('R_ray_only','C_only_pca','C_only_qwen'):
        return {'R_ray_only':ctx.ray_winner,'C_only_pca':0,'C_only_qwen':1}[tag]
    if tag not in ('R_dice','R_intersection','R_center','R_unclipped','E_algebra'):
        return trial.rank(ctx,boxes)[0]
    if ctx.rectangle is None:
        return ctx.ray_winner
    target=ctx.rectangle
    if tag in ('R_dice','E_algebra'):
        scores=trial.rank(ctx,boxes)[1]
        values=[None if v is None else 2*v/(1+v) for v in scores]
    else:
        values=[]
        for box in boxes:
            g=trial.geometry; official=trial.official
            corners=official.extent_corners(box[:3],box[3:6],official.z_rotation(np.rad2deg(box[6])))
            pixels=official.sensor_pixels(g.rigid_xyz(corners,ctx.transform),ctx.spec)
            if pixels is None:
                values.append(None)
                continue
            r=np.r_[pixels.min(0),pixels.max(0)]
            if tag!='R_unclipped':
                r=official.visible_rectangle(r,ctx.image_size)
            if r is None or not np.isfinite(r).all() or min(r[2:]-r[:2])<2:
                values.append(None)
                continue
            overlap=np.prod(np.maximum(0.,np.minimum(r[2:],target[2:])-np.maximum(r[:2],target[:2])))
            if tag=='R_unclipped':
                overlap/=np.prod(r[2:]-r[:2])+np.prod(target[2:]-target[:2])-overlap
            if tag=='R_center':
                overlap=1./(1.+np.linalg.norm((r[:2]+r[2:]-target[:2]-target[2:])/2))
            values.append(float(overlap))
    old=ctx.ray_winner; new=1-old
    return new if values[new] is not None and values[new]>(values[old] if values[old] is not None else 0.) else old


def sealed_inputs():
    from run_qwen_3eed import height_2d_registration,input_frames,registered_predictor
    source, old, *_ = load_blind()
    current=read(REGISTRATION); archived,_=height_2d_registration(current)
    public=registered_predictor(current)
    assert current['method']=='ray_equivalent_extent_2d_rank'
    pred=REFERENCE/'predictions-private.json'; pin=check(pred,'payload',REFERENCE/'prediction-seal.json')
    assert not pin['labels_read'] and pin['registration']==digest(archived)
    refs={r['uid']:r for r in read(pred)}
    mono=MONO/'mono-raw-predictions-private.json'; stamp=check(mono,'payload',MONO/'mono-prediction-seal.json')
    assert not stamp['target_labels_read'] and stamp['registered_method']==digest(archived)
    inp=MONO/'mono-inputs.json'; assert digest(inp)==pin['sources']['mono_inputs']
    inputs=input_frames(inp)
    source.update({r['uid']:dict(target='mono',recording=None,frame=inputs[r['uid']],raw=r['replies']) for r in read(mono)['predictions']})
    assert refs.keys()==source.keys() and len(source)==873
    return source,old,refs,dict(registration=digest(REGISTRATION),history=pin['payload'],external=stamp['payload']),public


def fitting_audit(ctx, ref, output, pack, rays, before):
    import axiswise_2d as trial
    from center_sets import local

    off=local(output[None,:3],ref)[0]
    half=output[3:6]/2
    _,measured,entries=pack
    lost=lambda a:int((abs(a-off)>half+EPS).any(1).sum())
    out=dict(non_nested=bool((abs(off)+half>ref[3:6]/2+EPS).any()),measured_lost=lost(measured),entries_lost=lost(entries))
    for name,b,r in zip(('selection','audit'),before,rays,strict=True):
        out[name]=ray_changes(b,trial.live.first_depth(output,r,ctx.transform))
    return out


def one(task):
    import axiswise_2d as trial
    uid, row, original, sealed = task
    frame, replies = row['frame'],row['raw']
    memo,resident={},{}
    predictions={}
    fit_audit={}
    with ExitStack() as st:
        for name in ('camera','raw_returns','fit_sample','model_sample'):
            fn=getattr(trial.geometry,name)
            def stored(*args,key=name,fn=fn):
                if key not in resident:
                    resident[key]=fn(*args)
                return resident[key]
            st.enter_context(patch.object(trial.geometry,name,stored))
        with changed('formal',memo):
            ctx=trial.prepare(frame,replies)
            formal = trial.fallback(frame,replies) if ctx is None else None
        if ctx is not None:
            pack=[trial.witnesses(ctx,b) for b in ctx.boxes]
            formal_boxes=tuple(modified_clip(ctx,b,w,'formal') for b,w in zip(ctx.boxes,pack,strict=True))
            formal=trial.output(ctx,formal_boxes,trial.rank(ctx,formal_boxes)[0])
        assert formal==sealed['outputs']['axis_2d'],uid
        predictions=dict.fromkeys(ARMS,formal)
        for tag in UPSTREAM:
            if formal['box'] is None or (tag.startswith('R_') and ctx is None):
                continue
            with changed(tag,memo):
                predictions[tag]=trial.predict(frame,replies)
        if ctx is None:
            branch=formal['readout']
            for tag in (*FALLBACK,'E_algebra'):
                if branch in ('H','A') and tag.startswith(branch+'_') or (branch=='H' and tag=='E_algebra'):
                    with changed(tag,memo):
                        predictions[tag]=trial.fallback(frame,replies)
        else:
            for tag in RANK:
                k=rank_variant(ctx,formal_boxes,tag)
                predictions[tag]=trial.output(ctx,formal_boxes,k)
            if original and original['route']=='C_ray':
                audit=np.asarray(original['audit']).reshape(-1,3)
                np.testing.assert_array_equal(ctx.selection,np.asarray(original['selection']).reshape(-1,3))
            else:
                with changed('formal',memo):
                    _, physical=trial.live.incumbent(frame,replies,collect_audit=True)
                assert physical is not None
                np.testing.assert_array_equal(ctx.selection,physical[4]); audit=physical[5]
            rays=(ctx.selection,audit)
            first=[tuple(trial.live.first_depth(b,a,ctx.transform) for a in rays) for b in ctx.boxes]
            for tag in ('formal',*FIT,'E_algebra'):
                boxes=formal_boxes if tag=='formal' else tuple(modified_clip(ctx,b,w,tag) for b,w in zip(ctx.boxes,pack,strict=True))
                k=rank_variant(ctx,boxes,tag)
                predictions[tag]=trial.output(ctx,boxes,k)
                fit_audit[tag]=[fitting_audit(ctx,b,v,w,rays,z) for b,v,w,z in zip(ctx.boxes,boxes,pack,first,strict=True)]
            for arm, other in (('F_height','height_2d'),('R_ray_only','axis_ray')):
                compare_boxes(predictions[arm]['box'],sealed['outputs'][other]['box'])
    return dict(uid=uid,target=row['target'],recording=row['recording'],boxes={k:v['box'] for k,v in predictions.items()},
                route={k:v['readout'] for k,v in predictions.items()}, audit=fit_audit)


def main():
    from concurrent.futures import ProcessPoolExecutor
    import cv2

    cv2.setNumThreads(1)
    def reject_labels(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            from os import fsdecode
            if Path(fsdecode(args[0])).name in {'scored-private.json','mono-scores-private.json','test.parquet','data_object_label_2.zip'}:
                raise PermissionError('Offline labels are blocked during prediction')
    sys.addaudithook(reject_labels)
    source,old,refs,pin,_=sealed_inputs()
    RUN.mkdir(exist_ok=True,parents=True)
    jobs=[(u,r,old.get(u),refs[u]) for u,r in source.items()]
    with ProcessPoolExecutor(10) as pool:
        predictions=[]
        for row in pool.map(one,jobs,chunksize=3):
            predictions.append(row)
            if len(predictions)%100==0 or len(predictions)==len(jobs):
                print('Replayed',len(predictions),'/',len(jobs),'across',len(ARMS)-1,'variants',flush=True)
    payload=RUN/'blind.json'
    publish(payload,predictions)
    publish(RUN/'prediction-seal.json',dict(source=pin,n=len(predictions),arms=ARMS,payload=digest(payload),
        files={k:digest(HERE/k) for k in ('ablate.py','PROTOCOL_ZH.md')},labels_read=False,training=False,model_calls=0))


if __name__=='__main__':
    main()
