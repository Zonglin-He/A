"""Prepare the locked sensor evidence, run CPU solvers, and seal all boxes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

from solver import ARMS, Inputs, physical, ray_depth, solve

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
sys.path.insert(0,str(STUDIES/'3eed_2d_relocalization_20261128'))
import fresh_relocalize  # Resolve historical modules before selecting the runtime paths.
from shared import FrameEvidence, ROOT, canonical, digest, guard_labels, publish, ray_geometry, read, setup, subsample
from metric_image_common import inputs
from replay import context_pins

RUN = ROOT/'runs/nullspace-solver-20261128'
RELOC = ROOT/'runs/2d-relocalization-20261128'
R4 = ROOT/'runs/five-branches-20261128'
REPLAY = ROOT/'runs/cascaded-oracle-20261128'


def check(path, key, receipt):
    value = read(receipt)
    assert value[key] == digest(path)
    return value


def load_blind():
    from geometry_predict import guard_study_annotations

    guard_labels()
    guard_study_annotations()
    setup()
    o0 = REPLAY/'o0.json'
    official = check(o0,'payload',REPLAY/'o0-seal.json')
    pins = context_pins()
    legacy = STUDIES/'3eed_qwen_extent_height_only_20261122/LEGACY_REGISTRATION.json'
    pins['registration'] = digest(legacy)
    assert pins==official['pins'] and not official['target_labels_read']
    inventory = RELOC/'candidate-blind.json'
    selection = RELOC/'choices-blind.json'
    inv_stamp = check(inventory,'file',RELOC/'candidate-seal.json')
    cho_stamp = check(selection,'file',RELOC/'choices-seal.json')
    assert inv_stamp['file'] in cho_stamp['input'] and inv_stamp['lock']==official['payload']
    assert not inv_stamp['labels_read'] and not cho_stamp['labels_read']
    rows = inputs()
    old = {x['uid']:x for x in read(o0)}
    inv = {x['uid']:x for x in read(inventory) if x['group']=='main'}
    chose = {x['uid']:x for x in read(selection) if x['group']=='main'}
    assert rows.keys()==old.keys()==inv.keys()==chose.keys() and len(rows)==702
    receipt = dict(o0=official['payload'], old_inventory=inv_stamp['file'], old_choices=cho_stamp['file'],
                   registered=digest(HERE.parents[1]/'QWEN_CURRENT_METHOD.json'))
    return rows,old,inv,chose,receipt


def rect_for(row, ev):
    if row['raw']['2d']['value'] is None:
        return None
    window = np.asarray(row['raw']['2d']['value'],dtype=float)*np.tile(ev.size,2)/1000.
    window = np.clip(window,0,np.tile(ev.size,2))
    if np.isfinite(window).all() and min(window[2:]-window[:2])>=2.:
        return window
    return None


def rays_for(ev, rows, window):
    K,_,dist,_ = ev.spec.validated()
    rows = subsample(np.asarray(rows).reshape(-1,3),192)
    eye, rays = ray_geometry(rows,ev.mat)
    return Inputs(eye,rays,rows[:,2],ev.mat,K,dist,np.asarray(ev.size),window)


def routefree_seed(prior, raw, ev, observed, window):
    from geometry import front_offset, pixel_rays

    pose = prior.copy()
    fit = np.asarray(observed['fit']).reshape(-1,3)
    if len(fit):
        guide = pixel_rays((window[:2]+window[2:])/2,ev.spec)[0]
        depth = float(np.median(fit[:,2])) + front_offset(raw,guide)
        metric = ev.module.rigid_xyz((guide*depth)[None,:],ev.cam_to_raw)[0]
        blend = len(fit)/(len(fit)+5.)
        pose[:3] += blend*(metric-pose[:3])
    return pose


def read_routefree(qwen, raw, ev, generic, window, evidence):
    seed = routefree_seed(qwen, raw, ev, generic, window)
    box, diagnostics = solve(seed,qwen,evidence,ARMS['joint_soft'])
    return seed,box,diagnostics


def predict_routefree(frame, replies):
    """Use cached Qwen replies and sensor inputs with no C_ray/H/A dependency."""
    if replies['3d']['value'] is None:
        return None
    row = dict(frame=frame,raw=replies)
    ev = FrameEvidence(frame)
    raw = np.asarray(replies['3d']['value'],dtype=float)
    qwen = canonical(ev.module.to_sensor(raw,ev.mat)[0])
    window = rect_for(row,ev)
    observed = ev.observed(window)
    sensor = rays_for(ev,observed['selection'],window)
    _,box,diagnostics = read_routefree(qwen,raw,ev,observed,window,sensor)
    return dict(box=box.tolist(),diagnostics=diagnostics)


def marks(box, audit):
    if box is None or not len(audit.rays):
        return None
    depths,face = ray_depth(box[None,:],audit.eye,audit.rays)
    _,robust,_ = physical(depths[0],face[0],audit.observed,True)
    cut = np.where(face[0]>=0,np.minimum(abs(depths[0]-audit.observed),.75),.75)
    return dict(truncated_m=float(cut.mean()),huber=float(robust.mean()),hit=float((face[0]>=0).mean()))


def one(job):
    uid,row,diag,inv,choice = job
    tick = perf_counter()
    b = inv['pool']['cray']['boxes']
    formal = None if not b else b[choice['selected']['cray']['cached']]
    legacy = None if not b else b[0]
    raw = row['raw']['3d']['value']
    meta = dict(uid=uid,target=row['target'],recording=row['recording'],route=diag['route'])
    references = dict(formal=formal,legacy=legacy)
    keys = (*ARMS,'routefree_initial','routefree')
    if raw is None or formal is None:
        assert raw is None and formal is None
        return dict(meta,boxes=dict(references,**dict.fromkeys(keys)), details={},audit={},
                    n_selection=0,n_audit=0,n_routefree=0,n_fit_routefree=0,seconds=perf_counter()-tick)

    ev = FrameEvidence(row['frame'])
    qwen = canonical(ev.module.to_sensor(np.asarray(raw),ev.mat)[0])
    window = rect_for(row,ev)
    generic = ev.observed(window)
    physical_selection = diag['selection'] or generic['selection']
    audit_selection = diag['audit'] or generic['audit']
    observation = rays_for(ev,physical_selection,window)
    audit = rays_for(ev,audit_selection,window)
    generic_observation = rays_for(ev,generic['selection'],window)
    generic_audit = rays_for(ev,generic['audit'],window)
    boxes, details = {}, {}
    for name,arm in ARMS.items():
        boxes[name],details[name] = solve(canonical(formal),qwen,observation,arm)
    seed, independent, diagnostics = read_routefree(qwen,np.asarray(raw),ev,generic,window,generic_observation)
    boxes['routefree_initial'] = seed
    boxes['routefree'] = independent
    details['routefree'] = diagnostics
    info = {k:marks(box,generic_audit if k.startswith('routefree') else audit) for k,box in boxes.items()}
    info.update(formal=marks(canonical(formal),audit),legacy=marks(canonical(legacy),audit))
    # Compare the route-free change to its own initialization on its own held-out rays.
    info['formal_on_routefree_audit'] = marks(canonical(formal),generic_audit)
    output = dict(references,**{k:b.tolist() for k,b in boxes.items()})
    return dict(meta,boxes=output,details=details,audit=info,
                n_selection=len(observation.rays),n_audit=len(audit.rays),
                n_routefree=len(generic_observation.rays),n_fit_routefree=len(generic['fit']),
                seconds=perf_counter()-tick)


def main():
    cv2.setNumThreads(1)
    RUN.mkdir(exist_ok=True,parents=True)
    data,old,inv,chose,receipt=load_blind()
    jobs=[(u,row,old[u],inv[u],chose[u]) for u,row in data.items()]
    started=perf_counter()
    out=[]
    with ProcessPoolExecutor(10) as threads:
        for n,x in enumerate(threads.map(one,jobs,chunksize=3),1):
            out.append(x)
            if n%100==0 or n==len(jobs):
                print('Solved',n,'/',len(jobs),flush=True)
    assert len(out)==len(data)
    artifact=RUN/'predictions-private.json'
    publish(artifact,out)
    pins={p:digest(HERE/p) for p in ('solver.py','run_experiment.py','PROTOCOL_ZH.md')}
    seal=dict(artifact=digest(artifact),n=len(out),planned_input=receipt,pins=pins,
              routes=dict(Counter(x['route'] for x in out)),source_model_calls=0,training=False,
              target_labels_read=False,cpu_wall_s=perf_counter()-started)
    publish(RUN/'prediction-seal.json',seal)
    print(seal,flush=True)


if __name__=='__main__':
    main()
