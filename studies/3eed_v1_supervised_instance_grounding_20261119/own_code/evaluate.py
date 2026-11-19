"""Generate and seal all six-direction predictions before reading target labels."""

import argparse
import os
import secrets
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from features import encode, observation, feature_dependencies
from ground_instance import SourceInstanceGrounder, tensorize
from model import NATIVE, VOTES
from scope import HERE, PLATFORMS, evaluation_rows, file_sha, protocol, read, save
from train import MODELS, decisions, metrics
from crossplatform.candidate_frontend import sensor_frame_box

sys.path.insert(0,str(HERE.parent / '3eed_v1_component_ablation_20261119'))
from score_ablation import (DIRECTIONS, many_iou, recording_intervals, save_csv,
                            score_boxes, sensor_truth, summarize)

PREVIOUS = HERE.parent / '3eed_v1_frontend_simplification_20261119' / 'predictions_PRIVATE'
PRIVATE = HERE / 'predictions_PRIVATE'
RANGE = ('v1_four_slots','A256','A256_plus_votes','geometry_union','selected_geometry_pool')


def old_boxes(rows, source, target):
    path = PREVIOUS / f'{source}_to_{target}.npz'
    root = PREVIOUS/'SEALED.json'
    assert file_sha(root) == protocol()['baseline_predictions_sha256']
    seal = read(root)
    assert not seal['FINAL_access'] and not seal['target_labels_read']
    assert file_sha(path) == seal['files'][path.name]['sha256']
    selected = np.flatnonzero([row['platform']==target for row in rows])
    with np.load(path,allow_pickle=False) as n:
        np.testing.assert_array_equal(n['ordinals'],selected)
        return selected,n['boxes']


def fingerprint():
    items = {p.name:file_sha(p) for p in (Path(__file__),HERE/'ground_instance.py',
                                          HERE/'model.py',HERE/'train.py')}
    items.update(feature_dependencies())
    items['weight_seal'] = file_sha(MODELS/'SEALED.json')
    return items


def to_sensor(boxes,target,view):
    a = np.asarray(boxes,dtype=np.float64).copy()
    if target == 'waymo':
        from crossplatform.round7.camera_projection import WAYMO_ANGLES,z_rotation
        a[:,:3] = a[:,:3] @ z_rotation(WAYMO_ANGLES[view])
        a[:,6] = (a[:,6]-np.deg2rad(WAYMO_ANGLES[view])+np.pi/2) % np.pi - np.pi/2
    elif target == 'drone':
        a[:,2] -= 1.8
    return a


@torch.inference_mode()
def predict_row(method,row,before):
    frame = observation(row)
    slot_boxes = []

    def capture_slots(_module, inputs):
        slot_boxes.append(inputs[0]['subject_boxes'][0].cpu().numpy().copy())

    handle = method.frozen._decoder.register_forward_pre_hook(capture_slots)
    try:
        feature,_ = encode(method.frozen,frame)
    finally:
        handle.remove()
    slots, = slot_boxes
    assert slots.shape == (4, 7)
    batch = tensorize(feature)
    head = method.head
    out = head(batch)
    actual, refs, known, ids = decisions(out,batch)
    native_out = head(batch,use_votes=False)
    no_mask = head(batch,use_mask=False)
    altered = head.joint(batch,out,out['subject'],out['unsuppressed_reference'],
                         out['camera_xy'],out['camera_visible'],mismatch=True)
    # Same candidates and model scores, different reference-to-instance assignment.
    for name,value in (('native_carriers_only',native_out),('no_membership_weight',no_mask),
                       ('wrong_reference',dict(out,**altered))):
        chosen,nrefs,nknown,nids = decisions(value,batch)
        actual[name],refs[name],known[name],ids[name] = (chosen['full_pool_joint'],
                   nrefs['full_pool_joint'],nknown['full_pool_joint'],nids['full_pool_joint'])
    selection = method.source_readout
    actual['source_selected'],refs['source_selected'] = actual[selection],refs[selection]
    known['source_selected'],ids['source_selected'] = known[selection],ids[selection]
    target = row['platform']
    view = int(feature['view_index'])
    arms = protocol()['comparisons']
    def converted(value):
        return sensor_frame_box(value[0].cpu().numpy(),target,view)
    picked = np.stack([converted(actual[x]) for x in arms])
    referents = np.stack([converted(refs[x]) for x in arms])
    np.testing.assert_array_equal(picked[0],before)
    assert np.isfinite(picked).all() and (picked[:,3:6] > 0).all()
    pool = np.concatenate((feature['anchors'],out['proposal'][0,NATIVE:].cpu().numpy(),
                           out['refined'][0].cpu().numpy(),out['boxes'][0].cpu().numpy()))
    pair_selected = np.asarray([bool(known[x][0]) for x in arms])
    relation = int(feature['relation'])
    return dict(picked=picked, referents=referents, pool=to_sensor(pool,target,view),
                v1_slots=to_sensor(slots,target,view),
                reference_known=pair_selected,indices=np.asarray([int(ids[x][0]) for x in arms]),
                role_flags=np.array([bool(feature['reference_phrase']),abs(relation)==1,relation!=0]),
                source_readout=np.asarray(selection))


def predict(smoke):
    cfg = protocol()
    rows = evaluation_rows()
    target_dir = HERE/'target_smoke_PRIVATE' if smoke else PRIVATE
    target_dir.mkdir(mode=0o700)
    lock = dict(implementation=fingerprint(),registered_inputs=cfg['evaluation']['sha256'],
                FINAL_access=False,target_labels_read=False)
    save(target_dir/'LOCK.json',lock)
    start = perf_counter()
    files = {}
    for source in PLATFORMS:
        model = SourceInstanceGrounder(source)
        for target in PLATFORMS:
            if source==target:
                continue
            indices, old = old_boxes(rows,source,target)
            if smoke:
                indices,old=indices[:2],old[:2]
            outputs=[]
            for j,(ordinal,before) in enumerate(zip(indices,old),1):
                value = predict_row(model,rows[int(ordinal)],before)
                outputs.append(value)
                if j%64==0 or j==len(indices):
                    print(source,target,j,'/',len(indices),'seconds',round(perf_counter()-start,1),flush=True)
            p=target_dir/f'{source}_to_{target}.npz'
            fields={key:np.stack([d[key] for d in outputs]) for key in outputs[0]}
            draft=p.with_suffix('.tmp')
            with draft.open('wb') as stream:
                np.savez_compressed(stream,ordinals=indices,**fields)
            os.replace(draft,p)
            files[p.name]=dict(n=len(indices),sha256=file_sha(p))
        del model
        torch.cuda.empty_cache()
    n=sum(x['n'] for x in files.values())
    assert n==(12 if smoke else cfg['evaluation']['records'])
    assert fingerprint()==lock['implementation']
    save(target_dir/'SEALED.json',dict(n=n,files=files,lock_sha256=file_sha(target_dir/'LOCK.json'),
                                      target_labels_read=False,FINAL_access=False,
                                      bit_exact_v1=n,seconds=perf_counter()-start))


def aggregate_coverage(scores):
    rows=[]
    for direction in ('six_equal',*scores):
        collected=[x for d,x in scores.items() if direction=='six_equal' or direction==d]
        a=np.concatenate(collected,axis=0)
        result=np.mean([np.array([metrics(block[:,j])['Acc25'] for j in range(len(RANGE))])
                        for block in collected],axis=0)
        strict=np.mean([np.array([metrics(block[:,j])['Acc50'] for j in range(len(RANGE))])
                        for block in collected],axis=0)
        for j,key in enumerate(RANGE):
            rows.append(dict(direction=direction,stage=key,n=len(a),
                         count25=int((a[:,j]>.25).sum()),count50=int((a[:,j]>.50).sum()),
                         Acc25=float(result[j]),Acc50=float(strict[j])))
    return rows


def attribute_coverage(scored, coverage, arms):
    rows = []
    for group in ('six_equal', *scored):
        directions = tuple(scored) if group == 'six_equal' else (group,)
        iou = np.concatenate([scored[d]['ious'] for d in directions])
        oracles = np.concatenate([coverage[d] for d in directions])
        for threshold in (.25, .50):
            good_slot, good_anchor, good_vote, good_union, good_delivered = (oracles > threshold).T
            assert np.all(~good_slot | good_anchor)
            assert np.all(~good_anchor | good_vote) and np.all(~good_vote | good_union)
            incumbent = iou[:,0] > threshold
            assert np.all(~incumbent | good_slot)
            for arm in ('full_pool_unary','full_pool_joint','source_selected'):
                improved = (iou[:,arms.index(arm)] > threshold) & ~incumbent
                rows.append(dict(group=group,threshold=threshold,arm=arm,n=len(iou),
                    v1_fail_no_native_candidate=int((~good_anchor).sum()),
                    v1_fail_missed_by_four_slots=int((good_anchor & ~good_slot).sum()),
                    v1_fail_four_slots_contain_good_box=int((good_slot & ~incumbent).sum()),
                    geometry_oracle_rescues=int((good_union & ~good_anchor).sum()),
                    geometry_selection_new_coverage=int((good_delivered & ~good_anchor).sum()),
                    geometry_selection_lost_coverage=int((~good_delivered & good_anchor).sum()),
                    gained_no_old_candidate=int((improved & ~good_anchor).sum()),
                    gained_old_candidate_pruned=int((improved & good_anchor & ~good_slot).sum()),
                    gained_good_old_slot=int((improved & good_slot).sum())))
    return rows


def score():
    rows=evaluation_rows()
    cfg=protocol()
    arms=tuple(cfg['comparisons'])
    lock,seal = read(PRIVATE/'LOCK.json'),read(PRIVATE/'SEALED.json')
    assert seal['lock_sha256']==file_sha(PRIVATE/'LOCK.json')
    assert seal['n']==cfg['evaluation']['records'] and seal['bit_exact_v1']==seal['n']
    assert not seal['FINAL_access'] and not seal['target_labels_read']
    assert lock['implementation']==fingerprint()
    saved={}
    for source,target in DIRECTIONS:
        path=PRIVATE/f'{source}_to_{target}.npz'
        assert seal['files'][path.name]['sha256']==file_sha(path)
        indices,old=old_boxes(rows,source,target)
        with np.load(path,allow_pickle=False) as n:
            np.testing.assert_array_equal(n['ordinals'],indices)
            np.testing.assert_array_equal(n['picked'][:,0],old)
            saved[source,target]={key:n[key] for key in n.files}
    coverage={}
    # Index-free scalar rows are published. The physical frames and coordinates stay sealed locally.
    rows_out=[]
    err=0.
    independent=0
    total=NATIVE+VOTES
    all_io={}
    for (source,target), data in saved.items():
        direction=f'{source}_to_{target}'
        values,groups,oracles = [],[],[]
        for k,ordinal in enumerate(data['ordinals']):
            row=rows[int(ordinal)]
            gt=sensor_truth(target,row['metapath'],row['caption'],row['suffix'])
            ious,delta=score_boxes(data['picked'][k],gt)
            err=max(err,delta)
            values.append(ious)
            groups.append(row['scene_cluster'])
            raw=many_iou(data['pool'][k],gt)
            intervals=(raw[:NATIVE],raw[:total],raw[:2*total],raw[2*total:])
            assert len(raw)==3*total
            slots,slot_error=score_boxes(data['v1_slots'][k],gt)
            oracles.append([float(slots.max()),*(float(x.max()) for x in intervals)])
            sizes=[0,total,2*total]
            checks=np.unique([j+raw[j:k].argmax() for j,k in zip(sizes,(total,2*total,3*total))])
            exact,e=score_boxes(data['pool'][k][checks],gt)
            np.testing.assert_allclose(raw[checks],exact,atol=1e-10,rtol=0)
            err=max(err,e,slot_error)
            independent+=len(checks)+len(slots)
            counts=data['indices'][k]
            relation=data['role_flags'][k]
            ra=data['reference_known'][k]
            rows_out.append(dict(source=source,target=target,reference=int(relation[0]),
                             direct_left_right=int(relation[1]),active_relation=int(relation[2]),
                             joint_unknown=int(not ra[arms.index('full_pool_joint')]),
                             joint_vote_selected=int(counts[arms.index('full_pool_joint')]>=NATIVE),
                             source_vote_selected=int(counts[arms.index('source_selected')]>=NATIVE),
                             **dict(zip(arms,(float(x) for x in ious)))))
        all_io[direction]=dict(ious=np.stack(values),roles=data['role_flags'][:,:2],recordings=np.asarray(groups))
        coverage[direction]=np.stack(oracles)
        print('independently scored',direction,len(values),flush=True)
    out=HERE/'results'
    out.mkdir()
    role_subsets = ('reference_mention','direct_left_right','no_reference')
    selectors = (lambda bits: bits[:,0],lambda bits: bits[:,1],lambda bits: ~bits[:,0])
    subsets = ('all',) + tuple(name for name,choose in zip(role_subsets,selectors)
                               if all(choose(part['roles']).any() for part in all_io.values()))
    endpoints,paired=summarize(all_io,arms,subsets)
    save_csv(out/'endpoints.csv',endpoints)
    save_csv(out/'paired.csv',paired)
    save_csv(out/'coverage.csv',aggregate_coverage(coverage))
    save_csv(out/'coverage_attribution.csv',attribute_coverage(all_io,coverage,arms))
    secrets.SystemRandom().shuffle(rows_out)
    save_csv(out/'anonymous_rows.csv',rows_out)
    save(out/'recording_intervals.json',recording_intervals(all_io,arms))
    save(out/'audit.json',dict(n=len(rows_out),endpoints=len(rows_out)*len(arms),
         independent_pool_checks=independent,oriented_IoU_max_abs_error=err,threshold_disagreements=0,
         exact_v1_matches=seal['bit_exact_v1'],prediction_seal_sha256=file_sha(PRIVATE/'SEALED.json'),
         source_weight_seal=file_sha(MODELS/'SEALED.json'),all_labels_read_after_prediction_seal=True,
         reported_subsets=list(subsets),target_updates=0,FINAL_access=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('smoke','predict','score'))
    mode=cli.parse_args().phase
    if mode=='score':
        score()
    else:
        predict(smoke=mode=='smoke')
