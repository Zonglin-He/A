"""Use the frozen R169 C/D readout on the recording-separated trial."""
from collections import Counter
from functools import lru_cache
import fcntl
import gc
import time

import numpy as np

from scope170 import (ARMS, HERE, INPUT, PLATFORMS, R159, REC_INPUT, ROOT, SOURCE_SETTINGS, VARIANTS, WORK,
                      atomic_json, lock, read, sha, status)
from calibrate168 import edges_of, frozen_pool
from common160 import array_sha, npz_atomic
from geometry162 import in_box
from requery169 import associate, native_donors, native_forward, pad
from replay168 import oriented
from retrieval167 import repaired, support_matrix
from scope167 import BANK
from select168 import solve
from crossplatform.ground_single_frame import orient_box


@lru_cache(maxsize=3)
def old_baseline(source):
    path=ROOT/'crossplatform/round159'/(source+'_PRIVATE.npz')
    sealed=read(R159/'RECORDING_PREDICTION_SEAL.json')
    assert not sealed['FINAL_access'] and not sealed['target_grounding_labels_read']
    assert sha(path)==sealed['sources'][source]['sha256']
    with np.load(path,allow_pickle=False) as old:
        return {int(i):(w,b) for i,w,b in zip(old['rows'],old['winners'],old['boxes'][:,2],strict=True)}


def requery(model, entry, ev, old, original, pool_mask, receiver_points, arm):
    role=entry['role']
    available=old['bank'][role]>=0
    p=entry['rewrite']
    if p['status']=='supported':
        z=native_forward(model,entry['row'],ev.points,ev.indices,None,p[arm])
        np.testing.assert_array_equal(z['z']['xyz'],original['xyz'])
        np.testing.assert_array_equal(z['z']['point_indices'],ev.indices)
        np.testing.assert_array_equal(z['z']['point_indices'],original['point_indices'])
        ids,value,_=native_donors(z)
        native=np.stack([orient_box(ev.points,z['z']['boxes'][j]) for j in ids])
        started=time.perf_counter()
        donors=np.stack([in_box(receiver_points,np.r_[z['z']['boxes'][j],0.]) for j in ids])
        evidence,links,witness=associate(donors,pool_mask,value)
        slots=int((z['z']['query_points_sample_inds']==original['query_points_sample_inds']).sum())
        elapsed=z['forward_seconds']
        did_rewrite=True
    else:
        started=time.perf_counter()
        native=np.zeros((0,7))
        evidence=np.full(len(pool_mask),-np.inf)
        links=np.zeros(0,dtype=np.int16)
        witness=np.zeros(0)
        slots=-1
        elapsed=0.
        did_rewrite=False
    scores=old['score'].copy()
    linked=int(np.isfinite(evidence).sum())
    scores[role,available]=evidence
    scores[role,old['bank'][role]==-1]=-2. if linked else 0.
    active=int(linked>0 and np.any(old['left'][role-1][:,available][:,np.isfinite(evidence)]))
    chosen=np.stack([solve(old['bank'],scores,old['left'],old['overlap'],{role:entry['sign']},
                           overlap_on=False,independent=i)[0][[0,role]] for i in (True,False)])
    boxes=oriented(entry['row'],old,ev.boxes,chosen,lambda _:ev.points)
    assert chosen[0,0]==ev.old_winner
    np.testing.assert_array_equal(boxes[0,0],ev.old_pca)
    if not linked:
        assert (chosen[:,1]==-1).all() and (chosen[:,0]==ev.old_winner).all()
    return dict(natural=pad(native,BANK,np.nan),native_n=len(native), links=pad(links,BANK,-1),
                witnesses=pad(witness,BANK,0.),bank_scores=pad(evidence,BANK,-np.inf),
                mapped_n=linked, choices=chosen, boxes=boxes, executed=int(did_rewrite),
                active_relation=active,forward_seconds=elapsed, association_seconds=time.perf_counter()-started,
                same_slot_seeds=slots)


def original_evidence(model,row,setting):
    from interface66 import capture

    ev=model.evidence_frame(row)
    original,_=capture(model.base,ev.points,ev.indices,row['caption'])
    np.testing.assert_array_equal(original['boxes'],ev.boxes)
    np.testing.assert_array_equal(original['xyz'][0],ev.seeds)
    prior=frozen_pool(ev,setting)
    old=repaired(ev,prior,original['query'],original['tokens'])
    return ev,original,prior,old


def bridge(model,setting):
    from scope169 import PANEL as R169_PANEL
    from scope170 import R169,roster
    from panel169 import lookup

    path=R169_PANEL/'INPUT_PRIVATE.json'
    assert sha(path)==read(R169/'PANEL_INPUT_SEAL.json')['roster']
    entry=next(x for x in read(path) if x['source']==model.source and x['status']=='supported')
    row=roster()[entry['target']][entry['ordinal']]
    k,arch,base,earlier=lookup(entry)
    ev,z,prior,old=original_evidence(model,row,setting)
    np.testing.assert_array_equal(ev.boxes,base['raw'][k])
    np.testing.assert_array_equal(ev.seeds,earlier['seeds'][k])
    np.testing.assert_array_equal(prior['starts'],earlier['starts'][k])
    routes=earlier['routes'][k]
    active=len(prior['routes'])
    np.testing.assert_array_equal(prior['routes'],routes[:active])
    assert (routes[active:]==-1).all()
    # R166's middle budget cell counts its original-only ablation. R167 consumes
    # the full and novel counts, and does not read that ablation column.
    np.testing.assert_array_equal(prior['budget'][:,[0,2]],earlier['budget'][k][:,[0,2]])
    for key in ('bank','novel','winners'):
        np.testing.assert_array_equal(old[key],arch[key][k])
    score_error=0.
    for key in ('score','left','overlap'):
        np.testing.assert_allclose(old[key],arch[key][k],atol=1.e-10,rtol=0)
        finite=np.isfinite(old[key])
        score_error=max(score_error,float(np.abs(old[key][finite]-arch[key][k][finite]).max(initial=0.)))
    print('R170 rebuilt R169 bank and winners exactly',model.source,score_error,flush=True)
    return dict(source=model.source,bank_and_winners_bit_exact=True,max_score_error=score_error,
                shared_points_bit_exact=True,query_label_read=False)


def predict_case(model,entry,setting):
    row=entry['row']
    exp=entry['baseline_index']
    assert sha(row['pointpath'])==row['point_sha256'] and sha(row['imagepath'])==row['image_sha256']
    ev,original,_,old=original_evidence(model,row,setting)
    registered_winner,registered_box=old_baseline(model.source)[exp]
    assert (int(ev.probability.argmax()),ev.old_winner)==tuple(registered_winner)
    np.testing.assert_array_equal(ev.old_pca,registered_box)
    np.testing.assert_array_equal(ev.edges,edges_of(entry['parsed']['graph']))
    np.testing.assert_array_equal(ev.role_spans,[e['span'] for e in entry['parsed']['graph']['entities']])
    role=entry['role']
    assert int((old['bank'][role]==-1).sum())==1
    native=ev.boxes
    extra=old['novel'][np.isfinite(old['novel']).all(1)]
    complete=np.concatenate((np.c_[native,np.zeros(len(native))],extra))
    IDs=old['bank'][role,old['bank'][role]>=0]
    points=np.unique(ev.points,axis=0)
    mask=support_matrix(complete,points,len(native))[IDs]
    common=oriented(row,old,native,IDs,lambda _:ev.points) if len(IDs) else np.zeros((0,7))
    fields=[requery(model,entry,ev,old,original,mask,points,arm) for arm in VARIANTS]
    values={k:np.stack([x[k] for x in fields]) for k in fields[0]}
    baseline=np.stack((ev.old_pca,np.full(7,np.nan)))
    values['method_boxes']=np.concatenate((baseline[None],values['boxes'].reshape(4,2,7)))
    return dict(**values,common=pad(common,BANK,np.nan),common_n=len(common),seeds=ev.seeds,
                point_sha=array_sha(ev.points),sample_sha=array_sha(ev.indices),source_reference_bank_rebuilt=True)


def predict():
    import torch
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint

    digest=lock('prediction',('select170.py','predict170.py'))
    panel=read(HERE/'INPUT_SEAL.json')
    assert sha(INPUT)==panel['registry'] and panel['recordings_overlap_with_D_selection']==0
    assert panel['sensor_overlap_with_R165_or_R168_source']==0 and not panel['FINAL_access']
    labs=read(INPUT)
    settings=read(SOURCE_SETTINGS)['sources']
    dest=WORK/'predictions'
    dest.mkdir(exist_ok=True)
    run={}
    checks=[]
    counts=Counter()
    status('predicting',physical_frames=panel['n'],direction_records=panel['direction_n'])
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as resource:
        fcntl.flock(resource,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for source in PLATFORMS:
            net=EvidenceGrounder(source)
            torch.set_num_threads(1)
            initial=fingerprint(net.base),fingerprint(net.matcher)
            with torch.inference_mode():
                checks.append(bridge(net,settings[source]))
            for item in labs:
                if item['target']==source:
                    continue
                key=f"{item['case']:02d}_{source}.npz"
                path=dest/key
                started=time.perf_counter()
                with torch.inference_mode():
                    values=predict_case(net,item,settings[source])
                npz_atomic(path,**values,arms=np.asarray(ARMS),protocol=digest)
                run[key]=sha(path)
                counts.update(rows=1,rewrite=(item['rewrite']['status']=='supported'),
                              requery_forwards=int(values['executed'].sum()))
                print('R170 prediction',key,item['rewrite']['status'],round(time.perf_counter()-started,3),flush=True)
            assert initial==(fingerprint(net.base),fingerprint(net.matcher))
            del net
            gc.collect()
            torch.cuda.empty_cache()
    assert counts['rows']==panel['direction_n']==len(run)
    atomic_json(HERE/'PREDICTION_SEAL.json',dict(n=panel['direction_n'],physical_frames=panel['n'],
                input_seal=sha(HERE/'INPUT_SEAL.json'),prediction_lock=digest,files=run,counts=dict(counts),
                r159_r158_pca_bit_exact=True,source_fingerprints_unchanged=True,
                reconstruction_bridge=checks,
                target_identity_labels_read=False,FINAL_access=False))


if __name__=='__main__':
    predict()
