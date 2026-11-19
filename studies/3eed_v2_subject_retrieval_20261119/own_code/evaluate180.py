"""Seal six-direction subject retrieval before comparing exact raw-box outcomes."""

import argparse
import secrets
from time import perf_counter

import numpy as np
import torch

from retrieval180 import ARMS, HERE, NATIVE, RetrievalGrounder, TRAINING
from source180 import calibrated, implementation, protocol
from scope import ROOT, PLATFORMS, evaluation_rows, file_sha, read, save
from features import observation
from train import metrics, prepare_device
from evaluate import (DIRECTIONS, PRIVATE as PRIOR_PREDICTIONS, fingerprint,
                      old_boxes, to_sensor, score_boxes, many_iou, sensor_truth, save_csv, recording_intervals)


OUT = HERE/'predictions_PRIVATE'
K_MAX = 16


def own_pin():
    hashes = implementation()
    hashes['evaluate180.py'] = file_sha(HERE/'evaluate180.py')
    return hashes


def archive_lock():
    pin=read(PRIOR_PREDICTIONS/'SEALED.json')
    lock=read(PRIOR_PREDICTIONS/'LOCK.json')
    assert file_sha(PRIOR_PREDICTIONS/'SEALED.json')==protocol()['archived_candidate_seal_sha256']
    assert pin['lock_sha256']==file_sha(PRIOR_PREDICTIONS/'LOCK.json')
    assert lock['implementation']==fingerprint() and not pin['FINAL_access'] and not pin['target_labels_read']
    return pin


def previous(rows, source, target, pin):
    path=PRIOR_PREDICTIONS/f'{source}_to_{target}.npz'
    assert file_sha(path)==pin['files'][path.name]['sha256']
    ids, baseline=old_boxes(rows,source,target)
    assert len(ids)==512 and baseline.shape==(512,7)
    with np.load(path,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['ordinals'],ids)
        np.testing.assert_array_equal(data['picked'][:,0],baseline)
        return ids,baseline,data['pool'][:,:NATIVE],data['v1_slots']


def predict(smoke=False):
    chosen=calibrated()
    config=protocol()
    assert tuple(config['arms'])==ARMS
    assert file_sha(ROOT/'crossplatform/round159/RECORDING_INPUTS_PRIVATE.json')==config['registry_sha256']
    earlier=archive_lock()
    rows=evaluation_rows()
    prepare_device()
    root=HERE/'smoke_PRIVATE' if smoke else OUT
    root.mkdir(mode=0o700)
    lock=dict(implementation=own_pin(),calibration_sha256=file_sha(TRAINING/'SELECTION.json'),
              raw_pool_seal=config['archived_candidate_seal_sha256'],FINAL_access=False,target_labels_read=False)
    save(root/'LOCK.json',lock)
    results,seen={},set()
    start=perf_counter()
    for source in PLATFORMS:
        method=RetrievalGrounder(source)
        k=chosen['sources'][source]['selected_k']
        for target in PLATFORMS:
            if source==target:
                continue
            ids,baseline,pool,v1_slots=previous(rows,source,target,earlier)
            n=2 if smoke else len(ids)
            chosen_ids=np.empty((n,len(ARMS)),dtype=np.int16)
            all_slots=np.full((n,len(ARMS),K_MAX),-1,dtype=np.int16)
            reach=np.zeros((n,NATIVE),dtype=bool)
            caps=np.array([4,4,8,16,4,8,16,k,k],dtype=np.int8)
            for j,i in enumerate(ids[:n]):
                row=rows[int(i)]
                key=(row['pointpath'],row['point_sha256'])
                if key not in seen:
                    assert file_sha(key[0])==key[1]
                    seen.add(key)
                models,trials,bound,view=method.compare(observation(row))
                boxes=to_sensor(models,target,view)
                np.testing.assert_array_equal(boxes,pool[j])
                base_ids,base_winner=trials['v1']
                np.testing.assert_array_equal(boxes[base_winner],baseline[j])
                np.testing.assert_array_equal(boxes[base_ids],v1_slots[j])
                trials['native_source_K']=trials[f'native{k}']
                trials['trained_source_K']=trials[f'trained{k}']
                for arm,name in enumerate(ARMS):
                    slots,win=trials[name]
                    assert len(slots)==caps[arm] and base_winner in slots and win in slots
                    all_slots[j,arm,:len(slots)]=slots
                    chosen_ids[j,arm]=win
                reach[j]=bound
                if (j+1)%64==0 or j+1==n:
                    print(source,target,j+1,'/',n,round(perf_counter()-start,1),flush=True)
            path=root/f'{source}_to_{target}.npz'
            np.savez_compressed(path,ordinals=ids[:n],chosen=chosen_ids,slate=all_slots,can_win=reach,caps=caps)
            results[path.name]=dict(n=n,sha256=file_sha(path))
        del method
        torch.cuda.empty_cache()
    n=sum(info['n'] for info in results.values())
    assert n==(12 if smoke else 3072) and own_pin()==lock['implementation']
    save(root/'SEALED.json',dict(n=n,files=results,lock_sha256=file_sha(root/'LOCK.json'),
         baseline_matches=n,raw_candidate_pools_match=n,incumbent_in_every_shortlist=n*len(ARMS),
         elapsed_seconds=perf_counter()-start,target_labels_read=False,target_updates=0,FINAL_access=False))


def verified_directions(rows, earlier, pin, selection):
    result={}
    for source,target in DIRECTIONS:
        k=selection['sources'][source]['selected_k']
        path=OUT/f'{source}_to_{target}.npz'
        assert file_sha(path)==pin['files'][path.name]['sha256']
        index,baseline,pool,v1_slots=previous(rows,source,target,earlier)
        with np.load(path,allow_pickle=False) as data:
            block={name:data[name] for name in data.files}
        np.testing.assert_array_equal(block['ordinals'],index)
        np.testing.assert_array_equal(block['caps'],[4,4,8,16,4,8,16,k,k])
        slate,win=block['slate'],block['chosen']
        np.testing.assert_array_equal(pool[np.arange(len(pool)),win[:,0]],baseline)
        np.testing.assert_array_equal(pool[np.arange(len(pool))[:,None],slate[:,0,:4]],v1_slots)
        assert np.all((win>=0)&(win<NATIVE)) and np.all(np.any(slate==win[...,None],axis=2))
        assert np.all(np.any(slate==win[:,:1,None],axis=2))
        for j,cap in enumerate(block['caps']):
            assert np.all((slate[:,j,:cap]>=0)&(slate[:,j,:cap]<NATIVE))
            assert (slate[:,j,cap:]==-1).all()
            assert (np.diff(np.sort(slate[:,j,:cap],axis=1),axis=1)>0).all()
        for four,eight,sixteen in ((1,2,3),(4,5,6)):
            assert (slate[:,four,:4,None]==slate[:,eight,None,:8]).any(-1).all()
            assert (slate[:,eight,:8,None]==slate[:,sixteen,None,:16]).any(-1).all()
        result[source,target]=dict(block,pool=pool)
    return result


def score_direction(target, rows, block):
    records, err, independent=[],0.,0
    for j, i in enumerate(block['ordinals']):
        row=rows[int(i)]
        gt=sensor_truth(target,row['metapath'],row['caption'],row['suffix'])
        pool=block['pool'][j]
        all_ious=many_iou(pool,gt)
        slate,win,bound=block['slate'][j],block['chosen'][j],block['can_win'][j]
        ids=[s[:k] for s,k in zip(slate,block['caps'])]
        top=[s[all_ious[s].argmax()] for s in ids]
        threshold_checks=np.unique(np.r_[win,top,all_ious.argmax()])
        audited,error=score_boxes(pool[threshold_checks],gt)
        np.testing.assert_allclose(all_ious[threshold_checks],audited,atol=1e-10,rtol=0)
        err=max(err,error)
        independent+=len(threshold_checks)
        records.append(dict(ious=all_ious[win],covered=all_ious[top],
             reachable=np.array([all_ious[s[bound[s]]].max(initial=0.) for s in ids]),
             oracle=all_ious.max(),switched=win!=win[0],outside=~np.isin(win,ids[0]),
             recordings=row['scene_cluster']))
    return {name:np.stack([row[name] for row in records]) for name in records[0]},err,independent


def diagnostics(block, arm, threshold):
    hit=block['oracle']>threshold
    original=block['covered'][:,0]>threshold
    miss=hit & ~original
    found=block['covered'][:,arm]>threshold
    reachable=block['reachable'][:,arm]>threshold
    final=block['ious'][:,arm]>threshold
    before=block['ious'][:,0]>threshold
    outside=block['outside'][:,arm]
    rescued=final & ~before
    harmed=before & ~final
    assert np.all(~found | hit) and np.all(~final | found)
    return dict(pool_hits=int(hit.sum()),shortlist_hits=int(found.sum()),
        recall_pct=float(100*found.mean()),conditional_recall_pct=float(100*found.sum()/hit.sum()),
        four_slot_misses=int(miss.sum()),four_slot_misses_recalled=int((miss & found).sum()),
        four_slot_misses_reachable=int((miss & reachable).sum()),
        four_slot_misses_rescued=int((miss & rescued).sum()),
        other_outside_rescues=int((rescued & outside & ~miss).sum()),
        within_original_rescues=int((rescued & ~outside).sum()),
        total_rescues=int(rescued.sum()),total_harms=int(harmed.sum()),
        harms_by_outside=int((harmed & outside).sum()),net=int(rescued.sum()-harmed.sum()))


def score():
    selection=calibrated()
    pin,lock=read(OUT/'SEALED.json'),read(OUT/'LOCK.json')
    assert pin['n']==3072 and pin['lock_sha256']==file_sha(OUT/'LOCK.json')
    assert lock['implementation']==own_pin() and lock['calibration_sha256']==file_sha(TRAINING/'SELECTION.json')
    assert not pin['FINAL_access'] and not pin['target_labels_read'] and not pin['target_updates']
    earlier=archive_lock()
    rows=evaluation_rows()
    data=verified_directions(rows,earlier,pin,selection)
    scored, maximum, unique_checks={},0.,0
    for (source,target), block in data.items():
        measured,error,checks=score_direction(target,rows,block)
        scored[f'{source}_to_{target}']=measured
        maximum=max(maximum,error)
        unique_checks+=checks
        print(source,target,len(block['ordinals']),'scored',flush=True)
    overall={field:np.concatenate([b[field] for b in scored.values()],axis=0)
             for field in ('oracle','ious','covered','reachable','switched','outside')}
    scores,transitions,anonymous=[],[],[]
    for direction,b in (*scored.items(),('six_equal',overall)):
        for a,name in enumerate(ARMS):
            n=len(b['ious'])
            iou=b['ious'][:,a]
            scores.append(dict(direction=direction,arm=name,n=n,**metrics(iou),
                 Recall25=float(100*(b['covered'][:,a]>.25).mean()),
                 Recall50=float(100*(b['covered'][:,a]>.5).mean()),
                 switched=int(b['switched'][:,a].sum()),
                 iou_better=int((iou-b['ious'][:,0]>1e-12).sum()),
                 iou_worse=int((b['ious'][:,0]-iou>1e-12).sum())))
            for t in (.25,.5):
                transitions.append(dict(direction=direction,arm=name,threshold=t,n=n,**diagnostics(b,a,t)))
        if direction=='six_equal':
            continue
        source,target=direction.split('_to_')
        for index,row in enumerate(b['ious']):
            anonymous.append(dict(source=source,target=target,A256_iou=float(b['oracle'][index]),
                **{f'iou_{a}':float(row[j]) for j,a in enumerate(ARMS)},
                **{f'recall_iou_{a}':float(b['covered'][index,j]) for j,a in enumerate(ARMS)},
                **{f'switch_{a}':int(b['switched'][index,j]) for j,a in enumerate(ARMS)},
                **{f'outside_{a}':int(b['outside'][index,j]) for j,a in enumerate(ARMS)}))
    out=HERE/'results'
    out.mkdir()
    save_csv(out/'direction.csv',scores)
    save_csv(out/'transitions.csv',transitions)
    secrets.SystemRandom().shuffle(anonymous)
    save_csv(out/'anonymous_rows.csv',anonymous)
    save(out/'source_selection.json',selection)
    save(out/'source_training.json',read(TRAINING/'FITTED.json'))
    bounds=recording_intervals(scored,ARMS)
    bounds['contrast']='method minus v1 in percentage points'
    bounds['intervals']={a:[[-hi,-lo] for lo,hi in b] for a,b in bounds['intervals'].items()}
    save(out/'recording_intervals.json',bounds)
    scene_counts={t:len(np.unique(scored[f'{s}_to_{t}']['recordings'])) for s,t in DIRECTIONS}
    save(out/'audit.json',dict(n=len(anonymous),fixed_raw_anchors=pin['raw_candidate_pools_match'],
         exact_v1=pin['baseline_matches'],incumbent_retained=pin['incumbent_in_every_shortlist'],
         endpoint_count=len(anonymous)*len(ARMS),independent_unique_checks=unique_checks,
         exact_oriented_iou_max_abs_error=maximum,threshold_disagreements=0,recordings=scene_counts,
         source_selection_sha256=lock['calibration_sha256'],prediction_seal_sha256=file_sha(OUT/'SEALED.json'),
         target_labels_read_only_after_seal=True,target_updates=0,FINAL_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('smoke','predict','score'))
    name=parser.parse_args().phase
    score() if name=='score' else predict(smoke=name=='smoke')
