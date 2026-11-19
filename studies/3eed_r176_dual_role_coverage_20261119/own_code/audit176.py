"""Independent sensor-frame IoU of sealed outputs and exhaustive A256 spot checks."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from scope176 import (N, ORDER, ORIGINAL, PLATFORMS, PRED_ARMS, PRIOR, PRIVATE, R165_DATA, RESULTS,
                      array_sha, camera_view, json_atomic, ordinary_rows, points_for, read, sha)
from diagnose176 import DUAL_CUTS, SUBJECT_CUTS
from retrieve176 import OUTPUT
from score176 import diagnosis
from independent_audit import canonical, independent_annotations, sensor_iou, transform_gt
from crossplatform.ground_single_frame import orient_box
from train175 import load_split


def has_pair(ious, ids, cut):
    s, r = [q[x > t] for q, x, t in zip(ids,ious,cut,strict=True)]
    return bool(len(s) and len(r) and (len(s)>1 or len(r)>1 or s[0]!=r[0]))


def audit_direction(direction):
    source, target = direction
    route = f'{source}_to_{target}'
    prior_root = ORIGINAL/route
    new_root = OUTPUT/route
    input_root = R165_DATA/route
    cold, cnew, bank = [read(p/'SEAL.json') for p in (prior_root,new_root,input_root)]
    rows = ordinary_rows(target)
    with np.load(PRIVATE/f'subject_{route}.npz',allow_pickle=False) as r:
        previous={k:r[k] for k in ('stage','pair_iou','full_iou','r158_iou')}
    with np.load(PRIVATE/f'new_subject_{route}.npz',allow_pickle=False) as r:
        assert str(r['prediction'])==sha(new_root/'SEAL.json')
        now={k:r[k] for k in ('iou','slot_ious','stage')}
    review = read(PRIVATE/'REVIEW_JOIN_PRIVATE.json')['records'][target]
    variant = {r['index']:r for r in read(PRIVATE/f'new_dual_{route}.json')}
    known = {r['index']:r for r in read(PRIVATE/f'dual_target_{route}.json') if r['status']=='verified'}
    picks = set(np.random.default_rng(176).choice(len(rows),4,replace=False)) | set(known)
    oldest, examined, error, disagreement = 0, 0, 0., 0
    for name, digest in sorted(cnew['files'].items()):
        assert sha(new_root/name)==digest and sha(prior_root/name)==cold['files'][name]
        with np.load(new_root/name,allow_pickle=False) as n, np.load(prior_root/name,allow_pickle=False) as o:
            f={k:n[k] for k in ('ordinals','pca','anchors','choice','boxes')}
            a={k:o[k] for k in ('slot_boxes','anchor_ids','choice')}
        need_all = any(int(i) in picks for i in f['ordinals'])
        if need_all:
            p=input_root/name
            assert sha(p)==bank['files'][name]
            with np.load(p,allow_pickle=False) as b:
                raw, point_sha, index_sha = b['raw'], b['points_sha'], b['indices_sha']
        np.testing.assert_array_equal(f['ordinals'],np.arange(oldest,oldest+len(f['ordinals'])))
        for j,ind in enumerate(f['ordinals']):
            i=int(ind)
            row=rows[i]
            view=camera_view(row)
            caption,native=independent_annotations(row['metapath'])[row['object_index']]
            assert canonical(caption)==row['caption']
            gt=transform_gt(native,target,view)
            current=sensor_iou(f['pca'][j,0],gt,target,view)
            np.testing.assert_array_equal(f['pca'][j,0,f['choice'][j,0]],f['boxes'][j,0])
            actual=now['slot_ious'][i]
            error=max(error,float(abs(current-actual).max()))
            disagreement+=int(((current[:,None]>SUBJECT_CUTS)!=(actual[:,None]>SUBJECT_CUTS)).sum())
            np.testing.assert_allclose(current[f['choice'][j,0]],now['iou'][i],atol=1e-7,rtol=0)
            if i not in picks:
                continue
            sampled, ix=points_for(row)
            assert array_sha(sampled)==point_sha[j].tobytes() and array_sha(ix)==index_sha[j].tobytes()
            full=np.stack([orient_box(sampled,box) for box in raw[j]])
            v=sensor_iou(full,gt,target,view)
            np.testing.assert_array_equal(v.max()>SUBJECT_CUTS,previous['stage'][i,:,0]!=0)
            np.testing.assert_array_equal(full[f['anchors'][j]],f['pca'][j])
            np.testing.assert_array_equal(full[a['anchor_ids'][j]],a['slot_boxes'][j])
            if i in known:
                ref=np.asarray(review[str(i)]['ref_box'])
                values=np.stack([v,sensor_iou(full,ref,target,view)])
                roles=np.arange(2)[:,None]
                full_ids=np.broadcast_to(np.arange(256),values.shape)
                for info,anchors,choices in (
                    (known[i],a['anchor_ids'][j],a['choice'][j,[PRED_ARMS.index('pair_only'),PRED_ARMS.index('full')]]),
                    (variant[i],f['anchors'][j],f['choice'][j,None])):
                    for c,cut in enumerate(DUAL_CUTS):
                        assert has_pair(values,full_ids,cut)==bool(info['good_roles'][c][0][2])
                        pools=values[roles,anchors]
                        assert has_pair(pools,anchors,cut)==bool(info['good_roles'][c][1][2])
                        chosen=anchors[roles,choices.T]
                        passes=(values[roles,chosen].T>cut).all(1)&(chosen[0]!=chosen[1])
                        np.testing.assert_array_equal(passes,np.array(info['stage'][c])==3)
            examined+=1
        oldest+=len(f['ordinals'])
    assert oldest==len(rows) and examined==len(picks) and error<1e-7 and disagreement==0
    result=dict(n=oldest, brute_a256_rows=examined, reviewed_dual_rows=len(known),
                max_abs_iou_error=error, threshold_disagreements=disagreement)
    print('R176_INDEPENDENT',route,result,flush=True)
    return route,result


def audit_source(source):
    source_rows=read(PRIOR/source/'monitor_PRIVATE.json')
    records=read(PRIVATE/f'dual_source_{source}.json')
    archived=read(PRIOR/source/'pair_only_TRAINING.json')['source_monitor']
    assert len(records)==archived['n_explicit_reference']
    assert sum(r['good_roles'][0][1][2] for r in records)==archived['n_both_covered']
    assert sum(r['stage'][3][0]==3 for r in records)==archived['pair25']
    data=load_split(source,'monitor',device='cpu')
    rng=np.random.default_rng(176)
    chosen=[]
    for stage in range(4):
        pool=[r for r in records if r['stage'][0][0]==stage]
        chosen.extend(rng.choice(pool,min(4,len(pool)),replace=False).tolist())
    universal=np.broadcast_to(np.arange(256),(2,256))
    for r in chosen:
        i=r['index']
        row=source_rows[i]
        points=points_for(row)[0]
        boxes=np.stack([orient_box(points,b) for b in data['prior'][i].numpy()])
        anchor=data['anchors'][i].numpy()
        np.testing.assert_array_equal(boxes[anchor],data['slot_boxes'][i].numpy())
        v=np.stack([sensor_iou(boxes,y,source,camera_view(row)) for y in data['gt'][i].numpy()])
        for c,cut in enumerate(DUAL_CUTS):
            assert has_pair(v,universal,cut)==r['good_roles'][c][0][2]
            assert has_pair(v[np.arange(2)[:,None],anchor],anchor,cut)==r['good_roles'][c][1][2]
    return source,dict(n=len(records),brute_a256_rows=len(chosen))


def main():
    diagnosis()
    proof=read(PRIVATE/'RETRIEVAL_SCORE_SEAL.json')
    assert proof['predictions']==sha(OUTPUT/'PREDICTION_SEAL.json')
    assert all(sha(PRIVATE/k)==v for k,v in proof['files'].items())
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        results=dict(pool.map(audit_direction,ORDER))
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        held=dict(pool.map(audit_source,PLATFORMS))
    assert sum(r['n'] for r in results.values())==N
    assert sum(r['n'] for r in held.values())==336
    json_atomic(RESULTS/'independent_audit.json',dict(n=N, directions=results, held_sources=held,
                diagnosis_seal=proof['diagnosis'],r175_score_seal=sha(PRIOR/'SCORING_SEAL.json'),
                score_seal=sha(PRIVATE/'RETRIEVAL_SCORE_SEAL.json'),
                FINAL_access=False))


if __name__ == '__main__':
    main()
