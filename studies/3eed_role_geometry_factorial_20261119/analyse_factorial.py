"""Compute paired factorial contrasts and bounded reference-error attribution."""
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from multiprocessing import get_context
import secrets

import numpy as np

from study import (ARMS, BASE, CELLS, HERE, NEW, OLD_PRED, OLD_PRIVATE, ORDER, PREDICTIONS,
                   PRIVATE, RESULTS, ROOT, THRESHOLDS, check_lock, json_atomic, npz_atomic,
                   ordinary_rows, read, recording, sha, state)
from batches import columns
from predict import NAMES as OLD_NAMES
from reference_truth import identity_context, verified_refs
from score import labels
from batch_yaw_iou import many_iou

METRICS = ('Acc25','Acc50','mIoU')
CONTRASTS = {
    'role_at_geometry_off': (0,-1,1,0,0),
    'role_at_geometry_on': (0,0,0,-1,1),
    'geometry_at_role_off': (0,-1,0,1,0),
    'geometry_at_role_on': (0,0,-1,0,1),
    'role_main': (0,-.5,.5,-.5,.5),
    'geometry_main': (0,-.5,-.5,.5,.5),
    'interaction': (0,1,-1,-1,1),
    'joint_minus_off_off': (0,-1,0,0,1),
    'joint_minus_R158_PCA': (-1,0,0,0,1),
    'role_only_minus_R158_PCA': (-1,0,1,0,0),
    'geometry_only_minus_R158_PCA': (-1,0,0,1,0),
    'off_off_minus_R158_PCA': (-1,1,0,0,0),
}


def verify_seal():
    check_lock()
    seal = read(PREDICTIONS/'PREDICTION_SEAL.json')
    file = PRIVATE/'EVALUATION_LOCK.json'
    contract = read(file)
    assert seal['contract']==sha(file) and seal['n']==18620 and len(seal['directions'])==6
    assert not seal['FINAL_access'] and not seal['target_truth_used'] and contract['target_updates']==0
    assert sha(OLD_PRED/'PREDICTION_SEAL.json')==contract['prior_prediction']
    for k,h in contract['code'].items():
        assert sha(HERE/k)==h
    for k,h in contract['identity_artifacts'].items():
        assert sha(ROOT/k)==h
    for d,(digest,n) in seal['directions'].items():
        p = PREDICTIONS/d/'SEAL.json'
        info = read(p)
        assert not info['target_truth_used'] and not info['FINAL_access']
        assert sha(p)==digest and info['n']==info['joint_replay_matched_rows']==n
        assert all(sha(p.parent/k)==h for k,h in info['files'].items())
    return seal


def blocks(direction):
    dst,original = PREDICTIONS/direction,OLD_PRED/direction
    p,q = read(dst/'SEAL.json'),read(original/'SEAL.json')
    assert p['archive']==sha(original/'SEAL.json')
    for name in sorted(p['files']):
        with np.load(dst/name,allow_pickle=False) as f,np.load(original/name,allow_pickle=False) as g:
            x = dict(f)
            needed=('ordinals','boxes','refs','choices','is_relation','left_right','seeds','subject_slots')
            z = {k:g[k] for k in needed}
        assert str(x['archived'])==q['files'][name]
        np.testing.assert_array_equal(x['ordinals'],z['ordinals'])
        yield x,z


def merged_boxes(x,z):
    return np.stack((z['boxes'][:,0], z['boxes'][:,OLD_NAMES.index('without_feedback')],
                     x['mixed_boxes'][:,0,0], x['mixed_boxes'][:,1,0], z['boxes'][:,OLD_NAMES.index('joint')]),axis=1)


def choice_with_mask(score, valid):
    restricted = np.where(valid[None,:],score,-np.inf)
    flat = int(restricted.argmax())
    return flat//len(valid)


def score_direction(direction):
    source,target = direction
    name = f'{source}_to_{target}'
    known_panel = verified_refs()
    obs = ordinary_rows(target)
    joint_id = OLD_NAMES.index('joint')-1
    bank = PRIVATE/'score'/f'{name}.npz'
    rows = []
    for x,z in blocks(name):
        ends = merged_boxes(x,z)
        route = z['is_relation']
        all_choices = z['choices'][:,joint_id,:2]
        score, u = x['joint_score'], x['subject_unary']
        top_unary, conditioned = columns(score,u,route,all_choices)
        for j,k in enumerate(x['ordinals']):
            k = int(k)
            row = obs[k]
            truth,_ = labels(row)
            value = many_iou(ends[j],truth)
            slots = many_iou(z['subject_slots'][j,joint_id],truth)
            np.testing.assert_allclose(value[-1],slots[all_choices[j,0]],rtol=0.,atol=2e-12)
            head = slots[top_unary[j]]
            choices = conditioned[j]
            alt_iou = slots[choices]
            missed = any(value[-1]<=cut<slots.max() for cut in THRESHOLDS)
            panel = (target,k) in known_panel
            owners = np.full(4,-2,dtype=np.int16)
            gold,confidence,panel_matched = -1,0,False
            if route[j] and (missed or panel):
                owners,gold,confidence,panel_matched = identity_context(
                    source,target,row,k,z['seeds'][j],x['replay_refs'][j],include_proxy=missed)
            native_owner = owners[all_choices[j,1]]
            known_other = (owners>=0)&(native_owner>=0)&(owners!=native_owner)
            known_same = (owners>=0)&(owners==native_owner)
            known_same[all_choices[j,1]] = False
            has_gold = (owners==gold)&(gold>=0)
            fixed = float(slots[choice_with_mask(score[j],has_gold)]) if has_gold.any() else -1.
            ref_status = 0 if gold<0 or native_owner<0 else 1 if gold==native_owner else -1
            ref_result = (k, slots.max(), head, alt_iou.max(),
                          alt_iou[known_other].max() if known_other.any() else -1.,
                          alt_iou[known_same].max() if known_same.any() else -1.,
                          int(confidence), int(ref_status), fixed, int(panel), int(panel_matched),
                          int(native_owner>=0), int((conditioned[j]!=all_choices[j,0]).any()),
                          int(route[j]), int(z['left_right'][j]))
            rows.append((value,ref_result,slots,alt_iou,has_gold))
    v,d,s,c,g = zip(*rows,strict=True)
    value, diag = np.stack(v),np.asarray(d)
    assert np.array_equal(diag[:,0],np.arange(len(obs)))
    old = OLD_PRIVATE/'score'/f'{name}.npz'
    assert sha(old)==read(OLD_PRIVATE/'SCORING_SEAL.json')['scores'][old.name]
    with np.load(old,allow_pickle=False) as earlier:
        np.testing.assert_allclose(value[:,[0,1,4]],earlier['values'][:,[0,2,1]],rtol=0.,atol=2e-12)
        np.testing.assert_allclose(diag[:,1],earlier['slot_ceiling'][:,0],rtol=0.,atol=2e-12)
        np.testing.assert_array_equal(diag[:,13:],earlier['flags'][:,:2])
    npz_atomic(bank, values=value, diag=diag, slots=np.stack(s), conditional=np.stack(c), gold_columns=np.stack(g))
    print('SCORED', name,len(value),flush=True)
    return name


def save_csv(file, rows):
    with (RESULTS/file).open('w',newline='',encoding='utf-8') as f:
        out = csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        out.writeheader()
        out.writerows(rows)


def metric(v):
    return np.stack((v>.25,v>.50,v),axis=-1).astype(float)


def cohorts(d):
    return {'all':np.ones(len(d),bool),'has_reference':d[:,13]>0,'no_reference':d[:,13]==0,
            'left_right':d[:,14]>0}


def sensitivity(d,iou,thr):
    missed = (iou<=thr)&(d[:,1]>thr)
    alt = missed&(d[:,3]>thr)
    keep = iou>thr
    result = dict(rows=len(iou), reached=int((d[:,1]>thr).sum()), missed=int(missed.sum()),
                  missed_with_reference=int((missed&(d[:,13]>0)).sum()),
                  missed_with_no_verified_or_proxy_reference=int((missed&(d[:,6]==0)).sum()),
                  switch_column_rescues=int(alt.sum()),
                  switch_to_known_other_owner_rescues=int((missed&(d[:,4]>thr)).sum()),
                  switch_to_same_known_owner_rescues=int((missed&(d[:,5]>thr)).sum()),
                  missed_without_existing_column_rescue=int((missed&~alt).sum()),
                  without_pair_readout_rescues=int(((~keep)&(d[:,2]>thr)).sum()),
                  without_pair_readout_harms=int((keep&(d[:,2]<=thr)).sum()),
                  missed_after_removing_pair_but_switch_works=int((alt&(d[:,2]<=thr)).sum()),
                  subject_unary_mIoU=float(d[:,2].mean()*100))
    for key,val in (('visual',2),('proxy',1)):
        chosen = missed&(d[:,6]==val)
        wrong = chosen&(d[:,7]==-1)
        unknown = chosen&(d[:,7]==0)
        correct = chosen&(d[:,7]==1)
        result.update({
            key+'_known_missed':int(chosen.sum()),
            key+'_wrong_ref_missed':int(wrong.sum()),
            key+'_unknown_owner_missed':int(unknown.sum()),
            key+'_correct_ref_missed':int(correct.sum()),
            key+'_wrong_ref_correct_slot_available':int((wrong&(d[:,8]>=0)).sum()),
            key+'_wrong_ref_correct_slot_rescued':int((wrong&(d[:,8]>thr)).sum()),
            key+'_unknown_owner_correct_slot_rescued':int((unknown&(d[:,8]>thr)).sum()),
            key+'_same_truth_identity_correct_slot_rescued':int((correct&(d[:,8]>thr)).sum()),
        })
    return result


def summarize():
    catalog = {}
    for s,t in ORDER:
        name = f'{s}_to_{t}'
        with np.load(PRIVATE/'score'/f'{name}.npz',allow_pickle=False) as f:
            catalog[name] = dict(f)
    endpoints, effects, sensitivity_rows, recording_rows = [], [], [], []
    for scope in (*catalog,'six_equal','pooled'):
        groups = list(catalog) if scope in ('six_equal','pooled') else [scope]
        for bucket in ('all','has_reference','no_reference','left_right'):
            parts, ds = [], []
            for name in groups:
                z = catalog[name]
                mask = cohorts(z['diag'])[bucket]
                if mask.any():
                    parts.append(metric(z['values'][mask]))
                    ds.append(z['diag'][mask])
            all_metrics = np.concatenate(parts)
            d = np.concatenate(ds)
            result = 100*(np.mean([a.mean(0) for a in parts],axis=0) if scope=='six_equal' else all_metrics.mean(0))
            prop = dict(scope=scope,cohort=bucket,n=len(d))
            for name,r in zip(ARMS,result,strict=True):
                endpoints.append(dict(**prop,arm=name,**dict(zip(METRICS,r,strict=True))))
            for key,coef in CONTRASTS.items():
                per_sample = np.einsum('a,nam->nm',coef,all_metrics)
                output = dict(**prop,contrast=key)
                for i,m in enumerate(METRICS):
                    delta = per_sample[:,i]
                    output.update({m+'_pp':float(np.asarray(coef)@result[:,i]),
                                   m+'_up':int((delta>1e-10).sum()),m+'_down':int((delta< -1e-10).sum()),
                                   m+'_tie':int((np.abs(delta)<=1e-10).sum())})
                effects.append(output)
            if bucket in ('all','has_reference','left_right') and scope!='six_equal':
                for n,thr in enumerate(THRESHOLDS):
                    # Strict threshold bits are exact. Sensitivity counts use IoUs.
                    if scope=='pooled':
                        vv = np.concatenate([catalog[k]['values'][cohorts(catalog[k]['diag'])[bucket],-1] for k in groups])
                    else:
                        vv = catalog[scope]['values'][cohorts(catalog[scope]['diag'])[bucket],-1]
                    sensitivity_rows.append(dict(scope=scope,cohort=bucket,threshold=thr,**sensitivity(d,vv,thr)))
    for direction,z in catalog.items():
        t = direction.split('_to_')[1]
        rec = np.asarray([recording(r) for r in ordinary_rows(t)])
        arr = metric(z['values'])
        for key in ('role_main','geometry_main','interaction','joint_minus_R158_PCA'):
            per_recording = np.stack([np.einsum('a,am->m',CONTRASTS[key],arr[rec==g].mean(0))*100
                                      for g in np.unique(rec)])
            row = dict(direction=direction,contrast=key,recordings=len(per_recording))
            for m,c in zip(METRICS,per_recording.T,strict=True):
                row.update({m+'_min_pp':float(c.min()),m+'_max_pp':float(c.max()),
                            m+'_positive':int((c>1e-10).sum()),m+'_negative':int((c< -1e-10).sum())})
            recording_rows.append(row)
        # Retain every endpoint and negative comparison with randomized row ordering.
        rng_file = PRIVATE/'scalar_order_private.bin'
        if not rng_file.exists():
            rng_file.write_bytes(secrets.token_bytes(32))
        rng = rng_file.read_bytes()
        items = sorted(range(len(arr)),key=lambda k:hashlib.blake2b(f'{direction}:{k}'.encode(),key=rng,digest_size=12).digest())
        public = []
        for k in items:
            d = z['diag'][k]
            public.append(dict(parsed_reference=int(d[13]),direct_left_right=int(d[14]),
                               **dict(zip(ARMS,z['values'][k],strict=True)),
                               best_subject_slot_iou=d[1],zero_pair_iou=d[2],best_reference_column_iou=d[3]))
        save_csv(f'rows_{direction}.csv',public)
    panels = np.concatenate([z['diag'] for z in catalog.values()])
    summary = dict(panel_rows=int((panels[:,9]>0).sum()),language_role_matched=int((panels[:,6]==2).sum()),
                   panel_role_unmatched_or_unavailable=int(((panels[:,9]>0)&(panels[:,6]!=2)).sum()),
                   visual_correct_reference=int(((panels[:,6]==2)&(panels[:,7]==1)).sum()),
                   visual_wrong_reference=int(((panels[:,6]==2)&(panels[:,7]==-1)).sum()),
                   visual_unknown_owner=int(((panels[:,6]==2)&(panels[:,7]==0)).sum()),
                   visual_correct_reference_covered=int(((panels[:,6]==2)&(panels[:,8]>=0)).sum()),
                   FINAL_access=False)
    json_atomic(RESULTS/'identity_coverage.json',summary)
    save_csv('endpoints.csv',endpoints)
    save_csv('factorial_contrasts.csv',effects)
    # Shard wide sensitivity tables by scope to keep the main table easily inspectable.
    save_csv('reference_diagnostics.csv',[r for r in sensitivity_rows if r['scope']=='pooled'])
    save_csv('reference_by_direction.csv',[r for r in sensitivity_rows if r['scope']!='pooled'])
    save_csv('recording_effects.csv',recording_rows)


def main():
    verify_seal()
    state('post_seal_scoring','All new predictions and score interventions sealed.')
    RESULTS.mkdir(exist_ok=True)
    (PRIVATE/'score').mkdir(exist_ok=True,mode=0o700)
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        for result in pool.map(score_direction,ORDER):
            print('SCORE_DONE',result,flush=True)
    summarize()
    json_atomic(PRIVATE/'SCORING_SEAL.json',dict(predictions=sha(PREDICTIONS/'PREDICTION_SEAL.json'),
                  private_scores={p.name:sha(p) for p in (PRIVATE/'score').glob('*.npz')},
                  public_scores={p.name:sha(p) for p in RESULTS.iterdir() if p.is_file()},FINAL_access=False))


if __name__=='__main__':
    main()
