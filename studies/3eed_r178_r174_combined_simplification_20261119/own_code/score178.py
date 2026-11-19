"""Score sealed combinations against their parent and the complete R174."""
from math import fsum

import numpy as np

from scope178 import (ARMS,N,ORDER,ORIGINAL,PLATFORMS,PRIOR,PRIVATE,P177,PROTOCOL,
                      R177,RESULTS,json_atomic,read,sha,state)
from model178 import CONFIGS
from predict178 import OUTPUT,verify
from score174 import write_csv
from score177 import COHORTS, EPS, METRICS, cell, masks
from scope177 import ARMS as OLD_ARMS
from reference178 import reference_panel


def read_direction(route,old_scoring):
    original = ORIGINAL/route
    current = OUTPUT/route
    baseline = read(original/'SEAL.json')
    stamp = read(current/'SEAL.json')
    assert sha(original/'SEAL.json')==stamp['original']
    p = PRIOR/'score'/f'{route}.npz'
    assert sha(p)==old_scoring['score_arrays'][route]
    with np.load(p,allow_pickle=False) as a:
        source = {k:a[k] for k in ('ordinal','slots','ious','flags','recording')}
        assert str(a['predictions'])==stamp['original']
    observed = []
    for name,digest in sorted(stamp['files'].items()):
        p,q = current/name,original/name
        assert sha(p)==digest and sha(q)==baseline['files'][name]
        with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
            assert str(x['original'])==baseline['files'][name]
            idx,choices = x['ordinals'],x['choices']
            np.testing.assert_array_equal(idx,y['ordinals'])
            np.testing.assert_array_equal(y['choices'][:,-1],choices[:,0])
            np.testing.assert_array_equal(y['direct_left_right'],source['flags'][idx,1])
            np.testing.assert_array_equal(y['reference_mention'],source['flags'][idx,0])
            k = np.arange(len(idx))[:,None]
            sb = y['subject_slots'][k,choices[...,0]]
            rb = y['reference_slots'][k,choices[...,1]]
            iou = source['slots'][idx[:,None],choices[...,0]]
            np.testing.assert_array_equal(iou[:,0],source['ious'][idx,-1])
        observed.append(dict(ordinal=idx,iou=iou,subject_boxes=sb,reference_boxes=rb,
                             same_box=(sb==rb).all(-1)))
    data = {k:np.concatenate([p[k] for p in observed]) for k in observed[0]}
    np.testing.assert_array_equal(data['ordinal'],source['ordinal'])
    data.update(flags=source['flags'],recording=source['recording'])
    earlier_score = P177/f'scored_{route}.npz'
    old_audit = read(R177/'results/independent_audit.json')
    assert sha(earlier_score)==old_audit['score_arrays'][route]
    with np.load(earlier_score,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['ordinal'],data['ordinal'])
        np.testing.assert_array_equal(a['iou'][:,OLD_ARMS.index('fixed_internal_boxes')],data['iou'][:,1])
    return data


def pair_counts(iou,baseline):
    d = iou-baseline
    return dict(iou_up=int((d>EPS).sum()),iou_down=int((d<-EPS).sum()),
                iou_tie=int((abs(d)<=EPS).sum()),
                acc25_gain=int(((iou>.25)&(baseline<=.25)).sum()),
                acc25_loss=int(((iou<=.25)&(baseline>.25)).sum()),
                acc50_gain=int(((iou>.5)&(baseline<=.5)).sum()),
                acc50_loss=int(((iou<=.5)&(baseline>.5)).sum()))


def scalar_tables(data):
    rows,paired,timing = [],[],[]
    max_error = 0.
    for route,items in data.items():
        iou = items['iou']
        values = 100*cell(iou)
        for subset,mask in masks(items).items():
            q = values[mask].mean(0)
            n = int(mask.sum())
            for j,arm in enumerate(ARMS):
                vector = iou[mask,j]
                direct = [100*sum(vector>.25)/n,100*sum(vector>.5)/n,100*fsum(vector)/n]
                max_error = max(max_error,float(abs(q[j]-direct).max()))
                rows.append(dict(direction=route,subset=subset,variant=arm,n=n,
                                 **dict(zip(METRICS,q[j].tolist(),strict=True))))
                declared = PROTOCOL['arms'][arm]['parent']
                for base in dict.fromkeys(('full',declared or 'full')):
                    k = ARMS.index(base)
                    subject_switch = (items['subject_boxes'][mask,j]!=items['subject_boxes'][mask,k]).any(-1)
                    reference_switch = (items['reference_boxes'][mask,j]!=items['reference_boxes'][mask,k]).any(-1)
                    paired.append(dict(direction=route,subset=subset,variant=arm,baseline=base,n=n,
                          **dict(zip(METRICS,(q[j]-q[k]).tolist(),strict=True)),
                          **pair_counts(vector,iou[mask,k]),
                          subject_box_switches=int(subject_switch.sum()),
                          reference_box_switches_with_reference=int((reference_switch&items['flags'][mask,0]).sum()),
                          exact_two_role_same_box=int((items['same_box'][mask,j]&items['flags'][mask,0]).sum())))
        delta = iou-iou[:,:1]
        public = []
        order = np.random.default_rng(1781+len(timing)).permutation(len(delta))
        for i,k in enumerate(order,1):
            public.append(dict(record=f'{i:04d}',reference_mention=int(items['flags'][k,0]),
                               direct_left_right=int(items['flags'][k,1]), full_iou=float(iou[k,0]),
                               **{a:float(delta[k,j]) if delta[k,j] else 0 for j,a in enumerate(ARMS) if j}))
        write_csv(RESULTS/f'paired_{route}.csv',public)
        prof = read(OUTPUT/route/'TIMING.json')
        medians = {arm:float(np.median(prof['times_ms'][arm])) for arm in ARMS}
        for arm in ARMS:
            op = CONFIGS[arm]
            timing.append(dict(direction=route,variant=arm,batch=prof['batch'],device=prof['device'],
                               decoder_ms=medians[arm],relative_to_full=medians[arm]/medians['full'],
                               rounds=op.rounds,reference_slots=op.reference_slots,
                               slot_updates=op.rounds*(4+op.reference_slots)))
    assert max_error<1e-10
    for subset in COHORTS:
        for arm in ARMS:
            group = [r for r in rows if r['subset']==subset and r['variant']==arm]
            assert len(group)==6
            rows.append(dict(direction='six_equal',subset=subset,variant=arm,n=sum(r['n'] for r in group),
                             **{m:float(np.mean([r[m] for r in group])) for m in METRICS}))
            parent = PROTOCOL['arms'][arm]['parent']
            for base in dict.fromkeys(('full',parent or 'full')):
                matched = [r for r in paired if r['subset']==subset and r['variant']==arm and r['baseline']==base]
                assert len(matched)==6
                counts = {k:sum(r[k] for r in matched) for k in matched[0]
                          if k not in ('direction','subset','variant','baseline',*METRICS)}
                paired.append(dict(direction='six_equal',subset=subset,variant=arm,baseline=base,
                                   **{m:float(np.mean([r[m] for r in matched])) for m in METRICS},**counts))
    return rows,paired,timing,max_error


def recording_intervals(data):
    rng = np.random.default_rng(178)
    repeats = 4000
    result = {}
    for subset in ('all','direct_left_right'):
        estimates,recordings = [],{}
        for target in PLATFORMS:
            birds = [data[f'{s}_to_{target}'] for s in PLATFORMS if s!=target]
            mask = masks(birds[0])[subset]
            np.testing.assert_array_equal(birds[0]['recording'],birds[1]['recording'])
            np.testing.assert_array_equal(mask,masks(birds[1])[subset])
            rec,ix = np.unique(birds[0]['recording'][mask],return_inverse=True)
            n = len(rec)
            recordings[target] = n
            weights = rng.multinomial(n,np.full(n,1/n),size=repeats)
            den = weights@np.bincount(ix)
            for d in birds:
                scores = cell(d['iou'][mask])
                accum = np.zeros((n,len(ARMS),len(METRICS)))
                np.add.at(accum,ix,scores-scores[:,:1])
                estimates.append(100*np.einsum('dn,nak->dak',weights,accum)/den[:,None,None])
        result[subset] = dict(recordings=recordings,repeats=repeats,
                              deltas_pp={a:dict(zip(METRICS,z.tolist(),strict=True)) for a,z in zip(
                                  ARMS,np.quantile(np.mean(estimates,0),(.025,.975),axis=0).transpose(1,2,0),
                                  strict=True)})
    return result


def main():
    pred = verify()
    state('post_seal_scoring','Paired scoring with the independently audited original four subject-slot IoUs.')
    old_path = PRIOR/'SCORING_SEAL.json'
    old = read(old_path)
    earlier_audit = read(R177/'results/independent_audit.json')
    geom_path = PRIOR.parent/'results/independent_audit.json'
    geom = read(geom_path)
    assert old['n']==earlier_audit['n']==N and not (old['FINAL_access'] or earlier_audit['FINAL_access'])
    assert earlier_audit['previous_four_slot_geometry_audit']==sha(geom_path)
    assert geom['n']==N and not geom['FINAL_access'] and geom['scoring']==sha(old_path)
    assert all(x['all_four_proposals_audited'] for x in geom['directions'].values())
    data = {}
    for s,t in ORDER:
        d = f'{s}_to_{t}'
        data[d] = read_direction(d,old)
        print('R178_SCORED',d,len(data[d]['iou']),flush=True)
    assert sum(len(r['iou']) for r in data.values())==pred['n']==N
    rows,deltas,timing,err = scalar_tables(data)
    for name,table in (('endpoints',rows),('paired_deltas',deltas),('decoder_timing',timing)):
        write_csv(RESULTS/f'{name}.csv',table)
    json_atomic(RESULTS/'recording_intervals.json',recording_intervals(data))
    review = reference_panel()
    json_atomic(RESULTS/'independent_audit.json',dict(n=N,directions=6,arms=list(ARMS),
                    full_replay_disagreements=0,anchor_only_ablation_disagreements=0,
                    original_score_seal=sha(old_path),original_geometric_audit=sha(geom_path),
                    prediction=sha(OUTPUT/'PREDICTION_SEAL.json'),
                    reviewed_reference_identity=review,
                    independent_scalar_max_error_pp=err,source_training=0,target_updates=0,FINAL_access=False))
    proof = dict(n=N,results={p.name:sha(p) for p in RESULTS.iterdir()},
                 predictions=sha(OUTPUT/'PREDICTION_SEAL.json'),FINAL_access=False)
    json_atomic(PRIVATE/'SCORING_SEAL.json',proof)
    print('R178_SCORE_SEALED',N,flush=True)


if __name__=='__main__':
    main()
