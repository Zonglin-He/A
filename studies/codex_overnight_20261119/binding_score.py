"""Score the source-locked role permutation variants and their paired controls."""
from collections import Counter
from pathlib import Path

import numpy as np

from binding_collect import ARMS, HERE, PLATFORMS, PRIVATE, WORK, R171, read, sha, atomic_json
from edge_score import csv_atomic, directions, iou_metrics, macro, METRICS, orders


def checked():
    from predict171 import read_npz

    pred = PRIVATE / 'predictions'
    good = read(pred / 'SEAL.json')
    assert good['lock'] == sha(pred / 'LOCK.json') and good['n'] == 168
    assert not good['target_labels_read'] and not good['protected_data_access']
    assert all(sha(pred/name) == h for name,h in good['files'].items())
    official = read(R171 / 'PREDICTION_SEAL.json')
    assert good['files'].keys() == official['files'].keys()
    assert all(sha(WORK / 'predictions' / name) == h for name,h in official['files'].items())
    return pred,good,read_npz


def independent_assignment(old, fields, weights, shuffled):
    """Enumerate the energies independently of the producer's NumPy reductions."""
    su, ru = old['subject_unary'], old['D_native_unary']
    g = old['sign']*old['native_left']
    marg = fields['offset']
    term = fields['edge']
    for k in range(2):
        np.testing.assert_allclose(term[k].sum(0), 0., rtol=0., atol=1e-12)
        np.testing.assert_allclose(term[k].sum(1), 0., rtol=0., atol=1e-12)
        np.testing.assert_array_equal(np.sort(term[k],axis=1),np.sort(shuffled[k],axis=1))
    i,j = np.indices(g.shape)
    unary = su[i]+ru[j]
    x, y = np.unravel_index(unary.argmax(),g.shape)
    hit = [tuple(old['choices'][0]),(x,y)]
    extras = [g, marg[0]+marg[1], g+marg[0]+marg[1]]
    common = g+marg[0]+marg[1]
    extras += [common + term[0]+term[1],common+shuffled[0]+shuffled[1],
               common+weights[0]*term[0]+weights[1]*term[1],
               common+weights[0]*shuffled[0]+weights[1]*shuffled[1]]
    for offset in extras:
        x = int(np.argmax(np.max(unary+offset,axis=1)))
        y = int(np.argmax(unary[x]+offset[x]))
        hit.append((x,y))
    return hit


def score_once(entry, blind, key, cohort, source, alias, prediction, old, features, weight):
    from labels166 import occupied_owners, physical_labels
    from score165 import many_iou, sensor_values
    from score170 import label, status_for

    row = entry['row']
    subject, ref = label(key,blind['subject']), label(key,blind['reference'])
    names, gt, seeds, owners = physical_labels(row,old['seeds'])
    assert len(names) == key['inventory_size'] and max(subject,ref) < len(names)
    positions = prediction['choices']
    shapes = prediction['boxes']
    actual = independent_assignment(old,features,weight,prediction['shuffled_edge'])
    np.testing.assert_array_equal(actual, positions)
    np.testing.assert_array_equal(shapes[:,0], old['subject_boxes'][positions[:,0]])
    has_ref = np.isfinite(shapes[:,1]).all(1)
    np.testing.assert_array_equal(shapes[has_ref,1], old['D_natural_boxes'][positions[has_ref,1]])
    ious = many_iou(shapes[:,0],gt[0])
    check = sensor_values(shapes[:,0],gt[0],entry['target'],int(row['suffix'] or 0))
    np.testing.assert_allclose(check,ious,rtol=0.,atol=1e-7)
    np.testing.assert_array_equal(iou_metrics(check)[1:], iou_metrics(ious)[1:])
    src = occupied_owners(old['subject_boxes'],seeds,owners,len(names))
    native = old['D_natural_boxes']
    des = occupied_owners(native,seeds,owners,len(names)) if len(native) else np.empty(0,int)
    assigned = occupied_owners(shapes,seeds,owners,len(names))
    ii,jj = features['subject_common'],features['reference_common']
    known = subject >= 0 and ref >= 0
    answers = []
    correct_s, correct_r = (src == subject) & (subject>=0), (des == ref) & (ref>=0)
    info = dict(cohort=cohort, frame=entry['case'], source=source, target=entry['target'], recording=alias,
                rewrite=entry['rewrite']['status'], both_verified=int(known),
                subject_verified=int(subject>=0), reference_verified=int(ref>=0),
                subject_coverage=int(correct_s.any()) if subject>=0 else '',
                reference_coverage=int(correct_r.any()) if ref>=0 else '',
                subject_after_transport=int(correct_s[ii].any()) if subject>=0 else '',
                reference_after_transport=int(correct_r[jj].any()) if ref>=0 else '',
                subject_regions=len(ii), reference_regions=len(jj), supported=int(bool(len(ii) and len(jj))),
                source_fit=int(prediction['source_fit']), timed_features=float(features['source_time']))
    for i,(arm,p) in enumerate(zip(ARMS,positions,strict=True)):
        pair = int((assigned[i]==(subject,ref)).all()) if known and i else ''
        answers.append(dict(**info,arm=arm,official_iou=float(ious[i]),dual_correct=pair,
                            subject_status=status_for(subject,assigned[i,0]),
                            reference_status=status_for(ref,assigned[i,1],has_ref[i]),
                            fallback=int(not has_ref[i]),subject_changed=int(p[0]!=0),
                            differing_role_permutation=int(not np.array_equal(features['edge'],prediction['shuffled_edge']))))
    return answers, float(np.max(np.abs(ious-check)))


PAIRS = (
    ('fixed_exchange_joint','additive_independent'),
    ('fixed_exchange_joint','additive_geometric'),
    ('fixed_exchange_joint','fixed_exchange_mismatched'),
    ('fixed_exchange_joint','r171_joint'),
    ('fixed_exchange_joint','r158_pca'),
    ('source_fit_joint','additive_geometric'),
    ('source_fit_joint','source_fit_mismatched'),
    ('source_fit_joint','fixed_exchange_joint'),
    ('additive_geometric','r171_joint'),
    ('r171_joint','r158_pca'),
)


def summarize(rows):
    from labels166 import RIGHT

    results, paired = [],[]
    for cohort in ('replay170','new_frames','combined'):
        c = [r for r in rows if cohort=='combined' or r['cohort']==cohort]
        for scope,pool in (('all',c),('parsed',[r for r in c if r['rewrite']=='supported']),
                           ('verified',[r for r in c if r['both_verified']]),
                           ('supported',[r for r in c if r['supported']])):
            groups = {'direction_macro':pool}
            groups.update({f"{g[0]['source']}_to_{g[0]['target']}":g for g in directions(pool)})
            for key,group in groups.items():
                arms = {a:sorted([r for r in group if r['arm']==a],key=orders) for a in ARMS}
                n = len(arms[ARMS[0]])
                if not n:
                    continue
                metadata = dict(cohort=cohort,scope=scope,route=key,n=n,dir_n=len(directions(group)),
                                frames=len({(r['cohort'],r['frame']) for r in group}),
                                recordings=len({r['recording'] for r in group}))
                for arm,rg in arms.items():
                    values = dict(zip(METRICS,macro(rg,lambda r:iou_metrics(r['official_iou'])),strict=True))
                    results.append(dict(**metadata,arm=arm,**values,subject_correct=sum(r['subject_status']==RIGHT for r in rg),
                                      subject_verified=sum(r['subject_verified'] for r in rg),
                                      reference_correct=sum(r['reference_status']==RIGHT for r in rg),
                                      reference_verified=sum(r['reference_verified'] for r in rg),
                                      both_verified=sum(r['both_verified'] for r in rg),
                                      dual_correct=sum(r['dual_correct']==1 for r in rg) if arm!=ARMS[0] else '',
                                      fallback=sum(r['fallback'] for r in rg)))
                for a,b in PAIRS:
                    x,y = arms[a],arms[b]
                    assert [orders(t) for t in x]==[orders(t) for t in y]
                    diff = np.stack([iou_metrics(t['official_iou'])-iou_metrics(u['official_iou'])
                                     for t,u in zip(x,y,strict=True)])
                    delta = macro([dict(r,delta=d) for r,d in zip(x,diff,strict=True)],lambda r:r['delta'])
                    paired.append(dict(**metadata,arm=a,control=b,**dict(zip(METRICS,delta,strict=True)),
                                       **{f'{k}_{name}':int(np.sum(w*diff[:,i]>1e-10))
                                          for name,w in [('gain',1),('loss',-1)] for i,k in enumerate(METRICS)},
                                       dual_gain=sum(t['dual_correct']==1 and u['dual_correct']!=1 for t,u in zip(x,y,strict=True)),
                                       dual_loss=sum(t['dual_correct']!=1 and u['dual_correct']==1 for t,u in zip(x,y,strict=True))))
    return results,paired


def main():
    from score171 import select_manifest

    path,seal,npz = checked()
    fit = PRIVATE / 'SOURCE_FIT.json'
    calibrations = read(fit)
    assert sha(fit) == read(path / 'LOCK.json')['source']
    features = PRIVATE / 'target'
    fs = read(features / 'SEAL.json')
    assert sha(features / 'SEAL.json') == read(path / 'LOCK.json')['target']
    out = HERE / 'results/binding_target'
    assert not (out / 'SCORE_SEAL.json').exists()
    out.mkdir(exist_ok=True,parents=True)
    atomic_json(out/'SCORE_LOCK.json',dict(code=sha(Path(__file__)),prediction=sha(path/'SEAL.json')))
    metrics,errors = [],[]
    seen = set()
    for cohort,manifest in select_manifest().items():
        aliases = {}
        for entry,blind,key in manifest:
            rec = entry['recording']
            aliases.setdefault(rec,f"{cohort[:3]}_{len(aliases)+1:02d}")
            for source in PLATFORMS:
                if source==entry['target']:
                    continue
                tag = f"{cohort}/{entry['case']:02d}_{source}.npz"
                assert tag not in seen and sha(features/tag)==fs['files'][tag]
                seen.add(tag)
                pred = npz(path/tag)
                assert pred['f_hash'] == fs['files'][tag] and tuple(pred['arms'])==ARMS
                weight = calibrations['coefficients'][source]['weights']
                r,e = score_once(entry,blind,key,cohort,source,aliases[rec],pred,npz(WORK/'predictions'/tag),
                                 npz(features/tag),weight)
                metrics.extend(r)
                errors.append(e)
    assert seen == seal['files'].keys() and len(metrics)==168*len(ARMS)
    assert sum(r['both_verified'] for r in metrics if r['arm']==ARMS[0])==60
    destinations = zip(('rows','metrics','contrasts'),(metrics,*summarize(metrics)),strict=True)
    for name,rows in destinations:
        csv_atomic(out/(name+'.csv'),rows)
    atomic_json(out/'SCORE_SEAL.json',dict(n=len(seen),hashes={p.name:sha(p) for p in out.glob('*.csv')},
            unchanged_model=True,enumeration_recheck=168,maximum_iou_error=max(errors),protected_data_access=False))
    print('Role exchange score sealed',Counter(r['supported'] for r in metrics if r['arm']==ARMS[0]),flush=True)


if __name__=='__main__':
    main()
