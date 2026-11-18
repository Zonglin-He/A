"""Score sealed geometry with a second oriented-IoU implementation and matched pair counts."""
import numpy as np

from scope164 import HERE, PLATFORMS, WORK, atomic_json, read, sha, source_scope, verify_seal
from fit164 import ARMS, FITS, outside_cost
from audit160 import raw_annotations, sensor_frame, transform_gt
from audit162 import sensor_iou
from cycle163 import AXES
from geometry162 import axial, effective_count, local_points, surface_cost, to_parameters
from manifest162 import canonical
from score154 import many_iou
from score162 import put_table
from score163 import paired_metrics

CHANGES = ('surface_s0','outside_s0','surface_s1','outside_s1','surface_s1_no_relation')
CONTRASTS = (('outside_s0','surface_s0'),('outside_s1','surface_s1'),
             ('surface_s1','surface_s0'),('outside_s1','outside_s0'),('surface_s1_no_relation','surface_s1'))


def ground_truth(r):
    view = int(r['suffix'] or 0)
    label = np.asarray(r['val_GT'],dtype=float)
    phrase, annot = raw_annotations(r['metapath'])[int(r['object_index'])]
    assert canonical(phrase)==r['caption']
    np.testing.assert_allclose(transform_gt(annot,r['source'],view),label,atol=1e-7,rtol=0.)
    return label, sensor_frame(label,r['source'],view), view


def audited_ious(boxes, r, gt):
    label,sensor,view = gt
    a = many_iou(boxes,label)
    b = np.asarray([sensor_iou(x,sensor,r['source'],view) for x in boxes])
    for t in (.25,.50):
        np.testing.assert_array_equal(a>t,b>t)
    err = float(np.max(np.abs(a-b)))
    assert err<1e-7
    return a,err


def changed(box, origin):
    return bool(max(abs(box[:6]-origin[:6]).max(),abs(axial(box[6]-origin[6])))>1e-6)


def assemble():
    out = HERE/'results'
    verify_seal('GEOMETRY',WORK/'geometry')
    rows, heldout, movements, errors = [],[],[],[]
    for i,record in enumerate(source_scope()):
        gt = ground_truth(record)
        with np.load(WORK/'geometry'/f'{i:05d}.npz',allow_pickle=False) as p:
            assert tuple(p['arms'])==ARMS
            full,split = p['full'],p['split']
            initial = full[1]
            weights,points,held,raw = [p[k] for k in ('weights','points','held','raw_indices')]
            assert len(raw)==len(np.unique(raw))
            assert not len(set(map(tuple,points[held]))&set(map(tuple,points[~held])))
            side = (np.abs(local_points(points,initial))-initial[3:6]/2).max(1)
            rows.append(dict(row=i+1,source=record['source'],has_winner=bool(p['has_winner']),
                             relation_configured=bool(p['relation_active']),points=len(points),held_points=int(held.sum()),
                             **{f's{j}_outside_mass_fraction':float(w[side>0].sum()/max(w.sum(),1e-12)) for j,w in enumerate(weights)}))
            row = rows[-1]
            for scope,boxes in (('full',full),('split',split)):
                iou,err = audited_ious(boxes,record,gt)
                errors.append(err)
                for j,(name,value) in enumerate(zip(ARMS,iou)):
                    prefix = name if scope=='full' else 'split_'+name
                    row[prefix] = float(value)
                    if j<2:
                        continue
                    evaluations,status,accepted=p[scope+'_fits'][j-2]
                    row[prefix+'_nfev'] = int(evaluations)
                    row[prefix+'_status'] = int(status)
                    row[prefix+'_accepted'] = bool(accepted)
                    box = boxes[j]
                    row[prefix+'_updated'] = changed(box,initial)
                    idx = FITS[j-2][0]
                    mass = weights[idx,held]
                    check = mass.sum()>0 and effective_count(mass)>=3
                    if scope=='split':
                        entry = dict(row=i+1,source=record['source'],arm=name,eligible=check,
                                     held_effective_count=effective_count(mass))
                        for label,cost in (('outside',outside_cost),('surface',surface_cost)):
                            for label_box,b in (('before',initial),('after',box)):
                                entry[label+'_'+label_box] = float(mass@cost(points[held],to_parameters(b),float(p['scale']))/mass.sum()) if check else ''
                        heldout.append(entry)
                    if evaluations and scope=='full':
                        d = np.r_[box[:3]-initial[:3],np.log(box[3:6]/initial[3:6]),np.degrees(axial(box[6]-initial[6]))]
                        movements.append((record['source'],name,d))
    assert len(rows)==192
    put_table(out/'geometry_rows.csv',rows)
    put_table(out/'heldout_rows.csv',heldout)

    measures, contrasts, signed = [],[],[]
    for source in (*PLATFORMS,'all'):
        group = [r for r in rows if source in (r['source'],'all')]
        for mode,prefix in (('full',''),('fit_holdout','split_')):
            value = {name:np.asarray([r[prefix+name] for r in group]) for name in ARMS}
            for name,x in value.items():
                key = prefix+name
                aux = dict(fitted=sum(r[key+'_nfev']>0 for r in group),updated=sum(r[key+'_updated'] for r in group),
                           accepted=sum(r[key+'_accepted'] for r in group),
                           loss_nonincrease_rejected=sum(r[key+'_nfev']>0 and not r[key+'_accepted'] for r in group),
                           evaluations_total=sum(r[key+'_nfev'] for r in group)) if name in CHANGES else dict(fitted=0,
                               updated=0,accepted=0,loss_nonincrease_rejected=0,evaluations_total=0)
                measures.append(dict(source=source,mode=mode,arm=name,n=len(group),mIoU=100*x.mean(),
                                     Acc25=100*(x>.25).mean(), Acc50=100*(x>.50).mean(), **aux,
                                     **{'vs_initial_'+k:v for k,v in paired_metrics(value['initial'],x).items()}))
            for new,base in CONTRASTS:
                contrasts.append(dict(source=source,mode=mode,arm=new,control=base,n=len(group),
                                      **paired_metrics(value[base],value[new])))
        for arm in CHANGES:
            by_arm = np.stack([a for p,q,a in movements if source in (p,'all') and q==arm])
            for j,axis in enumerate(AXES):
                v = by_arm[:,j]
                signed.append(dict(source=source,arm=arm,axis=axis,n=len(v),median=float(np.median(v)),
                                   decrease=int((v< -1e-7).sum()),increase=int((v>1e-7).sum())))
    for mode,prefix in (('full',''),('fit_holdout','split_')):
        rel=[r for r in rows if r['relation_configured']]
        contrasts.append(dict(source='relation_subset',mode=mode,arm='surface_s1_no_relation',control='surface_s1',
                              n=len(rel),**paired_metrics(*[np.array([r[prefix+k] for r in rel])
                                  for k in ('surface_s1','surface_s1_no_relation')])))
    put_table(out/'geometry_metrics.csv',measures)
    put_table(out/'factor_contrasts.csv',contrasts)
    put_table(out/'signed_displacement.csv',signed)
    pooled = {r['arm']:r for r in measures if r['mode']=='full' and r['source']=='all'}
    region_stats = {arm:dict(n_checked=len(rows_),
                            changed_fit=sum(r['split_'+arm+'_updated'] for r in rows),
                            exterior_residual_lower=sum(r['outside_after']<r['outside_before']-1e-12 for r in rows_),
                            exterior_residual_higher=sum(r['outside_after']>r['outside_before']+1e-12 for r in rows_),
                            outside_mean_before=float(np.mean([r['outside_before'] for r in rows_])),
                            outside_mean_after=float(np.mean([r['outside_after'] for r in rows_])))
                    for arm in CHANGES for rows_ in [[r for r in heldout if r['arm']==arm and r['eligible']]]}
    harm = {arm:pooled[arm]['vs_initial_Acc50_loss'] for arm in CHANGES}
    faces = all(harm['outside_s'+str(j)]<harm['surface_s'+str(j)] for j in (0,1))
    is_relation = sum(r['relation_configured'] for r in rows)
    projection = dict(relation_configured_inputs=is_relation,
                      no_initial_mass_outside={f's{j}':sum(r[f's{j}_outside_mass_fraction']<=1e-12 and r['has_winner'] for r in rows) for j in (0,1)},
                      no_initial_mass_outside_but_outside_fit_moved={f's{j}':sum(r[f's{j}_outside_mass_fraction']<=1e-12
                          and r[f'outside_s{j}_updated'] for r in rows) for j in (0,1)})
    atomic_json(out/'GEOMETRY_SUMMARY.json',dict(n=len(rows),Acc50_harms=harm,conditional_face_gate=faces,
                matched_holdout=region_stats, **projection, FINAL_access=False))
    atomic_json(HERE/'GEOMETRY_AUDIT.json',dict(rows=len(rows),endpoint_checks=len(rows)*len(ARMS)*2,
                max_abs_oriented_iou_difference=max(errors),all_threshold_decisions_agree=True,
                original_C_matched_first_cycle=True,no_shared_raw_or_coincident_holdout=True,
                geometry_seal_sha256=sha(HERE/'GEOMETRY_SEAL.json'),scalars_sha256=sha(out/'geometry_rows.csv'),FINAL_access=False))
    print('Geometry independently scored',len(rows),'max error',max(errors),'Acc50 harms',harm,'face test',faces,flush=True)


if __name__=='__main__':
    assemble()
