"""Score sealed 2×2 predictions and summarize physical observation checks."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

from replay_factorial import ARMS, HERE, MONO, REGISTRATION, RELOC, RUN, STUDIES, TOL, check, code_stamp, digest, publish, read
from shared import canonical, setup
from inference_stats import cluster_weights, intervals
from evaluate_structure import stat_panel

AXES = ('length', 'width', 'height')
COMPARISONS = {
    'axis_vs_height_ray': ('axis_ray','height_ray'),
    'axis_vs_height_2d': ('axis_2d','height_2d'),
    '2d_vs_ray_height': ('height_2d','height_ray'),
    '2d_vs_ray_axis': ('axis_2d','axis_ray'),
}
ERRORS = ('center_m','horizontal_center_m','vertical_center_m','size_relative_pct','length_m','width_m','height_m')


def estimate(values, unit=1.):
    if not len(values):
        return dict(n=0)
    v = np.asarray(values)*unit
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), p90=float(np.quantile(v,.90)))


def rank_scores(series, a, b):
    return series[a]-series[b]


def effects(rows):
    if not rows:
        return {}
    boot = cluster_weights(rows)
    iou = {k:np.asarray([x['iou'][k] for x in rows]) for k in ARMS}
    acc = {k:(v>.50).astype(float) for k,v in iou.items()}
    differences = {k:(rank_scores(iou,a,b),rank_scores(acc,a,b)) for k,(a,b) in COMPARISONS.items()}
    differences['interaction'] = tuple(a-b for a,b in zip(differences['2d_vs_ray_axis'],differences['2d_vs_ray_height'],strict=True))
    result = {}
    for key,(m,s) in differences.items():
        result[key] = dict(delta_miou_pp=100*float(m.mean()), miou_ci=intervals(m*100,boot),
                           delta_acc50_pp=100*float(s.mean()), acc50_ci=intervals(s*100,boot),
                           corrected=int((s>0).sum()), degraded=int((s<0).sum()))
    return result


def error_vector(box, gt):
    off = box[:3]-gt[:3]
    ext = abs(box[3:6]-gt[3:6])
    return np.r_[np.linalg.norm(off),np.linalg.norm(off[:2]),abs(off[2]),100*np.mean(ext/gt[3:6]),ext]


def errors(rows):
    matched = [r for r in rows if all(k in r['errors'] for k in ARMS)]
    summary = {k:{f:estimate([r['errors'][k][j] for r in matched]) for j,f in enumerate(ERRORS)} for k in ARMS}
    pairs = {}
    if matched:
        w = cluster_weights(matched)
        x = np.array([r['errors']['axis_2d'] for r in matched])
        y = np.array([r['errors']['height_2d'] for r in matched])
        for j,f in enumerate(ERRORS):
            d = x[:,j]-y[:,j]
            pairs[f] = dict(mean_delta=float(d.mean()), ci=intervals(d,w))
    return dict(matched=len(matched), arms=summary, axis_2d_minus_height_2d=pairs)


def aggregate_rays(rows):
    totals = {k:sum(r[k] for r in rows) for k in ('n','ref_hits','changed','gained','lost')}
    totals.update(fraction=totals['changed']/totals['n'] if totals['n'] else None,
                  comparisons=len(rows), changed_boxes=sum(r['changed']>0 for r in rows),
                  maximum_common_depth_m=max((r['maximum_common_depth_m'] for r in rows),default=0.))
    return totals


def clipping(rows):
    trims = np.array([r['trim'] for r in rows],float).reshape(-1,3)
    changed = trims>TOL
    return dict(count=len(rows), any=int(changed.any(1).sum()), horizontal_any=int(changed[:,:2].any(1).sum()),
                centers=estimate([r['center_shift'] for r in rows if r['center_shift']>TOL]),
                trim={name:dict(changed=int(changed[:,i].sum()), positive_m=estimate(trims[changed[:,i],i]))
                      for i,name in enumerate(AXES)},
                points=sum(r['protected_points'] for r in rows), entries=sum(r['protected_entries'] for r in rows),
                lost_measured=sum(r['lost_measured'] for r in rows), lost_entries=sum(r['lost_entries'] for r in rows),
                rays={k:aggregate_rays([r['rays'][k] for r in rows]) for k in ('selection','audit')})


def geometry_report(rows):
    active = [r for r in rows if r['active']]
    keys = ('axis_2d_vs_height_2d','height_2d_vs_height_ray','axis_2d_vs_axis_ray')
    both = {k:clipping([x for r in active for x in r['candidate_audit'][k]]) for k in ('height','axis')}
    chosen = {arm:clipping([r['candidate_audit'][arm.split('_')[0]][r['winner'][arm]] for r in active]) for arm in ARMS}
    between = {k:{name:aggregate_rays([r['between'][k][name] for r in active]) for name in ('selection','audit')} for k in keys}
    return dict(active=len(active), candidates=both, selected=chosen, between=between,
                rank_switches={k:sum(r['winner'][k+'_ray']!=r['winner'][k+'_2d'] for r in active) for k in ('axis','height')},
                axis_vs_height_2d_pose_change=sum(r['winner']['axis_2d']!=r['winner']['height_2d'] for r in active),
                horizontal_on_axis_2d=sum(any(x>TOL for x in r['candidate_audit']['axis'][r['winner']['axis_2d']]['trim'][:2]) for r in active),
                fallbacks=dict(Counter(r['outputs']['height_2d']['readout'] for r in rows if not r['active'])))


def timing_report(timings):
    scopes = {'all':timings}
    for group in ('3eed','kitti','mono'):
        scopes[group] = [r for r in timings if r['group']==group]
    result = {}
    for k,v in scopes.items():
        result[k] = {}
        for level in ('all','C_ray','fallback'):
            sub = [r for r in v if level=='all' or r['active']==(level=='C_ray')]
            result[k][level] = {a:estimate([r['ms'][a] for r in sub]) for a in ('height_2d','axis_2d')}
            result[k][level]['paired_delta_ms'] = estimate([r['ms']['axis_2d']-r['ms']['height_2d'] for r in sub])
    return result


def evaluate():
    setup()
    from metric_image_common import inputs
    from shared import score_boxes
    from truth_score_inputs import truth_data
    sys.path.insert(0,str(STUDIES/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel

    path = RUN/'predictions-private.json'
    raw = check(path,'payload',RUN/'prediction-seal.json')
    assert raw['code']==code_stamp() and not raw['labels_read'] and raw['registration']==digest(REGISTRATION)
    assert raw['model_calls']==raw['parameter_updates']==0
    assert raw['benchmark']==digest(RUN/'timing.json')
    # Nothing above this line opens evaluation labels.
    old_rows_path = RELOC/'scored.json'
    check(old_rows_path,'scored',RELOC/'score-seal.json')
    old = {r['uid']:r for r in read(old_rows_path) if r['group']=='main'}
    mono = read(MONO/'mono-results.json')
    assert mono['only_offline_gt'] and mono['score_payload']==digest(MONO/'mono-scores-private.json')
    old_external = {r['uid']:r for r in read(MONO/'mono-scores-private.json')}
    strict = set(read(MONO/'mono-strict-roster-private.json'))
    assert len(strict)==118
    source = inputs()
    gt,_ = truth_data({u:source[u] for u in old})
    external, official_label_sha = labels_for_panel(read(MONO/'mono-input-seal.json'))
    assert official_label_sha==mono['official_kitti_label_sha256'] and external.keys()==old_external.keys()
    obs = read(path)
    assert len(obs)==raw['n']==873
    raw_by_uid = {r['uid']:r for r in obs}
    scored = []
    for row in obs:
        uid = row['uid']; domain = row['target']
        if domain=='mono':
            ref = external[uid]
            previous = old_external[uid]['iou']
            old_ray,old_2d = previous['old'],previous['promoted']
            recording = ref['recording']
        elif uid in old:
            ref = gt[uid]
            previous = old[uid]['pool']['cray']['iou']
            old_ray,old_2d = previous['formal'],previous['cached']
            recording = row['recording']
            assert recording==old[uid]['recording']
        else:
            continue
        box = canonical(ref['box'])
        predictions = {k:v['box'] for k,v in row['outputs'].items()}
        results = score_boxes(predictions,box)
        iou = {k:r['iou'] for k,r in results.items()}
        assert abs(iou['height_ray']-old_ray)<1.e-10 and abs(iou['height_2d']-old_2d)<1.e-10,(domain,uid,iou,old_ray,old_2d)
        params = {k:error_vector(canonical(v),box).tolist() for k,v in predictions.items() if v is not None}
        active = row['active']
        horizontal = active and any(x>TOL for x in row['candidate_audit']['axis'][row['winner']['axis_2d']]['trim'][:2])
        scored.append(dict(uid=uid,target=domain,recording=recording,active=active,horizontal=horizontal,iou=iou,errors=params))
    assert len(scored)==836 and {r['uid'] for r in scored if r['target']!='mono'}==old.keys()
    private = RUN/'scored-private.json'
    publish(private,scored)
    panels = {'development':[r for r in scored if r['target']!='mono'],
              'mono_strict':[r for r in scored if r['uid'] in strict],
              'mono_all':[r for r in scored if r['target']=='mono'],
              '3eed':[r for r in scored if r['target'] in {'waymo','drone','quad'}]}
    panels.update({p:[r for r in scored if r['target']==p] for p in ('waymo','drone','quad','kitti')})
    for p in ('development','mono_strict'):
        base = panels[p]
        panels[p+'_C_ray'] = [r for r in base if r['active']]
        panels[p+'_fallback'] = [r for r in base if not r['active']]
        panels[p+'_horizontal'] = [r for r in base if r['horizontal']]
    report = {}
    for k,v in panels.items():
        report[k] = dict(**stat_panel(v,ARMS,'height_2d'),factorial=effects(v),errors=errors(v),
            observations=geometry_report([raw_by_uid[r['uid']] for r in v]))
        if v:
            print(k,len(v),report[k]['observations']['active'],{n:[round(x['miou'],3),round(x['acc50'],3)]
                 for n,x in report[k]['metrics'].items()},flush=True)
    output = dict(panels=report,timing=timing_report(read(RUN/'timing.json')),prediction=raw,score_file=digest(private),
                  scorer=digest(Path(__file__)),historical_iou_rows_matched=len(scored),gt_used_only_after_seal=True)
    publish(RUN/'results.json',output)
    print('Scored sealed predictions and verified three historical implementation controls.',flush=True)


if __name__=='__main__':
    evaluate()
