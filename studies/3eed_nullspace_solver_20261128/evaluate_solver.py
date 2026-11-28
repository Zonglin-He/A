"""Evaluate only sealed blind boxes, preserving paired R4 and 2D controls."""
from collections import Counter
from pathlib import Path
import json

import numpy as np

from run_experiment import ARMS, HERE, R4, RELOC, RUN, check, digest, inputs, publish, read, setup
from evaluate_structure import stat_panel
from inference_stats import cluster_weights, intervals
from shared import score_boxes
from truth_score_inputs import truth_data

EXPERIMENT_ARMS = (*ARMS,'routefree_initial','routefree')
METRICS = ('formal','legacy','r4_7d','wide_pool',*EXPERIMENT_ARMS)
PAIRINGS = {
    'isotropic_vs_ray': ('isotropic','ray_only'),
    'projected_vs_ray': ('projected','ray_only'),
    'projected_vs_isotropic': ('projected','isotropic'),
    'joint_vs_projected': ('joint','projected'),
    'joint_vs_quadratic': ('joint','joint_quadratic'),
    'soft_vs_hard': ('joint_soft','joint'),
    'shape_vs_full': ('joint_shape','joint'),
    'no_prior_image_vs_ray': ('joint_no_prior','ray_only'),
    'no_prior_huber_vs_quadratic': ('joint_no_prior','joint_no_prior_quadratic'),
    'seed_center_vs_qwen_center': ('joint_seed_center','joint'),
    'routefree_vs_seed': ('routefree','routefree_initial'),
}


def describe(values):
    values = list(values)
    if not values:
        return dict(n=0)
    x = np.array(values)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),
                p95=float(np.quantile(x,.95)))


def diff(rows, key, control, weights):
    a = np.array([r['iou'][key] for r in rows])
    b = np.array([r['iou'][control] for r in rows])
    m = 100*(a-b)
    t = 100*((a>.50).astype(float)-(b>.50))
    return dict(delta_miou=float(m.mean()),miou_ci=intervals(m,weights),delta_acc50=float(t.mean()),
                acc50_ci=intervals(t,weights),rescue=int((t>0).sum()),harm=int((t<0).sum()))


def audit_pairs(rows):
    out = {}
    for k in (*EXPERIMENT_ARMS,'legacy'):
        ref = 'formal_on_routefree_audit' if k.startswith('routefree') else 'formal'
        a = [r for r in rows if r['audit'].get(k) is not None and r['audit'].get(ref) is not None]
        if not a:
            continue
        delta = np.array([r['audit'][k]['truncated_m']-r['audit'][ref]['truncated_m'] for r in a])
        miou = 100*np.array([r['iou'][k]-r['iou']['formal'] for r in a])
        w = cluster_weights(a)
        down = delta < -1e-9
        out[k] = dict(n=len(a),baseline=ref,delta_cost=describe(delta),cost_ci=intervals(delta,w),
                      audit_decreases=int(down.sum()),worse_iou_given_lower_audit=int((down&(miou<-1e-7)).sum()),
                      better_iou_given_lower_audit=int((down&(miou>1e-7)).sum()))
    return out


def ranks(rows):
    out = {}
    for name in ('projected','joint','joint_soft','routefree'):
        x = [r['details'][name] for r in rows if name in r['details']]
        if not x:
            continue
        out[name] = {step:{key:dict(sorted(Counter(v[step][key] for v in x).items()))
                           for key in ('rank','joint_rank','modal_rank')}
                     for step in ('start','end')}
        out[name].update(n=len(x),accepted=describe(v['accepted'] for v in x),
                         start_rank=describe(v['start']['rank'] for v in x),
                         start_joint_rank=describe(v['start']['joint_rank'] for v in x),
                         start_modality_normalized=describe(v['start']['modal_rank'] for v in x),
                         gained_rank=sum(v['start']['joint_rank']>v['start']['rank'] for v in x),
                         gained_relative_modal_rank=sum(v['start']['modal_rank']>v['start']['rank'] for v in x))
    return out


def panel(rows, extended=False):
    out = stat_panel(rows, METRICS, 'formal')
    if not rows:
        return out
    weights = cluster_weights(rows)
    out['contrasts'] = {key:diff(rows,*arms,weights) for key,arms in PAIRINGS.items()}
    out['routes'] = dict(Counter(r['route'] for r in rows))
    out['eligible'] = sum(bool(r['details']) for r in rows)
    out['n_without_rays'] = sum(r['n_selection']==0 for r in rows)
    out['rank'] = ranks(rows)
    if extended:
        out['audit'] = audit_pairs(rows)
        out['centers_m'] = {k:describe(r['center'][k] for r in rows if k in r['center']) for k in METRICS}
        out['timing_s'] = describe(r['seconds'] for r in rows)
    return out


def main():
    source = RUN/'predictions-private.json'
    seal = check(source,'artifact',RUN/'prediction-seal.json')
    assert len(read(source))==seal['n']==702
    assert seal['pins']=={k:digest(HERE/k) for k in seal['pins']}
    assert not seal['target_labels_read'] and not seal['training'] and seal['source_model_calls']==0
    setup()
    gt,_ = truth_data(inputs())
    previous = check(R4/'scored.json','scored',R4/'score-seal.json')
    history = {r['uid']:r for r in read(R4/'scored.json')}
    reloc = check(RELOC/'scored.json','scored',RELOC/'score-seal.json')
    original = {r['uid']:r for r in read(RELOC/'scored.json') if r['group']=='main'}
    assert gt.keys()==history.keys()==original.keys() and len(gt)==665
    rows=[]
    for prediction in read(source):
        u = prediction['uid']
        if u not in gt:
            continue
        cached = original[u]
        prior = history[u]
        values = score_boxes(prediction['boxes'],gt[u]['box'])
        iou = {k:v['iou'] for k,v in values.items()}
        assert abs(iou['formal']-cached['pool']['cray']['iou']['cached'])<1e-10,(u,iou['formal'])
        assert abs(iou['legacy']-prior['baseline']['formal']['iou'])<1e-10
        rr = prior['R4']
        iou['r4_7d'] = rr['boxes']['E_pattern']['iou']
        iou['wide_pool'] = cached['pool']['natural']['iou']['cached']
        fields = ('uid','target','recording','route','details','audit','n_selection','n_audit',
                  'n_routefree','n_fit_routefree','seconds')
        rows.append(dict({k:prediction[k] for k in fields},iou=iou,identity=prior['identity'],
                         historical_hard_feasible=rr.get('gt',{}).get('feasible'),
                         center={k:v['center'] for k,v in values.items() if 'center' in v}))
    cray=[r for r in rows if r['route']=='C_ray']
    correct=[r for r in cray if r['identity']=='target']
    rejected=[r for r in correct if r['historical_hard_feasible'] is False]
    feasible=[r for r in correct if r['historical_hard_feasible'] is True]
    assert [len(cray),len(correct),len(rejected),len(feasible)]==[315,221,115,106]
    groups={'all':rows, 'cray':cray, 'cray_correct':correct,'cray_hard_excluded':rejected,'cray_hard_feasible':feasible}
    for p in ('waymo','drone','quad','kitti'):
        groups[p] = [r for r in rows if r['target']==p]
        groups['cray_'+p] = [r for r in cray if r['target']==p]
    for p in ('H','A','missing_3d'):
        groups['route_'+p] = [r for r in rows if r['route']==p]
    groups['no_ray_valid'] = [r for r in rows if r['details'] and not r['n_selection']]
    groups['rank_0_to_1'] = [r for r in rows if r['details'] and r['details']['joint']['start']['rank']<=1]
    groups['rank_3_to_4'] = [r for r in rows if r['details'] and r['details']['joint']['start']['rank']>=3]
    path=RUN/'scores-private.json'
    publish(path,rows)
    result=dict(panels={k:panel(v, extended=k in ('all','cray','cray_hard_excluded')) for k,v in groups.items()},
                prediction=seal,scored=digest(path), scorer=digest(Path(__file__)), only_offline_gt=True,
                prior_references=dict(r4=previous['scored'],xy=reloc['scored']))
    dest=RUN/'aggregate.json'
    publish(dest,result)
    print('Scored',len(rows),'and reports at',dest,flush=True)
    for key in ('all','cray','waymo','drone','quad','kitti','cray_hard_excluded'):
        p = result['panels'][key]
        print(key,p['n'],{k:(round(v['miou'],2),round(v['acc50'],2)) for k,v in p['metrics'].items()},flush=True)


if __name__=='__main__':
    main()
