"""Score only sealed geometry and audit rays, then summarize matched interventions."""
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from surface_grounding import (ARMS, BASE, FACTORS, FAMILIES, OUT, PRIVATE, SCALES, SPEC, STRICT,
                              digest, disagreement, read, read_panel, write)
from score import canonical_box, score_boxes, truth_for_slot
from score_confirmation import PANELS, by_panel, cluster_interval, paired_stats, strata
from score_region import membership, save


ORACLES = ('P_oracle', 'F_oracle', 'C_oracle')
COMPARE = (
    ('Q_yaw','P'), ('Q_center','Q_yaw'), ('Q_size','Q_yaw'),
    ('Q_joint','Q_yaw'), ('Q_joint','Q_center'), ('Q_joint','P'), ('P_oracle','P'),
    ('P_transfer','P'), ('F_pca','P'), ('C_pca','F_pca'), ('F_ray','P'), ('C_ray','P'),
    ('C_ray','C_qwen'), ('F_ray','F_qwen'),
    *tuple((name,'P') for name in SCALES),
    *tuple((f'{name}_{arm}',f'{name}_pca') for name in FAMILIES for arm in ('qwen','ray','oracle')),
)


def macro(values, group):
    target = np.array([r['target'] for r in group])
    return float(np.mean([np.asarray(values)[target==t].mean() for t in np.unique(target)]))


def stats(group, arm):
    x = np.array([r[arm] for r in group])
    return dict(acc25=100*macro(x>.25,group), acc50=100*macro(x>.50,group), miou=100*macro(x,group))


def audit_frame(estimate, truth, identity):
    payload = estimate['audit_payload']
    xyz = np.asarray(payload['optical'],float).reshape(-1,3)
    if not len(xyz):
        return []
    transform,cap = np.asarray(payload['transform']),payload['cap']
    masks = {'fit_node':np.asarray(payload['fit_node'],bool), 'all_window':np.ones(len(xyz),bool),
             'postseal_target':membership(xyz,transform,truth)}
    output = []
    for name in FAMILIES:
        pair = [estimate['boxes'][f'{name}_{a}'] for a in ('pca','qwen')]
        p,q = disagreement(pair,xyz,transform,cap)
        chose_q = estimate['chooses'][name]['qwen']
        chosen = q if chose_q else p
        for scope, active in masks.items():
            common = active & p[0] & q[0]
            output.append(dict(**identity,family=name,scope=scope,rays=int(active.sum()),
                pca_hits=int((active & p[0]).sum()),qwen_hits=int((active & q[0]).sum()),
                chosen_hits=int((active & chosen[0]).sum()),both_hits=int(common.sum()),
                pca_loss=float(p[2][active].sum()),qwen_loss=float(q[2][active].sum()),
                chosen_loss=float(chosen[2][active].sum()),
                pca_common_abs_sum=float(p[1][common].sum()),qwen_common_abs_sum=float(q[1][common].sum()),
                chosen_common_abs_sum=float(chosen[1][common].sum()),scored_selection=estimate['selection_rays']>=5))
    return output


def get_rows():
    panel = read_panel()
    seal = read(OUT/'prediction_seal.json')
    path = PRIVATE/'predictions.json'
    assert digest(path)==seal['predictions'] and digest(SPEC)==seal['protocol']
    assert digest(Path(__file__).with_name('surface_grounding.py'))==seal['implementation']
    assert seal['prior_prediction_seal']==digest(BASE/'results/prediction_seal.json')
    assert seal['panel']==digest(BASE/'private/panel.json') and not seal['FINAL_access']
    assert not seal['target_labels_read'] and seal['isolation']['retained_withheld_raw']==0
    predictions = read(path)
    with (BASE/'results/rows.csv').open(newline='') as stream:
        anchor = {r['uid']:r for r in csv.DictReader(stream)}
    assert set(anchor) == set(predictions) == {x['uid'] for x in panel}
    rows, audit, max_iou, max_anchor = [], [], 0., 0.
    for frame in panel:
        uid = frame['uid']
        est = predictions[uid]
        identity = {k:frame[k] for k in ('uid','panel','target','recording','separation')}
        truth = canonical_box(truth_for_slot(frame,read(frame['calibration'])['ground_info']))
        valid = {k:canonical_box(v) for k,v in est['boxes'].items() if v is not None}
        values = dict.fromkeys(ARMS,0.)
        details = {}
        if valid:
            q,gap = score_boxes(np.stack(tuple(valid.values())),truth)
            values.update(zip(valid,map(float,q)))
            max_iou = max(max_iou,gap)
            for arm in FACTORS:
                box = valid[arm]
                details[arm+'_center_m'] = float(np.linalg.norm(box[:3]-truth[:3]))
                details[arm+'_size_rel'] = float(np.abs(box[3:6]/truth[3:6]-1.).mean())
        expected = anchor[uid]
        for name,old in (('A','A'),('P','H'),('Q_yaw','H_yaw_only'),('Q_joint','H_qwen_reanchored')):
            max_anchor = max(max_anchor,abs(values[name]-float(expected[old])))
        derived = dict(P_oracle=max(values['P'],values['Q_joint']))
        for name in FAMILIES:
            derived[f'{name}_oracle'] = max(values[name+'_pca'],values[name+'_qwen'])
            info = est['chooses'].get(name,{})
            details[name+'_qwen_selected'] = info.get('qwen',False)
            costs = info.get('selection_loss')
            if costs is not None:
                details[name+'_selection_preference'] = costs[0]-costs[1]
        if est['strict']:
            audit.extend(audit_frame(est,truth,identity))
        rows.append(dict(**identity,**values,**derived,**details,
            **{k:est[k] for k in ('full','strict','status','selection_rays','audit_rays','fit_match','transfer')},
            face_center_shift=est.get('face_center_shift'),**{k+'_used':k in est['scales_valid'] for k in SCALES}))
    assert max_iou<1.e-8 and max_anchor<1.e-8
    return rows,audit,dict(max_two_scorer_gap=max_iou,max_previous_score_gap=max_anchor,
      prior_score_matches=4*len(rows),run_seal=digest(OUT/'prediction_seal.json'))


def groups(rows):
    for panel in PANELS:
        yield from strata(by_panel(rows,panel),panel)


def task_tables(rows):
    metrics,paired,intervals = [],[],{}
    for (panel,target), group in groups(rows):
        tag = dict(panel=panel,target=target)
        for arm in (*ARMS,*ORACLES):
            metrics.append(dict(**tag, scope='all', arm=arm, n=len(group),**stats(group,arm)))
        for a,b in COMPARE:
            paired.append(dict(**tag, scope='all', arm=a, reference=b, n=len(group),**paired_stats(group,a,b)))
            if (a,b) in (('P_transfer','P'),('Q_joint','P'),('F_ray','F_pca'),('C_ray','C_pca'),('C_ray','P')):
                intervals[f'{panel}/{target}/{a}-{b}'] = cluster_interval(group,a,b)
        for arm in SCALES:
            joint = [r for r in group if r['full'] and r[arm+'_used']]
            if joint:
                paired.append(dict(**tag,scope='both_supported',arm=arm,reference='P',n=len(joint),
                                   **paired_stats(joint,arm,'P')))
    return metrics,paired,intervals


def factorial_tables(rows):
    output, interactions = [],[]
    for (panel,target), group in groups(rows):
        for scope, selected in (('all',group),('fixed_full_node',[r for r in group if r['full']])):
            if not selected:
                continue
            for arm in FACTORS:
                center = [r[arm+'_center_m'] for r in selected if arm+'_center_m' in r]
                size = [r[arm+'_size_rel'] for r in selected if arm+'_size_rel' in r]
                output.append(dict(panel=panel,target=target,scope=scope,arm=arm,frames=len(selected),
                    center_median=float(np.median(center)),size_mean_abs_relative_median=float(np.median(size)),
                    **stats(selected,arm)))
            change = np.array([[r[k] for k in ('Q_joint','Q_center','Q_size','Q_yaw')] for r in selected])
            sign = np.array([1.,-1.,-1.,1.])
            shift = [r['face_center_shift'] for r in selected if r['full']]
            interactions.append(dict(panel=panel,target=target,scope=scope,frames=len(selected),
               median_cross_pose_center_shift=float(np.median(shift)),
               acc50_interaction_pp=100*macro((change>.50)@sign,selected),miou_interaction_pp=100*macro(change@sign,selected)))
    return output,interactions


def selection_table(rows):
    computed = []
    for (panel,target), group in groups(rows):
        eligible = [r for r in group if r['selection_rays']>=5]
        for family in FAMILIES:
            advantages = np.array([r[family+'_qwen']-r[family+'_pca'] for r in eligible])
            chooses = np.array([r[family+'_qwen_selected'] for r in eligible],bool)
            qbetter = advantages>1.e-10
            pbetter = advantages< -1.e-10
            decisive = qbetter|pbetter
            counts = dict(pca_better=int(pbetter.sum()),qwen_better=int(qbetter.sum()),
                chosen_correct=int(((qbetter&chooses)|(pbetter&~chooses)).sum()))
            computed.append(dict(panel=panel,target=target,family=family,n=len(group),
                fit_pair=sum(r['strict'] for r in group),eligible=len(eligible),
                fewer_than_five=sum(r['strict'] and r['selection_rays']<5 for r in group),
                selects_qwen=int(chooses.sum()),transfer=sum(r['transfer'] for r in group),
                decisive=int(decisive.sum()),**counts,
                accuracy_on_decisive=counts['chosen_correct']/int(decisive.sum()) if decisive.any() else None,
                always_pca_accuracy=counts['pca_better']/int(decisive.sum()) if decisive.any() else None,
                eligible_gt_oracle_gain_miou_pp=100*macro(np.maximum(advantages,0),eligible) if eligible else None))
    return computed


def audit_table(audit):
    fields = ('rays','pca_hits','qwen_hits','chosen_hits','both_hits','pca_loss','qwen_loss','chosen_loss',
              'pca_common_abs_sum','qwen_common_abs_sum','chosen_common_abs_sum')
    output=[]
    for (panel,target),group in groups(audit):
        for name in FAMILIES:
            for scope in ('fit_node','all_window','postseal_target'):
                subset=[r for r in group if r['scored_selection'] and r['scope']==scope and r['family']==name]
                sums={k:sum(r[k] for r in subset) for k in fields}
                if not sums['rays']:
                    continue
                output.append(dict(panel=panel,target=target,family=name,scope=scope,frames=len(subset),**sums,
                    **{k+'_cost':sums[k+'_loss']/sums['rays'] for k in ('pca','qwen','chosen')},
                    **{k+'_common_mae_m':sums[k+'_common_abs_sum']/sums['both_hits'] if sums['both_hits'] else None
                       for k in ('pca','qwen','chosen')}))
    return output


def run():
    rows,audit,pins = get_rows()
    metrics,paired,intervals = task_tables(rows)
    factorial,interactions=factorial_tables(rows)
    for name,data in dict(rows=rows,metrics=metrics,paired=paired,factorial=factorial,interactions=interactions,
        selection=selection_table(rows),ray_audit_rows=audit,ray_audit=audit_table(audit)).items():
        save(OUT/(name+'.csv'),data)
    coverage = {}
    for (panel,target),group in groups(rows):
        coverage[f'{panel}/{target}']=dict(frames=len(group),recordings=len({r['recording'] for r in group}),
            full=sum(r['full'] for r in group),fit_pair=sum(r['strict'] for r in group),
            selection=sum(r['selection_rays']>=5 for r in group),transfer=sum(r['transfer'] for r in group),
            statuses=dict(Counter(r['status'] for r in group)))
    write(OUT/'evaluation_receipt.json',dict(pins=pins,coverage=coverage,paired_recording_intervals=intervals,
        evaluator=digest(__file__),frames=len(rows),FINAL_access=False))
    print('Scored',len(rows),'frames and',len(audit),'independent ray summaries',pins,flush=True)


if __name__=='__main__':
    run()
