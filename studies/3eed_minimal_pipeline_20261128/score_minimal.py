"""Score immutable predictions and resolve the predeclared Minimal-D configuration."""
from collections import Counter
from pathlib import Path
import sys

from replay_minimal import BLIND_ARMS, HERE, PREVIOUS, RUN, check, digest, publish, read
from ablate import MONO
from evaluate_structure import stat_panel
from shared import canonical, score_boxes, setup


def score(row, truth):
    boxes = {arm:(None if r['box'] is None else tuple(r['box'])) for arm,r in row['predictions'].items()}
    scored = score_boxes({v:v for v in boxes.values() if v is not None}, canonical(truth))
    return {name:0. if value is None else scored[value]['iou'] for name,value in boxes.items()}


def evidence(rows, arm):
    witnesses = [r['audit'][arm] for r in rows]
    candidate = [check for w in witnesses for check in w['candidates']]
    def totals(group, fields):
        return {k:sum(v[k] for v in group) for k in fields}

    counters = totals(witnesses, ('support_points','selection','audit','overlap_selection','overlap_audit','ties'))
    counts = dict(Counter(r['predictions'][arm]['readout'] for r in rows))
    assert len(candidate) == 2 * counts.get('C_ray',0)
    order = 2 * (counters['ties'] if arm in ('Minimal-C','C_single','C_dual') else counts.get('C_ray',0))
    return dict(counts=counts, scope=len(rows), candidates=len(candidate),
                binding=counters, ranking_cost_evaluations=order,
                ranking_additional_first_depth_evaluations=order if arm in BLIND_ARMS[:3] else 0,
                witnesses=totals(candidate, ('non_nested','measured','measured_lost','entries','entries_lost')),
                selection=totals([c['selection'] for c in candidate], ('n','changed','gained','lost')),
                audit=totals([c['audit'] for c in candidate], ('n','changed','gained','lost')))


def resolve(panels, checks):
    trials = {}
    for arm, setting in (('Minimal-A','C_single'),('Minimal-B','C_dual')):
        primary = {p:panels[p]['metrics'][arm] for p in ('development','mono_strict')}
        passed_quality = all(v['delta_miou']>=-.5 and v['delta_acc50']>=-1. for v in primary.values())
        passed_evidence = checks[arm]['selection']['n']>0 and checks[arm]['selection']['changed']==0
        trials[arm] = dict(quality=passed_quality, selection=passed_evidence, passes=passed_quality and passed_evidence,
                           composition=setting)
    choice = next((x['composition'] for x in trials.values() if x['passes']), 'Minimal-C')
    return choice, trials


def main():
    seal = check(RUN/'blind.json', 'payload', RUN/'seal.json')
    assert seal['n']==873 and not seal['labels_read'] and not seal['model_calls'] and not seal['trained']
    assert all(digest(HERE/f)==value for f,value in seal['files'].items())
    historical = read(PREVIOUS/'results.json')
    assert historical['prediction']['payload']==seal['previous']
    timing = read(RUN/'timing.json')
    assert timing['code']==digest(HERE/'time_backend.py') and timing['n']==80
    setup()
    from metric_image_common import inputs
    from truth_score_inputs import truth_data
    sys.path.insert(0,str(HERE.parent/'3eed_2d_pose_finalization_20261128'))
    from run_mono import labels_for_panel
    previous_scores = {r['uid']:r for r in read(PREVIOUS/'scores-private.json')}
    assert historical['score_sha']==digest(PREVIOUS/'scores-private.json') and len(previous_scores)==836
    source = inputs()
    gt, _ = truth_data({u:source[u] for u,r in previous_scores.items() if r['target']!='mono'})
    mono, mono_stamp = labels_for_panel(read(MONO/'mono-input-seal.json'))
    assert mono_stamp==read(MONO/'mono-results.json')['official_kitti_label_sha256']
    strict = set(read(MONO/'mono-strict-roster-private.json'))
    assert len(strict)==118
    all_rows = read(RUN/'blind.json')
    by_uid = {r['uid']:r for r in all_rows}
    scored = []
    for uid, prior in previous_scores.items():
        row = by_uid[uid]
        box = (mono if row['target']=='mono' else gt)[uid]['box']
        iou = score(row, box)
        assert abs(iou['Full']-prior['iou']['formal'])<=1.e-12
        assert abs(iou['Minimal-B']-prior['iou']['O_two_scales'])<=1.e-12
        assert all(abs(iou[a]-iou[b])<=1.e-11 for a,b in
                   [('Minimal-C','Full'),('C_single','Minimal-A'),('C_dual','Minimal-B')])
        assert all((iou[a]>.5)==(iou[b]>.5) for a,b in
                   [('Minimal-C','Full'),('C_single','Minimal-A'),('C_dual','Minimal-B')])
        scored.append(dict(uid=uid, target=row['target'], recording=prior['recording'], branch=prior['branch'], iou=iou))
    develop = [r for r in scored if r['target']!='mono']
    held = [r for r in scored if r['uid'] in strict]
    cohorts = {'development':develop,'mono_strict':held,'mono_all':[r for r in scored if r['target']=='mono'],
               '3eed':[r for r in develop if r['target']!='kitti']}
    cohorts.update({p:[r for r in develop if r['target']==p] for p in ('waymo','drone','quad','kitti')})
    for label, rows in (('development',develop),('mono_strict',held)):
        for branch in ('C_ray','H','A'):
            cohorts[label+'_'+branch] = [r for r in rows if r['branch']==branch]

    panes = {key:stat_panel(value, BLIND_ARMS, 'Full') for key,value in cohorts.items()}
    audit = {arm:evidence(all_rows,arm) for arm in BLIND_ARMS}
    old = historical['candidate_witness_audit_873']['formal']
    assert all(audit['Full'][k]==old[k] for k in ('candidates','selection','audit'))
    assert audit['Minimal-C']['selection']['changed']==0
    assert audit['Minimal-C']['witnesses']['non_nested']==0
    assert audit['Minimal-C']['witnesses']['entries_lost']==0
    assert audit['Minimal-C']['witnesses']['measured_lost']==0
    assert audit['Minimal-C']['binding']['overlap_selection']==audit['Full']['binding']['selection']
    assert audit['Minimal-C']['binding']['overlap_audit']==audit['Full']['binding']['audit']
    selection, rule = resolve(panes, audit)
    for pane in panes.values():
        pane['metrics']['Minimal-D'] = pane['metrics'][selection]
    for result in timing['timing'].values():
        result['metrics']['Minimal-D'] = result['metrics'][selection]
    audit['Minimal-D'] = audit[selection]
    for row in scored:
        row['iou']['Minimal-D'] = row['iou'][selection]

    registry = dict(scalars_held_constant={'pixel_cell_px':4,'central_window_fraction':.5,'returns_gate':5,
                    'foreground_margin_m':.2,'neighbors':8,'core_neighbor_rank':4,'min_component':8,
                    'rectangle_min_side_px':2,'ray_truncation_floor_m':.1,'ray_truncation_span_fraction':.5},
                    scale_indices={'Full':[0,2,5,7], 'Minimal-A':[7], 'Minimal-B':[0,7], 'Minimal-C':[0,2,5,7],
                                   'C_single':[7],'C_dual':[0,7]},
                    new_hyperparameters=0, model_decodes_per_live_query=2, fresh_decodes_in_replay=0)
    registry['scale_indices']['Minimal-D'] = registry['scale_indices'][selection]
    max_error = {arm:max(r['equivalent_delta'][arm] for r in all_rows) for arm in ('Minimal-C','C_single','C_dual')}
    publish(RUN/'scores-private.json', scored)
    summary = dict(prediction_seal=seal, score_sha256=digest(RUN/'scores-private.json'), labels_only_after_seal=True,
                   scored=len(scored), controls_matched=len(scored), max_box_delta=max_error, metrics=panes,
                   evidence_873=audit, acceptance={'miou_tolerance_pp':.5,'acc50_tolerance_pp':1.,'scale_tests':rule},
                   minimal_d=selection, settings=registry, backend_ms=timing['timing'],
                   code={'score_minimal.py':digest(Path(__file__)), 'time_backend.py':timing['code']})
    publish(RUN/'results.json', summary)
    print('Minimal-D:',selection,rule,'\nMax box difference:',max_error,flush=True)
    for name, p in panes.items():
        print(name, p['n'], [(a, round(x['miou'],3), round(x['acc50'],3)) for a,x in p['metrics'].items()],flush=True)


if __name__ == '__main__':
    main()
