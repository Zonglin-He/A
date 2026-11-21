"""Score sealed task predictions, fixed-box yaw interventions and isolated rays."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cohort import OUT, PLATFORMS, PRIVATE, PROTOCOL, checked_lock, digest, read, read_panel, write
from geometry import axial, camera, cell_points, raw_returns
from inputs import case_id
from predict import ARMS, FIT, HEADING, PRIMARY, SUPPORT
from score import aggregate, canonical_box, delta, rotation, score_boxes, truth_for_slot
from score_region import membership, read_reference, save


PANELS = ('validation', 'recording_extension', 'all_recording_separated')
COMPARE = (('H','A'), ('H_yaw_only','H'), ('H_qwen_reanchored','H'),
           ('H','H_window'), ('H','H_central'), ('H_window','H_central'),
           ('Fit_region','Fit_window'), ('Fit_region','Fit_mismatch'))
FEATURES = ('normalized_gap', 'projected_arc', 'jitter_p90_degrees')


def by_panel(rows, panel):
    return [r for r in rows if (r['separation'] == 'new_recording' if panel == 'all_recording_separated'
                               else r['panel'] == panel)]


def task_stat(rows, arm):
    return aggregate([r[arm] for r in rows])


def cluster_interval(rows, arm, reference):
    by_record = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a,b = r[arm],r[reference]
        by_record[r['target']][r['recording']].append(
            [int(a>.25)-int(b>.25), int(a>.5)-int(b>.5), a-b])
    bootstrap = []
    rng = np.random.default_rng(20261121)
    for groups in by_record.values():
        total = np.array([np.asarray(g).sum(0) for g in groups.values()])
        counts = np.array([len(g) for g in groups.values()])
        picks = rng.integers(len(total), size=(4000,len(total)))
        bootstrap.append(total[picks].sum(1) / counts[picks].sum(1)[:,None])
    lower, upper = np.quantile(100*np.mean(bootstrap,axis=0), [.025,.975], axis=0)
    return {m:[float(a),float(b)] for m,a,b in zip(('acc25','acc50','miou'), lower,upper)}


def ray_residual(box, camera_points, transform):
    inv = np.linalg.inv(transform)
    axes = rotation(box[6])
    ray = (camera_points / camera_points[:,2:3]) @ inv[:3,:3].T @ axes
    origin = (inv[:3,3]-box[:3]) @ axes
    half = box[3:6]/2
    parallel = np.abs(ray) < 1e-12
    t1 = np.divide(-half-origin,ray,out=np.zeros_like(ray),where=~parallel)
    t2 = np.divide(half-origin,ray,out=np.zeros_like(ray),where=~parallel)
    enter = np.where(parallel,-np.inf,np.minimum(t1,t2)).max(1)
    leave = np.where(parallel,np.inf,np.maximum(t1,t2)).min(1)
    hits = ~(parallel & (np.abs(origin)>half)).any(1) & (leave>=enter) & (enter>0)
    return hits,np.abs(enter-camera_points[:,2])


def ray_pairs(frame, pred, gt):
    specification,size,transform = camera(frame)
    returns, xyz = raw_returns(frame,specification,size)
    window = np.asarray(pred['pixel'])*np.tile(size,2)/1000.
    _, held = cell_points(returns,xyz,window)
    target = membership(held, transform, gt)
    scored = {k: ray_residual(np.asarray(pred['boxes'][k]), held, transform) for k in FIT}
    totals = []
    identity = {k:frame[k] for k in ('uid','target','recording','panel','separation')}
    for scope, group in (('all_window', np.ones(len(held),bool)), ('inside_target',target)):
        region_hit, region_error = scored['Fit_region']
        for ref in ('Fit_window','Fit_mismatch'):
            ref_hit, ref_error = scored[ref]
            common = group & region_hit & ref_hit
            totals.append(dict(**identity,scope=scope,reference=ref, rays=int(group.sum()),
                both_hits=int(common.sum()), region_hits=int((group & region_hit).sum()),
                reference_hits=int((group & ref_hit).sum()),
                region_common_abs_sum=float(region_error[common].sum()),
                reference_common_abs_sum=float(ref_error[common].sum())))
    return totals


def aggregate_surfaces(rows):
    groups = defaultdict(list)
    for row in rows:
        for panel in (row['panel'], 'all_recording_separated') if row['separation']=='new_recording' else (row['panel'],):
            for target in (row['target'], 'pooled_rays'):
                groups[panel,target,row['scope'],row['reference']].append(row)
    fields = ('rays','both_hits','region_hits','reference_hits','region_common_abs_sum','reference_common_abs_sum')
    outputs = []
    for (panel,target,scope,ref), group in groups.items():
        sums = {f:sum(g[f] for g in group) for f in fields}
        n = sums['both_hits']
        outputs.append(dict(panel=panel,target=target,scope=scope,reference=ref, frames=len(group),**sums,
            region_common_mae_m=sums['region_common_abs_sum']/n if n else None,
            reference_common_mae_m=sums['reference_common_abs_sum']/n if n else None))
    return outputs


def join_predictions():
    panel = read_panel()
    prediction = read(OUT/'prediction_seal.json')
    locked = read(OUT/'locked_replay.json')
    assert prediction['method_lock'] == checked_lock() == locked['method_lock']
    assert prediction['code'] == digest(Path(__file__).with_name('predict.py'))
    assert prediction['qwen_seal'] == digest(OUT/'model_seal.json')
    assert prediction['coordinate_replay'] == digest(OUT/'locked_replay.json')
    assert locked['coordinate_comparisons'] == 288 and locked['maximum_coordinate_difference'] == 0.
    source = read(OUT/'v3_seal.json')
    assert source['stamp']['panel'] == prediction['panel'] == digest(PRIVATE/'panel.json')
    assert source['stamp']['protocol'] == prediction['protocol'] == digest(PROTOCOL)
    assert source['stamp']['method_lock'] == prediction['method_lock']
    assert source['stamp']['code'] == digest(Path(__file__).with_name('replay.py'))
    assert not prediction['task_labels_read'] and not source['task_labels_read']
    outputs = {}
    for src, sha in source['files'].items():
        path = PRIVATE/f'v3_{src}.json'
        assert digest(path) == sha
        values = read(path)
        assert values['stamp'] == source['stamp']
        assert set(values['rows']) == {f['uid'] for f in panel if f['target'] != src}
        outputs[src] = values['rows']
    path = PRIVATE/'predictions.json'
    assert digest(path) == prediction['predictions']
    estimates = read(path)
    assert set(estimates) == {f['uid'] for f in panel}
    return panel, estimates, outputs


def individual_records():
    panel, estimates, v3 = join_predictions()
    previous = read_reference()
    rows, directions, geometry = [], [], []
    historical_count, max_replay, cross_iou = 0, 0., 0.
    for frame in panel:
        uid,target = frame['uid'],frame['target']
        pred = estimates[uid]
        meta = read(frame['calibration'])['ground_info']
        gt = canonical_box(truth_for_slot(frame,meta))
        obs = pred['boxes'] | {src:v3[src][uid]['box'] for src in PLATFORMS if src != target}
        valid = {k:canonical_box(b) for k,b in obs.items() if b is not None}
        scored, err = score_boxes(np.stack(list(valid.values())), gt)
        values = dict.fromkeys(obs,0.) | dict(zip(valid,map(float,scored)))
        cross_iou = max(cross_iou,err)
        identity = {k:frame[k] for k in ('uid','target','recording','panel','separation')}
        diagnostics = {k:pred[k] for k in ('status','qwen_2d','qwen_3d','selected','fit_available',
                       'fit_bins','region_bins','region_points','support_central','support_window',
                       'matched_support','nontrivial_mismatch')}
        result = dict(**identity,**diagnostics, gt_aspect=float(gt[3]/gt[4]), **{k:values[k] for k in ARMS})
        for arm in ('A','H','H_qwen_reanchored'):
            if arm in valid:
                result[f'{arm}_center_error_m'] = float(np.linalg.norm(valid[arm][:3]-gt[:3]))
                result[f'{arm}_size_mean_relative_error'] = float(np.abs(valid[arm][3:6]/gt[3:6]-1).mean())
        if pred['selected']:
            xyz = np.asarray(pred['support_raw'])
            in_box = (np.abs((xyz-gt[:3]) @ rotation(gt[6])) <= gt[3:6]/2. + .05).all(1)
            result.update(pred['heading'], support_target_fraction=float(in_box.mean()),
                yaw_pca_error_deg=float(np.rad2deg(abs(axial(valid['H'][6]-gt[6])))),
                yaw_qwen_error_deg=float(np.rad2deg(abs(axial(valid['H_yaw_only'][6]-gt[6])))),
                yaw_gap_deg=float(np.rad2deg(abs(axial(valid['H'][6]-valid['H_yaw_only'][6])))))
            np.testing.assert_array_equal(valid['H'][:6],valid['H_yaw_only'][:6])
        if pred['fit_available']:
            geometry.extend(ray_pairs(frame,pred,gt))
        for src in (p for p in PLATFORMS if p != target):
            official = values[src]
            if frame['panel'] == 'validation':
                recorded = previous[src,case_id(frame)]
                historical_count += 1
                max_replay = max(max_replay,abs(float(recorded['separate_direct'])-official))
            directions.append(dict(**identity,source=src,A=result['A'],H=result['H'],B=official,
                region_branch=v3[src][uid]['region_branch']))
        rows.append(result)
    assert cross_iou < 1.e-8 and max_replay < 1.e-8
    assert historical_count == sum(r['panel']=='validation' for r in directions)
    isolation = Counter()
    for p in estimates.values():
        isolation.update(p.get('isolation',{}))
    assert not isolation['retained_withheld_raw']
    return rows,directions,geometry,dict(max_cross_iou_error=cross_iou, v3_archive_match_rows=historical_count,
          v3_archive_max_error=max_replay, fit_tree_isolation=dict(isolation))


def strata(rows, split):
    for target in [*dict.fromkeys(r['target'] for r in rows), 'macro']:
        subset = [r for r in rows if target == 'macro' or r['target'] == target]
        if subset:
            yield (split,target), subset


def average_metrics(group, arm):
    scores = [task_stat([r for r in group if r['target']==t],arm) for t in dict.fromkeys(r['target'] for r in group)]
    return {k:float(np.mean([s[k] for s in scores])) for k in scores[0]}


def paired_stats(group, arm, reference):
    stats = delta(group,arm,reference)
    a,b = (average_metrics(group,k) for k in (arm,reference))
    return stats | {f'delta{label}_pp':a[k]-b[k] for label,k in [('25','acc25'),('50','acc50'),('_miou','miou')]}


def task_and_pairs(rows,directions):
    main,paired,direct = [],[],[]
    for panel in PANELS:
        frame_rows, source_rows = by_panel(rows,panel), by_panel(directions,panel)
        for (_,target),subset in strata(frame_rows,panel):
            sr = [r for r in source_rows if target in (r['target'],'macro')]
            assert len(sr)==2*len(subset)
            for arm in ARMS:
                main.append(dict(panel=panel,target=target,arm=arm,frames=len(subset),scored_rows=len(subset),
                    recordings=len({r['recording'] for r in subset}),**average_metrics(subset,arm)))
            main.append(dict(panel=panel,target=target,arm='B',frames=len(subset),scored_rows=len(sr),
                    recordings=len({r['recording'] for r in subset}),**average_metrics(sr,'B')))
            for arm,ref in (*COMPARE,('H','B')):
                group = sr if ref=='B' else subset
                paired.append(dict(panel=panel,target=target,arm=arm,reference=ref,rows=len(group),
                    **paired_stats(group,arm,ref)))
        for src,dst in {(r['source'],r['target']) for r in source_rows}:
            same = [r for r in source_rows if (r['source'],r['target'])==(src,dst)]
            direct.append(dict(panel=panel,source=src,target=dst,rows=len(same), **task_stat(same,'B')))
    return main,paired,sorted(direct,key=lambda r:(r['panel'],r['source'],r['target']))


def auc(group, metric):
    p,n = [],[]
    for r in group:
        gain = r['H'] - r['H_yaw_only']
        value = r[metric] * (-1 if metric=='jitter_p90_degrees' else 1)
        if gain > 1.e-10:
            p.append(value)
        elif gain < -1.e-10:
            n.append(value)
    d = np.subtract.outer(p,n)
    return float((d>0).sum()+.5*(d==0).sum()),len(p),len(n)


def heading_and_support(rows):
    output, observability, coverage = [],[],[]
    for panel in PANELS:
        selected = [r for r in by_panel(rows,panel) if r['selected']]
        for (_,target),sub in strata(selected,panel):
            for group, cond in (('fixed_region',lambda r:True),('gt_elongated_target_majority',lambda r:
                    r['gt_aspect']>=1.2 and r['support_target_fraction']>=.5)):
                matching = [r for r in sub if cond(r)]
                if not matching:
                    continue
                for arm in HEADING:
                    output.append(dict(panel=panel,target=target,scope=group,method=arm,rows=len(matching),
                        median_pca_error_deg=float(np.median([r['yaw_pca_error_deg'] for r in matching])),
                        median_qwen_error_deg=float(np.median([r['yaw_qwen_error_deg'] for r in matching])),
                        median_yaw_gap_deg=float(np.median([r['yaw_gap_deg'] for r in matching])),
                        **paired_stats(matching,arm,'H')))
                for feature in FEATURES:
                    tied = len(matching)
                    good_pairs = total_pairs = within_good = within_total = 0
                    for t in dict.fromkeys(r['target'] for r in matching):
                        xs = [r for r in matching if r['target']==t]
                        value,p,n = auc(xs,feature)
                        tied -= p+n
                        good_pairs += value
                        total_pairs += p*n
                        for rec in dict.fromkeys(r['recording'] for r in xs):
                            value,p,n = auc([r for r in xs if r['recording']==rec],feature)
                            within_good += value
                            within_total += p*n
                    observability.append(dict(panel=panel,target=target,scope=group,feature=feature,rows=len(matching),
                         tie_iou=tied, target_stratified_pairs=total_pairs, recording_stratified_pairs=within_total,
                         target_stratified_auc=good_pairs/total_pairs if total_pairs else None,
                         recording_stratified_auc=within_good/within_total if within_total else None))
            supported = [r for r in sub if r['matched_support']]
            if supported:
                stats = {k:float(np.median([r[k] for r in supported])) for k in ('support_central','support_window','region_points')}
                for arm in SUPPORT+('H',):
                    coverage.append(dict(panel=panel,target=target,arm=arm,same_node_frames=len(supported),**stats,
                                         **average_metrics(supported,arm)))
    return output,observability,coverage


def main():
    rows, directions, surface_rows, audit = individual_records()
    metrics, pairs, v3 = task_and_pairs(rows,directions)
    orientation, reliability, coverage = heading_and_support(rows)
    for name, data in dict(rows=rows,source_rows=directions,metrics=metrics,paired=pairs,v3_directions=v3,
           orientation=orientation,observability=reliability,support_coverage=coverage,
           surface_rows=surface_rows,surface=aggregate_surfaces(surface_rows)).items():
        save(OUT/(name+'.csv'),data)
    intervals,counts = {},{}
    for panel in PANELS:
        for (_,target),subset in strata(by_panel(rows,panel),panel):
            group = f'{panel}/{target}'
            counts[group] = dict(frames=len(subset), selected=sum(r['selected'] for r in subset),
                fit_only=sum(r['fit_available'] for r in subset), matched_support=sum(r['matched_support'] for r in subset),
                nontrivial_mismatch=sum(r['nontrivial_mismatch'] for r in subset),
                statuses=dict(Counter(r['status'] for r in subset)), recordings=len({r['recording'] for r in subset}))
            sr = [r for r in by_panel(directions,panel) if target in (r['target'],'macro')]
            intervals[group] = {f'{arm}/{ref}':cluster_interval(sr if ref=='B' else subset,arm,ref)
                for arm,ref in (('H','A'),('H','B'),('H_yaw_only','H'),('Fit_region','Fit_mismatch'))}
    write(OUT/'score_receipt.json', dict(rows=len(rows), source_rows=len(directions),audit=audit, coverage=counts,
        intervals=intervals,seals={f:digest(OUT/f) for f in ('prediction_seal.json','v3_seal.json','panel_seal.json')},
        evaluator_sha256=digest(Path(__file__)), FINAL_access=False))
    print('Scored',len(rows),'physical frames',len(directions),'directional baselines',audit,flush=True)


if __name__=='__main__':
    main()
