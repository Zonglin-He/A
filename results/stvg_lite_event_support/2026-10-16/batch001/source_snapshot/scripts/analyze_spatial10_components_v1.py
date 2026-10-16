"""Offline F35 labels/metrics/statistics; never imported by a prediction worker."""
import argparse
import collections
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read, write, status, load, sha
from scripts.run_spatial10_components_v1 import OUT, F34, plan, dest, existing, SEEDS
from scripts.score_stvg_fullscale_v1 import score
from scripts.audit_decota_cal_v1 import metrics as independent_score
from vg_tta.box_stability_diagnostics_v1 import overlap

BASE_METRICS = ['vIoU_corrected', 'sIoU', 'tIoU', 'temporal_recall', 'temporal_precision', 'span']
SPACE_METRICS = BASE_METRICS+['U_all_sIoU', 'U_tube_sIoU', 'Outside_sIoU', 'observed_sIoU',
    'anchor_sIoU', 'view_common_U_all_sIoU', 'view_common_U_tube_sIoU', 'manual_sIoU', 'generated_sIoU']
TEMP_METRICS = BASE_METRICS+['start_error_seconds', 'end_error_seconds', 'start_error_normalized', 'end_error_normalized']


def labels_for(rows):
    # The immutable JSON is a container. Only requested keys are retained or
    # scored, and the dev-only selection path never evaluates panel outcomes.
    p = read(ROOT/'artifacts/stvg_fullscale_diagnostics_v1/lock.json')
    assert sha(p['labels']) == p['labels_sha256']
    all_labels = read(p['labels'])
    return {r['key']: all_labels[r['key']] for r in rows}


def aggregate(values, sources):
    d = collections.defaultdict(list)
    for v, s in zip(values, sources):
        if v is not None:
            assert math.isfinite(v)
            d[s].append(v)
    if not d:
        return dict(mean=None, ci95=None, sources=0, queries=0, query_mean=None)
    a = np.array([np.mean(d[s]) for s in sorted(d)], float)
    q = np.array([v for v in values if v is not None], float)
    rng = np.random.default_rng(20260914)
    boot = a[rng.integers(len(a), size=(10000, len(a)))].mean(axis=1)
    ntrim = int(.1*len(a)); aa = np.sort(a)
    return dict(mean=float(a.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), sources=len(a), queries=len(q),
        query_mean=float(q.mean()), median=float(np.median(a)), trimmed10=float(aa[ntrim:len(a)-ntrim].mean()),
        wins=int((a > .001).sum()), neutral=int((np.abs(a) <= .001).sum()), harms=int((a < -.001).sum()),
        negative_gt5pp=int((a < -.05).sum()), negative_gt10pp=int((a < -.1).sum()),
        minimum=float(a.min()), maximum=float(a.max()),
        worst10_mean=float(np.sort(a)[:max(1, math.ceil(len(a)*.1))].mean()))


def delta(a, b):
    return None if a is None or b is None else a-b


def summary(rows, names, metrics, comparisons):
    arms, contrasts = {}, {}
    for n in names:
        rr = [r for r in rows if n in r['arms']]
        if not rr:
            continue
        arms[n] = {m: aggregate([r['arms'][n].get(m) for r in rr], [r['group'] for r in rr]) for m in metrics}
    for a, b in comparisons:
        rr = [r for r in rows if a in r['arms'] and b in r['arms']]
        if not rr:
            continue
        contrasts[a+' - '+b] = {m: aggregate([delta(r['arms'][a].get(m), r['arms'][b].get(m)) for r in rr], [r['group'] for r in rr]) for m in metrics}
    return dict(queries=len(rows), sources=len({r['group'] for r in rows}), arms=arms, contrasts=contrasts)


def annotation_flags(rows, labels):
    from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations
    raw = load_vidor_annotations(ROOT/'downloads/vidor/validation-annotation.zip', None)
    result = {}
    for r in rows:
        ids = r['input']['frame_ids']; gt = labels[r['key']]
        flags = ['unknown']*len(ids)
        if r['key'].startswith('vidstg'):
            tid = gt['official_annotation']['target_id']; clip = raw[r['source']]
            for j, fid in enumerate(ids):
                if not gt['valid'][j]:
                    continue
                entries = [b for b in clip['trajectories'][fid] if b['tid'] == tid]
                assert len(entries) == 1
                b = entries[0]
                if 'generated' in b and 'tracker' in b:
                    flags[j] = 'manual' if b['generated'] == 0 else 'generated_'+b['tracker']
        result[r['key']] = flags
    return result


def checked_score(boxes, gt, ids, interval):
    m, quality = score(boxes, gt, ids, interval)
    if m['vIoU_corrected'] is not None and m['sIoU'] is not None:
        other = independent_score(boxes, gt, ids, interval)
        for k in ['vIoU_corrected', 'sIoU', 'tIoU', 'temporal_recall', 'temporal_precision']:
            assert abs(m[k]-other[k]) < 1e-10, (k, m[k], other[k])
    return m, quality


def spatial_metric(boxes, gt, x, observed, anchors, flags):
    m, q = checked_score(boxes, gt, x['frame_ids'], x['native_indices'])
    valid = np.array(gt['valid'], bool); n = len(valid)
    obs = np.zeros(n, bool); obs[observed] = True
    am = np.zeros(n, bool); am[[a['position'] for a in anchors]] = True
    tube = np.zeros(n, bool); l, h = x['native_indices']; tube[l:h+1] = True
    common = np.zeros(n, bool); common[x['expert']['common_observed']] = True
    masks = dict(G=valid, observed=valid&obs, anchor=valid&am, U_all=valid&~obs,
        U_tube=valid&tube&~obs, Outside=valid&~tube,
        view_common_U_all=valid&~common, view_common_U_tube=valid&tube&~common,
        manual=valid&np.array([v == 'manual' for v in flags]),
        generated=valid&np.array([v.startswith('generated_') for v in flags]))
    for k, mask in masks.items():
        m[k+'_sIoU'] = float(q[mask].mean()) if mask.any() else None
        m[k+'_count'] = int(mask.sum())
    return m, q, {k: np.flatnonzero(mask).tolist() for k, mask in masks.items()}


def fit_diagnostics(z):
    gradients = []
    for t in z['path']:
        a = dict(step=t['step'], loss=t['loss'], parts=t['parts'], group_delta=t['group_delta'],
                 indices=t['indices'], accepted=t.get('accepted'), group_update=t.get('group_update'), trials=t.get('trials'))
        gd = t.get('gradient_decomposition')
        if gd:
            a['gradient'] = {k: gd[k] for k in ('data_norm', 'reg_norm', 'cosine', 'sum_max_error')}
            a['gradient']['reg_data_ratio'] = gd['reg_norm']/max(gd['data_norm'], 1e-30)
            a['actual_Adam_displacement_norm'] = float(t['actual_displacement'].double().norm())
            a['proposal_norm'] = t['proposal_norm']
        gradients.append(a)
    return dict(parameter_count=z['parameter_count'], state_delta=z['state_delta'], best_step=z['best_step'],
        skipped=z['skipped'], failure=z['failure'], steps_attempted=sum('accepted' in t for t in z['path']),
        accepted=sum(bool(t.get('accepted')) for t in z['path']), rejected=sum(t.get('accepted') is False for t in z['path']),
        actual_updated=sum(any(v > 0 for v in t.get('group_update', {}).values()) for t in z['path']),
        trajectories=gradients, seconds=z['seconds'], backwards=z['backwards'], extra_vjp=z.get('extra_vjp', 0),
        peak_bytes=z.get('peak_bytes'), full_replay_seconds=z.get('full_replay_seconds'),
        suffix_forwards=z.get('suffix_forwards'), suffix_seconds=z.get('suffix_seconds'), backtrack_seconds=z.get('backtrack_seconds'),
        initial_loss=z['path'][0]['loss'], selected_loss=z['path'][z['best_step']]['loss'])


def teacher_quality(aa, gt, ids):
    rows = []
    for a in aa:
        j = a['position']
        iou = float(overlap(np.array([a['box']]), np.array([gt['boxes'][j]]))[0]) if gt['valid'][j] else None
        rows.append(dict(position=j, frame_id=ids[j], GT_valid=gt['valid'][j], IoU=iou,
                         weight=a.get('weight', 1.), score=a.get('score'), disagreement=a.get('disagreement')))
    valid = [a for a in rows if a['IoU'] is not None]
    return dict(anchors=rows, accepted=len(aa), valid_GT_anchors=len(valid),
        mean_IoU=float(np.mean([a['IoU'] for a in valid])) if valid else None,
        weight_sum=sum(a.get('weight', 1) for a in aa),
        support_range=(ids[max(a['position'] for a in aa)]-ids[min(a['position'] for a in aa)])/(ids[-1]+1-ids[0]) if aa else 0.)


def pregate_quality(ex, gt):
    out = []
    for (view, pos), z in ex['observations'].items():
        if not gt['valid'][pos]:
            continue
        d, probe = z['detection'], z['probe']; b = np.asarray(d['all_boxes'])
        if not len(b):
            continue
        q = overlap(b, np.repeat([gt['boxes'][pos]], len(b), axis=0))
        # Target-unary pre-gate scores are available for context; fallback has
        # its own original score and must not manufacture context probabilities.
        scores = d.get('target_all_scores')
        i = int(np.argmax(np.asarray(scores))) if scores is not None and len(scores) else None
        out.append(dict(view=view, position=pos, all_pool_best_IoU=float(q.max()),
            top_target_IoU=float(q[i]) if i is not None else None,
            accepted=probe['accepted'], reason=probe['reason'], candidates=len(b)))
    return out


def spatial_rows(p):
    wanted = [r for rr in p['rows'].values() for r in rr if 'core' in r['roles'] or 'guard' in r['roles']]
    assert all(existing(dest('spatial', r['key'])) for r in wanted), 'spatial_incomplete'
    labels = labels_for(wanted); flags = annotation_flags(wanted, labels); out = []
    for r in wanted:
        path = dest('spatial', r['key']); x = load(path); gt = labels[r['key']]
        ex = x['expert']; arms = {}; fm = {}; quality = {}; masks = {}; trajectories = {}
        for name, y in x['controls'].items():
            m, q, mm = spatial_metric(y['boxes'], gt, x, ex['observed4'], ex['anchors']['weak'], flags[r['key']])
            arms[name] = m; masks[name] = mm
        for name, y in x['fits'].items():
            observed = ex['observed8'] if name == 'B3' else ex['observed4']
            m, q, mm = spatial_metric(y['final']['boxes'], gt, x, observed, y['anchors'], flags[r['key']])
            arms[name] = m; masks[name] = mm; fm[name] = fit_diagnostics(y)
            quality[name] = teacher_quality(y['anchors'], gt, x['frame_ids'])
            if name == 'C0':
                for k in (0, 1, 3, 5, 10):
                    candidates = [t for t in y['path'] if t['step'] <= k]
                    chosen = min(candidates, key=lambda t: t['loss'])
                    arms['prefix'+str(k)] = spatial_metric(chosen['boxes'], gt, x, observed, y['anchors'], flags[r['key']])[0]
                for t in y['path']:
                    m = spatial_metric(t['boxes'], gt, x, observed, y['anchors'], flags[r['key']])[0]
                    trajectories[t['step']] = dict(pseudo_loss=t['parts']['spatial'], **m)
        arms['A3'] = arms['C0']; arms['B2'] = arms['C0']
        if all('C3_'+str(s) in arms for s in SEEDS):
            # Repeat conditions are averaged within the existing query/source,
            # including degenerate one-anchor episodes; no n inflation.
            arms['C3_mean'] = {m: float(np.mean([arms['C3_'+str(s)][m] for s in SEEDS]))
                if all(arms['C3_'+str(s)].get(m) is not None for s in SEEDS) else None for m in SPACE_METRICS}
        current = ex['anchors']['weak']; weights = [a['weight'] for a in current]
        active_weights = len(weights) >= 2 and max(weights)-min(weights) > 1e-12
        fusion = []
        for a, b in zip(ex['anchors']['weak'], ex['anchors']['original_member']):
            assert a['position'] == b['position'] and a['weight'] == b['weight']
            j = a['position']
            if gt['valid'][j]:
                qs = overlap(np.array([a['box'], b['box']]), np.repeat([gt['boxes'][j]], 2, axis=0))
                fusion.append(dict(position=j, mean_IoU=float(qs[0]), original_member_IoU=float(qs[1])))
        common_teacher = {}
        for a, b in [('C0', 'B0'), ('C0', 'B1'), ('C0', 'B3'), ('C0', 'BF')]:
            if b not in x['fits']:
                continue
            ta, tb = x['fits'][a]['anchors'], x['fits'][b]['anchors']
            keep = {z['position'] for z in ta} & {z['position'] for z in tb}
            qa1 = teacher_quality([z for z in ta if z['position'] in keep], gt, x['frame_ids'])
            qb1 = teacher_quality([z for z in tb if z['position'] in keep], gt, x['frame_ids'])
            common_teacher[a+' - '+b] = dict(common_positions=sorted(keep), first=qa1, second=qb1,
                delta_IoU=delta(qa1['mean_IoU'], qb1['mean_IoU']))
        out.append(dict(key=r['key'], source=r['source'], group=r['group'], roles=r['roles'], in_F34=r['in_F34'], caption=r['input']['caption'],
            frame_ids=x['frame_ids'], native_interval=x['native_indices'], GT_interval=gt['interval'],
            event_fraction=(gt['interval'][1]-gt['interval'][0])/(x['frame_ids'][-1]+1-x['frame_ids'][0]),
            arms=arms, fits=fm, masks=masks, flags=flags[r['key']], observation_positions=ex['observed4'],
            eight_positions=ex['observed8'], teacher_quality=quality, pregate=pregate_quality(ex, gt),
            common_teacher_quality=common_teacher, fusion_same_support=fusion,
            weights_active=active_weights, zero_anchor=not current, no_usable_phrase=ex['no_phrase'],
            actual_observed_physical_positions=len(ex['observed4']), trajectory_metrics=trajectories,
            costs={**x['costs'], 'episode_seconds': x['seconds'], 'new_DINO': ex['new_DINO'],
                   'expert_reused_calls': len(ex['reused']), 'expert_receipts': [z['receipt'] for z in ex['observations'].values()]},
            input_path=r['input']['video_path'], raw_path=str(path), raw_sha256=sha(path),
            native_logits_exact=x['actual_temporal_logits_unchanged'], parent_exact=x['fits']['C0'].get('parent_exact'),
            annotation_note='HC per-box provenance unknown; Vid raw generated/tracker retained and never invalidated'))
    return out


def temporal_metric(boxes, gt, ids, iv, fps):
    m, _ = checked_score(boxes, gt, ids, iv)
    lo, hi = ids[iv[0]], ids[iv[1]]+1; g, h = gt['interval']; duration = ids[-1]+1-ids[0]
    m.update(start_error_seconds=abs(lo-g)/fps, end_error_seconds=abs(hi-h)/fps,
             start_error_normalized=abs(lo-g)/duration, end_error_normalized=abs(hi-h)/duration)
    return m


def temporal_select():
    p = plan(); rows = [r for rr in p['rows'].values() for r in rr if 'development' in r['roles']]
    assert len(rows) == 16 and all(existing(dest('temporal', r['key'])) for r in rows), 'dev_incomplete'
    labels = labels_for(rows)
    values = []
    for r in rows:
        path = dest('temporal', r['key']); x = load(path); gt = labels[r['key']]
        m = {n: checked_score(x['systems'][n]['boxes'], gt, x['frame_ids'], x['systems'][n]['indices'])[0] for n in ['original', 'mapped']}
        values.append(dict(key=r['key'], group=r['group'], cohort=r['key'].split(':')[0], arms=m, raw_sha256=sha(path)))
    by = {n: {c: float(np.mean([r['arms'][n]['vIoU_corrected'] for r in values if r['cohort'] == c])) for c in p['rows']} for n in ('original', 'mapped')}
    means = {n: float(np.mean(list(v.values()))) for n, v in by.items()}
    chosen = max(['original', 'mapped'], key=lambda n: (means[n], -['original', 'mapped'].index(n)))
    write(OUT/'TEMPORAL_SELECTION.json', dict(selected_family=chosen, rule=p['temporal_selection'], means=means, by_direction=by,
        rows=values, sources=16, GT_selection=True, GT_online=False, evaluation_sources_used=0,
        created=time.time(), lock_sha256=sha(OUT/'LOCK.json'), labels_used_keys=[r['key'] for r in rows]))
    print('TEMPORAL_DEV_SELECTION', chosen, means, flush=True)


def temporal_rows(p):
    select = read(OUT/'TEMPORAL_SELECTION.json'); chosen = select['selected_family']
    rows = [r for rr in p['rows'].values() for r in rr if 'temporal' in r['roles']]
    assert len(rows) == 48 and all(existing(dest('temporal', r['key'])) for r in rows), 'time_incomplete'
    labels = labels_for(rows); out = []
    for r in rows:
        path = dest('temporal', r['key']); x = load(path); gt = labels[r['key']]; ids = x['frame_ids']; arms = {}; changes = {}; fd = {}
        boxes = x['native_boxes']; spatial = x['spatial_boxes']; native = x['native_indices']; fps = r['input']['fps']
        arms['native'] = temporal_metric(boxes, gt, ids, native, fps)
        arms['S0'] = temporal_metric(spatial, gt, ids, native, fps)
        for name, z in x['fits'].items():
            iv = z['final']['indices']; arms[name] = temporal_metric(boxes, gt, ids, iv, fps)
            arms['system_'+name] = temporal_metric(spatial, gt, ids, iv, fps)
            assert arms['system_'+name]['sIoU'] == arms['S0']['sIoU']
            for k in ('boxes',):
                assert torch.equal(x['systems'][name][k], spatial)
            a, b = native; c, d = iv
            kind = 'unchanged' if iv == native else 'expansion' if c <= a and d >= b else 'shrink' if c >= a and d <= b else 'shift'
            dv = arms[name]['vIoU_corrected']-arms['native']['vIoU_corrected']; dt = arms[name]['tIoU']-arms['native']['tIoU']
            changes[name] = dict(kind=kind, native_indices=native, adapted_indices=iv, delta_v=dv, delta_t=dt,
                time_outcome='good' if dt > .001 else 'failure' if dt < -.001 else 'neutral',
                teacher_native_delta_t=(temporal_metric(boxes, gt, ids, x['teacher_controls'][name]['indices'], fps)['tIoU']-
                                       arms['native']['tIoU']) if name in x['teacher_controls'] else None)
            fd[name] = fit_diagnostics(z)
            fd[name]['reused_from_F34'] = name in x['reused']
        for name, tc in x['teacher_controls'].items():
            arms['teacher_'+name] = temporal_metric(boxes, gt, ids, tc['indices'], fps)
            arms['teacher_system_'+name] = temporal_metric(spatial, gt, ids, tc['indices'], fps)
        arms['S1'] = arms['system_ensemble']; arms['S2'] = arms['system_'+chosen]; arms['S3'] = arms['teacher_system_'+chosen]
        fraction = (gt['interval'][1]-gt['interval'][0])/(ids[-1]+1-ids[0])
        length = 'short' if fraction < .25 else 'medium' if fraction < .75 else 'long'
        nm = arms['native']; ntype = 'already_accurate' if nm['tIoU'] >= .5 else 'undercoverage' if nm['temporal_recall'] < nm['temporal_precision'] else 'overcoverage'
        coords = None
        if x['coordination']:
            j = x['coordination']; jf = j['fits']; shared_prop_errors = []
            for a, b in zip(jf['grouped']['path'], jf['sum']['path']):
                if a.get('proposal') is None or b.get('proposal') is None:
                    continue
                shared_prop_errors.append(dict(step=a['step'], max_error=max(float((a['proposal'][n]-b['proposal'][n]).abs().max()) for n in a['proposal']),
                    grouped_accepted=a.get('accepted_groups'), sum_accepted=b['accepted'], sum_trials=b['trials']))
            private_prop_errors = []
            for a in jf['grouped']['path']:
                if a.get('proposal') is None:
                    continue
                i = a['step']; maxerr = 0.; checked_groups = []
                for part, group in [('private_T', 'head'), ('private_S', 'spatial')]:
                    # An empty spatial episode has only step 0, while the
                    # temporal group can still take five genuine proposals.
                    pp = next((z.get('proposal') for z in jf[part]['path'] if z['step'] == i), None)
                    if pp:
                        checked_groups.append(group)
                        maxerr = max(maxerr, max(float((a['proposal'][n]-v).abs().max()) for n, v in pp.items()))
                private_prop_errors.append(dict(step=i, max_error=maxerr, directly_saved_groups=checked_groups))
            jresults = {n: temporal_metric(z['final']['boxes'], gt, ids, z['final']['indices'], fps) for n, z in jf.items()}
            coords = dict(cross_gradients=j['cross_gradients'], finite_forward_cross_zero=j['finite_forward_cross_zero'],
                grouped_state_equals_private=j['grouped_state_equals_private'], grouped_vs_sum_proposals=shared_prop_errors,
                grouped_vs_private_proposals=private_prop_errors, metrics=jresults,
                fits={n: fit_diagnostics(z) for n, z in jf.items()}, full_traces_path=str(path))
        out.append(dict(key=r['key'], source=r['source'], group=r['group'], roles=r['roles'], caption=r['input']['caption'],
            frame_ids=ids, GT_interval=gt['interval'], event_fraction=fraction, length_group=length, native_error_group=ntype,
            arms=arms, changes=changes, fits=fd, coordination=coords,
            spatial_coordinates_exact=True, seed_fixed_I0=True, map_roundoff=x['map_rebuild_max_roundoff'],
            raw_path=str(path), raw_sha256=sha(path), costs=x['costs'], episode_seconds=x['seconds'], input_path=r['input']['video_path']))
    return out


def temporal_summary(ts, p):
    result = {}
    comparisons = [(n, 'native') for n in ['ensemble', 'original', 'mapped', 'duplicate', 'wrong']]
    comparisons += [(n, 'teacher_'+n) for n in ['ensemble', 'original', 'mapped']]
    comparisons += [('original', 'ensemble'), ('mapped', 'ensemble'), ('original', 'wrong'), ('mapped', 'wrong'),
                    ('S1', 'S0'), ('S2', 'S0'), ('S2', 'S1'), ('S2', 'S3')]
    for c in p['rows']:
        tr = [r for r in ts if r['key'].startswith(c+':')]
        for scope in ('development', 'panel'):
            rr = [r for r in tr if scope in r['roles']]
            names = sorted(set.union(*(set(r['arms']) for r in rr))) if rr else []
            result[c+'/'+scope] = summary(rr, names, TEMP_METRICS, comparisons)
            if scope == 'panel':
                for field, groups in [('length_group', ['short', 'medium', 'long']),
                                      ('native_error_group', ['already_accurate', 'undercoverage', 'overcoverage'])]:
                    for group in groups:
                        sub = [r for r in rr if r[field] == group]
                        result[c+'/'+group] = summary(sub, names, TEMP_METRICS, comparisons)
    return result


def temporal_only():
    p = plan(); ts = temporal_rows(p)
    write(OUT/'TEMPORAL_SOURCE_RESULTS.json', dict(rows=ts, summary=temporal_summary(ts, p),
        selection=read(OUT/'TEMPORAL_SELECTION.json'), created=time.time(), labels_online=False,
        historical_exposure=True, scoring_code_sha256=sha(Path(__file__)), lock_sha256=sha(OUT/'LOCK.json')))
    print('TEMPORAL_ANALYSIS_COMPLETE', len(ts), flush=True)


def analyze():
    p = plan()
    assert (OUT/'TEMPORAL_SELECTION.json').exists(), 'select_on_dev_before_eval'
    ss = spatial_rows(p); ts = temporal_rows(p)
    results = {'spatial': {}, 'temporal': {}}
    spatial_comparisons = [('A3', 'A0'), ('A3', 'A2'), ('A3', 'A1'), ('A3', 'A4'),
        ('B2', 'B1'), ('B2', 'B0'), ('B2', 'B3'), ('C0', 'BF'), ('C0', 'C1'), ('C0', 'C2'), ('C0', 'C3_mean'),
        ('C0', 'D_rho'), ('C0', 'D_gamma0'), ('C0', 'D_initial'), ('C0', 'E_query'), ('C0', 'E_ln')]
    for c in p['rows']:
        sr = [r for r in ss if r['key'].startswith(c+':')]; tr = [r for r in ts if r['key'].startswith(c+':')]
        for scope in ('core', 'panel', 'guard'):
            rr = [r for r in sr if scope in r['roles']]
            names = sorted(set.union(*(set(r['arms']) for r in rr))) if rr else []
            results['spatial'][c+'/'+scope] = summary(rr, names, SPACE_METRICS, spatial_comparisons)
        for name,member in [('core_in_F34',True),('core_outside_F34',False)]:
            rr=[r for r in sr if 'core' in r['roles'] and r['in_F34']==member]
            results['spatial'][c+'/'+name]=summary(rr,['A0','A1','A2','A3','A4'],SPACE_METRICS,
                [('A3','A0'),('A3','A2')])
        rr = [r for r in sr if 'panel' in r['roles'] and r['weights_active']]
        results['spatial'][c+'/weights_active'] = summary(rr, ['C0', 'C1', 'C2', 'C3_mean'], SPACE_METRICS,
            [('C0', 'C1'), ('C0', 'C2'), ('C0', 'C3_mean')])
    results['temporal'] = temporal_summary(ts, p)
    write(OUT/'ALL_SOURCE_RESULTS.json', dict(spatial_rows=ss, temporal_rows=ts, summary=results,
        cohort_counts=p['counts'], temporal_selection=read(OUT/'TEMPORAL_SELECTION.json'),
        historical_exposure=True, GT_online=False, aggregation='Within-query seed average then source macro; one query/source, never sum overlapping panels',
        uncertainty=p['uncertainty'], lock_sha256=sha(OUT/'LOCK.json'), created=time.time()))
    print('ANALYSIS_COMPLETE', len(ss), len(ts), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['select', 'temporal', 'analyze']); args = ap.parse_args()
    dict(select=temporal_select, temporal=temporal_only, analyze=analyze)[args.stage]()
