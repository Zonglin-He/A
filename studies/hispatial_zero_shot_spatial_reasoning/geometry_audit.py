"""Replay official scores and audit spatial failure signatures after sealing."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
from PIL import Image

from download import DATA, STUDY, read_json, sha256, write_json
from geometry_cohort import CONFIG, RUN, TOPIC, read_jsonl, verified_baseline
import score


CONDITIONS = ('aligned', 'rgb_only', 'mismatched_xyz')
PAIRWISE = (('aligned', 'rgb_only'), ('aligned', 'mismatched_xyz'), ('rgb_only', 'mismatched_xyz'))
MIRROR = {'left': 'right', 'right': 'left', 'front': 'front', 'back': 'back'}
FROZEN_PARSERS = ('correct_truncate', 'correct_regex')


def keyed(rows):
    result = {r['id']: r for r in rows}
    assert len(result) == len(rows)
    return result


def verify():
    inputs = read_json(RUN / 'inputs_seal.json')
    pred = read_json(RUN / 'predictions_seal.json')
    assert inputs['config_sha256'] == sha256(CONFIG)
    assert pred['input_seal_sha256'] == sha256(RUN / 'inputs_seal.json')
    assert pred['ground_truth_read'] is False and inputs['labels_in_inference_manifest'] is False
    for role in (inputs, pred):
        for filename, digest in role['files'].items():
            assert sha256(RUN / filename) == digest, filename
    maps = read_json(RUN / 'pointmap_sha256.json')
    frames = read_json(RUN / 'frames.json')
    assert pred['pointmaps'] == len(maps) == len(frames) and set(maps) == set(frames)
    for key, digest in maps.items():
        assert sha256(RUN / 'pointmaps' / f'{key}.npy') == digest, key
    for benchmark, info in inputs['baseline'].items():
        directory, _, _ = verified_baseline(benchmark)
        assert sha256(directory / 'score_seal.json') == info['score_seal']
        assert sha256(directory / 'predictions_seal.json') == info['prediction_seal']
        dataset_source = DATA / benchmark / 'source.json'
        assert sha256(dataset_source) == info['dataset_source']
        for file in read_json(dataset_source)['files']:
            assert sha256(dataset_source.parent / file['file']) == file['sha256']
    return inputs, pred, frames


def official_results(official_path, inputs, cases, meta, controls, anchors):
    result = {}
    anchor_summary = {}
    official_hashes = {}
    for benchmark in inputs['baseline']:
        directory, _, _ = verified_baseline(benchmark)
        original = keyed(read_jsonl(directory / 'predictions.jsonl'))
        complete_prompts = {r['id']: r['prompt'] for r in read_jsonl(directory / 'inputs.jsonl')}
        focused = {r['dataset_id']: r for r in meta.values() if r['benchmark'] == benchmark}
        assert all(cases[r['id']]['prompt'] == complete_prompts[k] for k, r in focused.items())
        anchors_here = [r for r in anchors if r['id'].startswith(benchmark + '/')]
        anchor_summary[benchmark] = {'compared': len(anchors_here),
                                     'matches': sum(r['aligned'] == original[meta[r['id']]['dataset_id']]['response']
                                                    for r in anchors_here)}
        assert len(anchors_here) == 24
        module, script_hash = score.import_official(benchmark, official_path)
        official_hashes[benchmark] = script_hash
        if benchmark == '3dsrbench':
            selected = [(i, row) for i, row in score.tsv_rows() if f'tsv/{i}' in focused]
            assert len(selected) == len(focused)
        else:
            selected = None
        result[benchmark] = {}
        for condition in CONDITIONS:
            work = RUN / 'official_replay' / benchmark / condition
            (work / 'official_replay').mkdir(parents=True, exist_ok=True, mode=0o700)
            generations = {**original, **{k: {'response': controls[r['id']][condition]} for k, r in focused.items()}}
            if benchmark == '3dsrbench':
                with patch.object(score, 'tsv_rows', return_value=iter(selected)):
                    rows, groups, metrics = score.score_sr(module, work, generations, complete_prompts)
                assert len(groups) == 175
                assert all(abs(metrics[rule]['accuracy'] - metrics[rule]['by_subcategory'][TOPIC]) < 1e-12
                           for rule in ('truncate', 'regex'))
            else:
                rows, _, all_metrics = score.score_emb(module, work, generations, complete_prompts)
                rows = [r for r in rows if r['id'] in focused]
                metrics = {'replay_all_rows': all_metrics,
                           'non_focus_predictions': 'reused aligned baseline for every condition'}
            assert {r['id'] for r in rows} == set(focused) and len(rows) == len(focused)
            write_json(work / 'official_metrics.json', score.sanitize(metrics))
            result[benchmark][condition] = {r['id']: r for r in rows}
        prior = read_json(directory / 'score_seal.json')
        prior_path = directory / 'scored_rows.jsonl'
        assert sha256(prior_path) == prior['derived_artifacts']['scored_rows.jsonl']
        same = {r['id']: r for r in read_jsonl(prior_path) if r['id'] in focused}
        assert same.keys() == result[benchmark]['aligned'].keys()
        result[benchmark]['previous_aligned'] = same
    return result, anchor_summary, official_hashes


def select_letter(response, allowed):
    value = response.replace('<eos>', '').strip()
    match = re.fullmatch(r'\s*\(?([' + allowed + r'])\)?[.\s]*', value, re.I)
    return match.group(1).upper() if match else None


def ordered_letters(response, count):
    allowed = 'ABCDEF'[:count]
    val = response.replace('<eos>', '').strip()
    if re.search(r'[^' + allowed + r'\s,.;>\-|]', val):
        return None
    order = re.findall('[' + allowed + ']', val)
    return order if len(order) == count and set(order) == set(allowed) else None


def combine(scored, meta, case_rows):
    paired = []
    for item in meta.values():
        official = {name: scored[item['benchmark']][name][item['dataset_id']] for name in CONDITIONS}
        previous = scored[item['benchmark']]['previous_aligned'][item['dataset_id']]
        assert previous['official_answer'] == official['aligned']['official_answer']
        row = {**item, 'key': case_rows[item['id']]['key'], 'prompt': case_rows[item['id']]['prompt'],
               'previous_aligned': {k: v for k, v in previous.items() if k in ('response', 'correct', *FROZEN_PARSERS)},
               'official_answer': official['aligned']['official_answer']}
        if item['benchmark'] == '3dsrbench':
            row['sides'] = {}
            row['canonical_sides'] = {}
            for condition, res in official.items():
                letter = res['parsed_regex_option']
                side = row['options']['ABCD'.index(letter)].lower() if letter else None
                assert side is None or side in MIRROR
                row['sides'][condition] = side
                row['canonical_sides'][condition] = MIRROR[side] if row['flip'] and side else side
            gt = row['options']['ABCD'.index(row['official_answer'])].lower()
            row['canonical_answer'] = MIRROR[gt] if row['flip'] else gt
        row['conditions'] = {condition: {k: v for k, v in res.items() if k in (
            'response', 'correct', 'correct_regex', 'correct_truncate', 'parsed_leading_option', 'parsed_regex_option')}
                             for condition, res in official.items()}
        paired.append(row)
    return paired


def sr_group_metrics(paired, probes, probe_meta):
    groups = defaultdict(list)
    for row in paired:
        if row['benchmark'] == '3dsrbench':
            groups[row['group']].append(row)
    cam = {r['case_id']: r for r in probe_meta if r['task'] == 'sr_camera_side'}
    results = []
    for group, variants in groups.items():
        base = next(r for r in variants if r['variant'] == group)
        gt = {r['canonical_answer'] for r in variants}
        norm_truth = next(iter(gt)) if len(gt) == 1 else None
        face = cam[base['id']]
        response = probes[face['id']]['aligned']
        camera_letter = select_letter(response, 'ABCD')
        camera_side = face['options']['ABCD'.index(camera_letter)] if camera_letter else None
        out = dict(group=group, scene=base['scene'], key=base['key'], base_case=base['id'],
                   subject=face['subject'], reference=face['reference'], canonical_gt=norm_truth,
                   rows=len(variants), camera_side_probe=camera_side, camera_side_probe_raw=response,
                   previous_aligned={p: all(r['previous_aligned'][p] for r in variants) for p in FROZEN_PARSERS},
                   flipped=any(r['flip'] for r in variants), conditions={})
        # The same view may be repeated with different choice orders.
        by_view = defaultdict(list)
        for row in variants:
            by_view[row['key']].append(row)
        for condition in CONDITIONS:
            view_sides = [set(r['canonical_sides'][condition] for r in replies) for replies in by_view.values()]
            all_sides = set().union(*view_sides)
            assessment = {parser: all(r['conditions'][condition][parser] for r in variants)
                          for parser in FROZEN_PARSERS}
            if assessment['correct_regex']:
                failure = 'correct'
            elif None in all_sides:
                failure = 'unparsed_variant'
            elif any(len(items) > 1 for items in view_sides):
                failure = 'same_view_choice_order_disagreement'
            elif len(all_sides) > 1:
                failure = 'mirror_disagreement'
            else:
                failure = 'stable_wrong_side'
            assessment.update(failure=failure, original_side=base['sides'][condition],
                              original_view_correct=all(r['conditions'][condition]['correct_regex'] for r in variants if not r['flip']),
                              each_view_has_stable_side=all(len(items) == 1 and None not in items for items in view_sides),
                              canonical_sides=sorted(v for v in all_sides if v))
            out['conditions'][condition] = assessment
        results.append(out)
    return results


def option_boxes(raw, options):
    boxes = defaultdict(list)
    for obj in raw['objects']:
        boxes[obj['name'].casefold().strip()].append(obj['bbox'])
    return [boxes.get(name.casefold().strip(), [None])[0] if len(boxes.get(name.casefold().strip(), [])) == 1 else None
            for name in options]


def box_depth(pointmap, box, size):
    if box is None:
        return None
    x0, y0, w, h = box
    W, H = size
    ny, nx = pointmap.shape[-2:]
    z, mask = pointmap[2], pointmap[3]
    z_ok = (mask > 0) & (z > 0) & (z < 250) & np.isfinite(z)
    values = []
    for trim in (0, .25):
        xs = slice(max(0, int(np.floor((x0 + trim*w)*nx/W))), min(nx, int(np.ceil((x0 + (1-trim)*w)*nx/W))))
        ys = slice(max(0, int(np.floor((y0 + trim*h)*ny/H))), min(ny, int(np.ceil((y0 + (1-trim)*h)*ny/H))))
        patch = z[ys, xs]
        eligible = z_ok[ys, xs]
        values.append({'mean_z': float(patch[eligible].mean(dtype=np.float64)),
                       'valid_fraction': float(eligible.mean()), 'count': int(eligible.sum())} if eligible.any() else None)
    return values


def grid_check(cell, box, size):
    if cell is None or box is None:
        return 'unknown'
    x, y, w, h = box
    W, H = size
    if not (w > 0 and h > 0):
        return 'unknown'
    k = 'ABCDEFGHI'.index(cell)
    cx, cy = k % 3, k // 3
    cells_x = (x*3/W, (x+w)*3/W)
    cells_y = (y*3/H, (y+h)*3/H)
    if cells_x[1] <= cx or cells_x[0] >= cx+1 or cells_y[1] <= cy or cells_y[0] >= cy+1:
        return 'disjoint'
    if cx <= (x+w/2)*3/W < cx+1 and cy <= (y+h/2)*3/H < cy+1:
        return 'contains_box_center'
    return 'overlaps_box'


def emb_depth_metrics(paired, frames, probes, probe_meta):
    raw = read_json(DATA / 'embspatial' / 'embspatial_bench.json')
    links = defaultdict(list)
    for row in probe_meta:
        if row['task'].startswith('emb_'):
            links[row['case_id']].append(row)
    geo = {}
    outputs = []
    for row in paired:
        if row['benchmark'] != 'embspatial':
            continue
        key = row['key']
        if key not in geo:
            origin = frames[key]
            path = RUN.parent / 'official_baseline_20261121/embspatial/images' / origin['image_sha256']
            with Image.open(path) as img:
                size = img.size
            geo[key] = (size, np.load(RUN / 'pointmaps' / f'{key}.npy', mmap_mode='r', allow_pickle=False))
        size, pointmap = geo[key]
        indexed = int(row['dataset_id'].split('/')[1])
        boxes = option_boxes(raw[indexed], row['options'])
        measures = [box_depth(pointmap, b, size) for b in boxes]
        orders = [None, None]
        for j in range(2):
            if all(m is not None and m[j] is not None for m in measures):
                pairs = sorted((m[j]['mean_z'], letter) for letter, m in zip('ABCDEF', measures))
                if len({p[0] for p in pairs}) == len(pairs):
                    orders[j] = [p[1] for p in pairs]
        gold = row['official_answer']
        extreme = -1 if row['topic'] == 'far' else 0
        pred = row['conditions']['aligned']['parsed_leading_option']
        gap = None
        if orders[0]:
            near, far = (measures['ABCDEF'.index(letter)][0]['mean_z'] for letter in (orders[0][0], orders[0][-1]))
            winner = far if extreme == -1 else near
            runner_up = orders[0][-2 if extreme == -1 else 1]
            gap = abs(winner - measures['ABCDEF'.index(runner_up)][0]['mean_z']) / winner
        order_probe = None
        order_present = False
        localizers = []
        for item in links[row['id']]:
            response = probes[item['id']]['aligned']
            if item['task'] == 'emb_order':
                order_present = True
                order_probe = ordered_letters(response, len(boxes))
            else:
                cell = select_letter(response, 'ABCDEFGHI')
                j = item['option']
                localizers.append(dict(option='ABCDEF'[j], name=row['options'][j], response=response, cell=cell,
                                       box=boxes[j], localization=grid_check(cell, boxes[j], size)))
        outputs.append(dict(id=row['id'], topic=row['topic'], source=row['source'], scene=row['scene'], key=key,
                            options=row['options'], official_answer=gold, aligned_prediction=pred,
                            aligned_correct=row['conditions']['aligned']['correct'], size=list(size), boxes=boxes,
                            depth=measures, full_box_order=orders[0], center_box_order=orders[1],
                            bbox_moge_prediction=orders[0][extreme] if orders[0] else None,
                            bbox_center_prediction=orders[1][extreme] if orders[1] else None,
                            relative_extreme_gap=gap,
                            opposite_extreme=(pred == orders[0][-1-extreme]) if (orders[0] and pred) else None,
                            rank_probe_present=order_present, rank_probe_order=order_probe,
                            rank_probe_prediction=order_probe[extreme] if order_probe else None,
                            localization_probes=localizers))
    return outputs


def paired_stats(rows, metric, left, right):
    a = np.array([r['conditions'][left][metric] for r in rows], dtype=np.int64)
    b = np.array([r['conditions'][right][metric] for r in rows], dtype=np.int64)
    units = defaultdict(list)
    for i, row in enumerate(rows):
        units[(row.get('source'), row['scene'])].append(i)
    delta = b-a
    d = np.array([delta[ids].sum() for ids in units.values()])
    n = np.array([len(ids) for ids in units.values()])
    rng = np.random.default_rng(20261121)
    ci = []
    for _ in range(10):
        sample = rng.integers(len(units), size=(200, len(units)))
        ci.extend((d[sample].sum(axis=1) * 100 / n[sample].sum(axis=1)).tolist())
    return dict(from_condition=left, to_condition=right, metric=metric, n=len(rows),
                from_correct=int(a.sum()), to_correct=int(b.sum()), both_right=int((a & b).sum()),
                rescued=int(((a == 0) & (b == 1)).sum()), harmed=int(((a == 1) & (b == 0)).sum()),
                both_wrong=int(((a == 0) & (b == 0)).sum()), delta_pp=float(delta.mean()*100),
                scene_clusters=len(units), ci95_pp=np.percentile(ci, [2.5, 97.5]).tolist())


def emb_summary(geo):
    counts = {}
    for relation in ('close', 'far', 'both'):
        only = [r for r in geo if relation == 'both' or relation == r['topic']]
        counts[relation] = {}
        for ok in (False, True):
            rows = [r for r in only if r['aligned_correct'] == ok]
            loc = [x for r in rows for x in r['localization_probes']]
            ref_options = [x for r in rows for x in r['localization_probes'] if x['option'] == r['official_answer']]
            favored = [x for r in rows for x in r['localization_probes'] if x['option'] == r['aligned_prediction']]
            ready_rank = [r for r in rows if r['rank_probe_prediction'] is not None]
            gaps = [r['relative_extreme_gap'] for r in rows if r['relative_extreme_gap'] is not None and
                    r['bbox_moge_prediction'] == r['official_answer']]
            counts[relation]['aligned_right' if ok else 'aligned_wrong'] = dict(
                eligible=len(rows), box_rank_available=sum(r['bbox_moge_prediction'] is not None for r in rows),
                missing_option_box=sum(any(box is None for box in r['boxes']) for r in rows),
                no_valid_full_box_depth_with_all_boxes=sum(all(box is not None for box in r['boxes']) and
                                                          any(d[0] is None for d in r['depth']) for r in rows),
                box_rank_same_as_gold=sum(r['bbox_moge_prediction'] == r['official_answer'] for r in rows),
                full_and_center_same_as_gold=sum(r['bbox_moge_prediction'] == r['bbox_center_prediction'] == r['official_answer']
                                                 for r in rows),
                median_relative_gap_when_box_depth_matches_gold=float(np.median(gaps)) if gaps else None,
                opposite_box_extreme=sum(r['opposite_extreme'] is True for r in rows),
                rank_probes=sum(r['rank_probe_present'] for r in rows), rank_probes_parsed=len(ready_rank),
                rank_probes_correct=sum(r['rank_probe_prediction'] == r['official_answer'] for r in ready_rank),
                rank_agrees_with_same_question_answer=sum(r['rank_probe_prediction'] == r['aligned_prediction'] for r in ready_rank),
                probe_options=dict(Counter(x['localization'] for x in loc)),
                probe_gold_options=dict(Counter(x['localization'] for x in ref_options)),
                probe_chosen_options=dict(Counter(x['localization'] for x in favored)),
                same_full_and_center_correct_rank_probe_wrong=sum(
                    r['bbox_moge_prediction'] == r['bbox_center_prediction'] == r['official_answer']
                    and r['rank_probe_prediction'] != r['official_answer'] for r in ready_rank))
    return counts


def sr_summary(groups):
    signatures = {}
    for name in CONDITIONS:
        sig = Counter(r['conditions'][name]['failure'] for r in groups)
        stable = [r for r in groups if r['conditions'][name]['failure'] == 'stable_wrong_side']
        comparable = [r for r in stable if r['camera_side_probe'] and r['canonical_gt']
                      and r['camera_side_probe'] != r['canonical_gt']]
        signatures[name] = dict(failures=dict(sig),
                                original_view_correct=sum(r['conditions'][name]['original_view_correct'] for r in groups),
                                original_correct_but_flip_fails=sum(r['conditions'][name]['original_view_correct'] and
                                    not r['conditions'][name]['correct_regex'] for r in groups),
                                wrong_stable_camera_comparable=len(comparable),
                                wrong_stable_matching_camera_probe=sum(r['conditions'][name]['original_side'] == r['camera_side_probe']
                                                                     for r in comparable))
    return signatures


def compare_previous(paired, groups):
    cohorts = [(TOPIC, groups, parser) for parser in FROZEN_PARSERS]
    cohorts.extend((topic, [r for r in paired if r['topic'] == topic], 'correct') for topic in ('close', 'far'))
    reports = []
    for topic, rows, metric in cohorts:
        previous = [r['previous_aligned'][metric] for r in rows]
        repeated = [r['conditions']['aligned'][metric] for r in rows]
        reports.append(dict(topic=topic, metric=metric, n=len(rows), previous_correct=sum(previous), repeated_correct=sum(repeated),
                            recovered=sum(b and not a for a, b in zip(previous, repeated)), lost=sum(a and not b for a, b in zip(previous, repeated))))
    return reports


def read_blind_review(sr, probes, meta):
    panel = RUN / 'blind_3dsr'
    manual_path = panel / 'annotations.json'
    if not manual_path.exists():
        return None
    assert manual_path.stat().st_mtime_ns < (RUN / 'predictions_seal.json').stat().st_mtime_ns
    assert sha256(manual_path) == (panel / 'ANNOTATION_SHA256.txt').read_text().split()[0]
    manual = read_json(manual_path)
    links = read_json(panel / 'lookup.json')
    assert {r['card_id'] for r in links} == manual.keys()
    grouped = {r['base_case']: r for r in sr}
    locations = {(r['case_id'], r['role']): select_letter(probes[r['id']]['aligned'], 'ABCDEFGHI')
                 for r in meta if r['task'] == 'sr_role_grid'}
    review = []
    for link in links:
        case, card = link['case_id'], link['card_id']
        known, group = manual[card], grouped[case]
        subject, reference = (locations[case, role] for role in ('subject', 'reference'))
        verified = [prediction in known[role] if known[role] and prediction else None
                    for role, prediction in zip(('subject', 'reference'), (subject, reference))]
        current = group['conditions']['aligned']
        review.append(dict(card=card, case=case, known=known, subject_grid=subject, reference_grid=reference,
                           both_locatable=all(val is not None for val in verified), both_grid_compatible=all(verified),
                           any_grid_mismatch=False in verified, camera_matches=((group['camera_side_probe'] == known['toward_camera'])
                               if known['toward_camera'] and group['camera_side_probe'] else None),
                           camera_probe=group['camera_side_probe'], target_side=group['canonical_gt'], predicted_side=current['original_side'],
                           group_correct=current['correct_regex'], group_failure=current['failure']))
    score.jsonl(panel / 'joined_review.jsonl', review)
    counts = {}
    for passed in (False, True):
        items = [r for r in review if r['group_correct'] == passed]
        counts['aligned_right' if passed else 'aligned_wrong'] = dict(
            reviewed=len(items), both_locatable=sum(r['both_locatable'] for r in items),
            both_grid_compatible=sum(r['both_grid_compatible'] for r in items),
            any_grid_mismatch=sum(r['any_grid_mismatch'] for r in items),
            camera_verifiable=sum(r['camera_matches'] is not None for r in items),
            camera_matches=sum(r['camera_matches'] is True for r in items),
            wrong_prediction_is_verified_camera_side=sum(r['camera_matches'] is True and
                r['predicted_side'] == r['camera_probe'] and r['predicted_side'] != r['target_side'] for r in items))
    return counts


def run(official_path):
    inputs, pred, frames = verify()
    cases, meta, controls, probes = (keyed(read_jsonl(RUN / f'{name}.jsonl')) for name in (
        'cases', 'metadata', 'condition_predictions', 'probe_predictions'))
    assert cases.keys() == meta.keys() == controls.keys() and len(cases) == pred['task_rows_per_condition']
    probe_meta = read_jsonl(RUN / 'probe_metadata.jsonl')
    assert set(probes) == {r['id'] for r in probe_meta} and len(probes) == pred['probe_rows']
    assert all(isinstance(r[k], str) for r in controls.values() for k in CONDITIONS)
    assert all(isinstance(r['aligned'], str) for r in probes.values())
    anchors = read_jsonl(RUN / 'aligned_anchor.jsonl')
    scored, parity, official_hashes = official_results(official_path, inputs, cases, meta, controls, anchors)
    paired = combine(scored, meta, cases)
    sr = sr_group_metrics(paired, probes, probe_meta)
    geo = emb_depth_metrics(paired, frames, probes, probe_meta)
    blind = read_blind_review(sr, probes, probe_meta)
    scope = [(TOPIC, sr, metric) for metric in FROZEN_PARSERS] + [
        (kind, [r for r in paired if r['benchmark'] == 'embspatial' and (kind == 'emb_both' or r['topic'] == kind)], 'correct')
        for kind in ('close', 'far', 'emb_both')]
    metrics = [dict(topic=topic, **paired_stats(rows, metric, a, b)) for topic, rows, metric in scope for a, b in PAIRWISE]
    for name, rows in (('paired_rows', paired), ('emb_box_depth', geo), ('sr_groups', sr)):
        score.jsonl(RUN / f'{name}.jsonl', rows)
    result = dict(frozen=True, original_anchor_matches=parity, official_script_hashes=official_hashes,
                  prediction_seal_sha256=sha256(RUN / 'predictions_seal.json'),
                  inputs_seal_sha256=sha256(RUN / 'inputs_seal.json'),
                  sr_groups=len(sr), sr_gold_flip_normalized=sum(r['canonical_gt'] is not None for r in sr),
                  sr_splits=sr_summary(sr), emb_splits=emb_summary(geo),
                  previous_repeat_comparison=compare_previous(paired, sr), blind_3dsr_review=blind,
                  result_files={f: sha256(RUN / f) for f in ('paired_rows.jsonl', 'emb_box_depth.jsonl', 'sr_groups.jsonl')},
                  paired_metrics=metrics)
    write_json(RUN / 'audit_summary.json', result)
    with (RUN / 'paired_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metrics)
    print('Verified and scored', len(paired), 'official rows,', len(probes), 'auxiliary rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source', type=Path, required=True)
    run(parser.parse_args().official_source)
