"""Predict on pinned ordinary inputs, seal every arm, then evaluate independently."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = HERE.parent / '3eed_v1_frontend_simplification_20261119' / 'predictions_PRIVATE'
sys.path[:0] = [str(ROOT), str(HERE.parent / '3eed_v1_component_ablation_20261119')]
from score_ablation import (DIRECTIONS, PLATFORMS, many_iou, recording_intervals, save_csv,
                            score_boxes, sensor_truth, summarize)

SLOT_PROFILES = ('v1', 'wide8', 'role4', 'role_swap')
SUBSETS = ('all', 'reference_mention', 'direct_left_right', 'no_reference')


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, value):
    draft = path.with_suffix('.tmp')
    draft.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(draft, path)


def register(protocol):
    roster = ROOT / protocol['input_registry']
    pin = read(ROOT / protocol['input_seal'])
    assert digest(roster) == pin['input_sha256'] == protocol['input_sha256']
    assert not pin['FINAL_access'] and not pin['target_labels_used_for_selection']
    rows = read(roster)
    assert len(rows) == len({row['id'] for row in rows}) == pin['expressions']
    baseline = read(PREVIOUS / 'SEALED.json')
    assert digest(PREVIOUS / 'SEALED.json') == protocol['baseline_prediction_seal_sha256']
    assert baseline['n'] == protocol['direction_records'] and not baseline['FINAL_access']
    assert not baseline['target_labels_read']
    return rows, baseline


def implementation():
    runtime = ('candidate_frontend.py', 'ground.py', 'v1_decoder.py', 'v1_evidence.py',
               'CURRENT_METHOD.json', 'single_frame_sources.json')
    paths = [HERE / name for name in ('execute.py', 'mechanisms.py', 'protocol.json')]
    paths.extend(ROOT / 'crossplatform' / name for name in runtime)
    paths.append(HERE.parent / '3eed_v1_component_ablation_20261119' / 'score_ablation.py')
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def registered_direction(rows, baseline, source, target):
    name = source + '_to_' + target + '.npz'
    path = PREVIOUS / name
    assert digest(path) == baseline['files'][name]['sha256']
    expected = np.flatnonzero([row['platform'] == target for row in rows])
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ordinals'], expected)
        return expected, saved['boxes']


def sample(model, decoder, row, seed, old_box, arms):
    import torch
    from crossplatform import ground
    from crossplatform.candidate_frontend import WAYMO_VIEWS, sensor_frame_box
    from mechanisms import CandidateGeometry, ReadoutInputs, distinct_entity_relation

    target = row['platform']
    view = WAYMO_VIEWS[int(row['suffix'] or 0)] if target == 'waymo' else 'F'
    frame = dict(target=target, points=row['pointpath'], rgb=row['imagepath'],
                 calibration=row['metapath'], text=row['caption'], seed=seed, waymo_view=view)
    traces = {}
    make_evidence, make_roles = ground.decoder_evidence, ground.role_masks

    def trace_evidence(observed, native, *args):
        traces['native'] = native
        return make_evidence(observed, native, *args)

    def trace_roles(caption, graph, *args):
        traces['graph'] = graph
        return make_roles(caption, graph, *args)

    with (torch.inference_mode(),
          patch.object(ground, 'decoder_evidence', trace_evidence),
          patch.object(ground, 'role_masks', trace_roles)):
        observed, original = model._read_observation(frame)
        deployed = model._decoder(original)
        g = CandidateGeometry(observed.boxes, observed.sampled_points)
        scores = ReadoutInputs(observed, original, traces['native'], g)
        graph = traces['graph']
        edges = [edge for edge in graph['relations'] if 'T' in (edge['subject'], edge['reference'])]
        lr = scores.active and len(edges) == 1 and edges[0]['predicate'] in {'LEFT', 'RIGHT'}
        eligible = distinct_entity_relation(graph, scores.active)
        slots = {name: scores.select(name) for name in SLOT_PROFILES}
        evidence = {name: scores.for_slots(*ids) for name, ids in slots.items()}
        for key in ('subject_boxes', 'reference_boxes', 'base_logit', 'ref_logit'):
            torch.testing.assert_close(evidence['v1'][key], original[key], atol=0, rtol=0)
        predicted = {}
        identities = {}

        def convert(box):
            return sensor_frame_box(box, target, observed.view_index)

        def store(name, profile, selected):
            subject_ids, ref_ids = slots[profile]
            s, r = (int(x) for x in selected)
            identities[name] = (subject_ids[s], ref_ids[r] if r >= 0 else -1)
            predicted[name] = convert(g.orient(subject_ids[s]))

        store('v1', 'v1', deployed['choices'][0])
        np.testing.assert_array_equal(predicted['v1'], old_box)
        store('direct4', 'v1', (0, -1))
        for name in SLOT_PROFILES[1:]:
            store(name, name, decoder.trial(evidence[name]))

        rejected = 0
        for profile, normal, output in (('v1', 'v1', 'binding'),
                                        ('role4', 'role4', 'role_binding')):
            mask = g.admissible(*slots[profile]) if eligible else None
            if mask is None or mask.all():
                predicted[output], identities[output] = predicted[normal], identities[normal]
                if profile == 'v1':
                    predicted['binding_shift'] = predicted[normal]
                    identities['binding_shift'] = identities[normal]
                continue
            valid = torch.as_tensor(mask[None], device=scores.device)
            store(output, profile, decoder.trial(evidence[profile], valid))
            if profile == 'v1':
                rejected = int((~mask).sum())
                store('binding_shift', profile, decoder.trial(evidence[profile], valid.roll(1, dims=-1)))
            subject, reference = identities[output]
            assert reference < 0 or (len(g.members(reference)) >= 4 and not g.same_support(subject, reference))

        sibling_counts = {}
        for output, before in (('refine', 'v1'), ('combined', 'role_binding')):
            subject = identities[before][0]
            box, n_neighbors = g.refine(subject)
            predicted[output] = convert(box)
            identities[output] = identities[before]
            sibling_counts[output] = n_neighbors
            assert predicted[output][6] == predicted[before][6]

        boxes = np.stack([predicted[name] for name in arms])
        pairs = np.array([identities[name] for name in arms], dtype=np.int16)
        same = np.asarray([r >= 0 and g.same_support(s, r) for s, r in pairs])
        pool = np.stack([convert(g.orient(i)) for i in range(len(observed.boxes))])
        indices = np.full((len(slots), 2, 8), -1, dtype=np.int16)
        reachable = np.zeros((len(slots), 8), dtype=bool)
        for j, (profile, (subjects, references)) in enumerate(slots.items()):
            indices[j, 0, :len(subjects)] = subjects
            indices[j, 1, :len(references)] = references
            reachable[j, :len(subjects)] = evidence[profile]['base_logit'][0].cpu().numpy() >= -2.5
        assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
        assert np.isfinite(pool).all() and (pool[:, 3:6] > 0).all()
        return dict(boxes=boxes, pool=pool, slots=indices, reachable=reachable, pairs=pairs, same=same,
                    flags=[scores.active, lr, eligible], invalid_pairs=rejected,
                    sibling_counts=[sibling_counts['refine'], sibling_counts['combined']])


def predict(protocol, rows, baseline, smoke):
    import torch
    from crossplatform.ground import Grounder
    from mechanisms import ReadoutTrial

    output = HERE / ('engineering_PRIVATE' if smoke else 'predictions_PRIVATE')
    output.mkdir(mode=0o700)
    lock = dict(implementation=implementation(), baseline=protocol['baseline_prediction_seal_sha256'],
                roster=protocol['input_sha256'], FINAL_access=False, target_labels_read=False)
    save(output / 'LOCK.json', lock)
    files = {}
    start = perf_counter()
    for source in PLATFORMS:
        model = Grounder(source)
        decoder = ReadoutTrial().cuda().eval().requires_grad_(False)
        decoder.load_state_dict(model._decoder.state_dict(), strict=True)
        for target in PLATFORMS:
            if target == source:
                continue
            ordinals, old = registered_direction(rows, baseline, source, target)
            assert len(ordinals) == protocol['inputs_per_target']
            if smoke:
                ordinals, old = ordinals[:2], old[:2]
            definitions = []
            for j, (i, before) in enumerate(zip(ordinals, old), 1):
                definitions.append(sample(model, decoder, rows[int(i)],
                                          protocol['point_seed'], before, protocol['arms']))
                if j % 64 == 0 or j == len(ordinals):
                    print(source, target, j, '/', len(ordinals),
                          f'{perf_counter() - start:.1f}s', flush=True)
            name = f'{source}_to_{target}.npz'
            path = output / name
            values = {key: np.stack([record[key] for record in definitions]) for key in definitions[0]}
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream, arms=np.asarray(protocol['arms']), ordinals=ordinals, **values)
            os.replace(path.with_suffix('.tmp'), path)
            files[name] = dict(n=len(ordinals), sha256=digest(path))
        del model, decoder
        torch.cuda.empty_cache()
    n = sum(file['n'] for file in files.values())
    assert n == (12 if smoke else protocol['direction_records'])
    assert implementation() == lock['implementation']
    save(output / 'SEALED.json', dict(n=n, files=files, lock_sha256=digest(output / 'LOCK.json'),
         baseline_exact_matches=n, elapsed_s=perf_counter() - start, FINAL_access=False,
         target_labels_read=False))


def coverage_summary(items):
    result = []
    for subset in ('all', 'direct_left_right'):
        rows = [row for row in items if subset == 'all' or row['left_right']]
        groups = sorted({row['direction'] for row in rows})
        for group in ('all_directions', *groups):
            selected = [row for row in rows if group == 'all_directions' or row['direction'] == group]
            for mode in SLOT_PROFILES:
                all_iou = np.asarray([row['oracle'] for row in selected])
                recalled = np.asarray([row['slot_' + mode] for row in selected])
                outcome = np.asarray([row[mode] for row in selected])
                for threshold in (.25, .50):
                    pool, slot, win = all_iou > threshold, recalled > threshold, outcome > threshold
                    assert np.all(~win | slot) and np.all(~slot | pool)
                    result.append(dict(direction=group, subset=subset, arm=mode, iou_threshold=threshold,
                                       n=len(selected), pool_pass=int(pool.sum()), slot_pass=int(slot.sum()),
                                       output_pass=int(win.sum()), fails_pool=int((~pool).sum()),
                                       fails_retrieval=int((pool & ~slot).sum()),
                                       fails_selection=int((slot & ~win).sum()),
                                       reachable_slot_pass=sum(row['reachable_' + mode] > threshold
                                                               for row in selected)))
    return result


def score(protocol, rows, baseline):
    path = HERE / 'predictions_PRIVATE'
    seal, lock = read(path / 'SEALED.json'), read(path / 'LOCK.json')
    assert seal['n'] == protocol['direction_records']
    assert seal['lock_sha256'] == digest(path / 'LOCK.json')
    assert lock['implementation'] == implementation()
    assert not seal['FINAL_access'] and not seal['target_labels_read']
    arms = tuple(protocol['arms'])
    opened = {}
    for source, target in DIRECTIONS:
        direction = f'{source}_to_{target}'
        name = direction + '.npz'
        assert digest(path / name) == seal['files'][name]['sha256']
        ordinals, old = registered_direction(rows, baseline, source, target)
        with np.load(path / name, allow_pickle=False) as saved:
            assert tuple(saved['arms']) == arms
            np.testing.assert_array_equal(saved['ordinals'], ordinals)
            np.testing.assert_array_equal(saved['boxes'][:, 0], old)
            assert saved['pool'].shape == (len(ordinals), 256, 7)
            opened[direction] = {key: saved[key] for key in saved.files if key != 'arms'}
    outputs, oracles, score_input = [], [], {}
    error = 0.
    checks = 0
    for direction, data in opened.items():
        source, target = direction.split('_to_')
        results, groups = [], []
        for j, ordinal in enumerate(data['ordinals']):
            row = rows[int(ordinal)]
            gt = sensor_truth(target, row['metapath'], row['caption'], row['suffix'])
            ious, err = score_boxes(data['boxes'][j], gt)
            pool = data['pool'][j]
            oracle = many_iou(pool, gt)
            # Audit the best frame box and boxes nearest each decision boundary.
            verify_ix = np.unique([oracle.argmax(), np.abs(oracle - .25).argmin(),
                                   np.abs(oracle - .5).argmin()])
            independent, oracle_error = score_boxes(pool[verify_ix], gt)
            np.testing.assert_allclose(oracle[verify_ix], independent, atol=1e-12, rtol=0)
            checks += len(verify_ix)
            error = max(error, err, oracle_error)
            results.append(ious)
            groups.append(row['scene_cluster'])
            ref, lr, eligible = (int(flag) for flag in data['flags'][j])
            outcome = dict(zip(arms, (float(x) for x in ious)))
            subject = dict(direction=direction, left_right=lr, oracle=float(oracle.max()), **outcome)
            for k, mode in enumerate(SLOT_PROFILES):
                indices = data['slots'][j, k, 0]
                feasible = data['reachable'][j, k]
                subject['slot_' + mode] = float(oracle[indices[indices >= 0]].max())
                subject['reachable_' + mode] = float(oracle[indices[feasible]].max())
            oracles.append(subject)
            outputs.append(dict(
                source=source, target=target, reference=ref, direct_left_right=lr,
                binding_eligible=eligible, invalid_pairs=int(data['invalid_pairs'][j]),
                binding_unknown=int(data['pairs'][j, arms.index('binding'), 1] < 0),
                baseline_same_support=int(data['same'][j, 0]),
                binding_same_support=int(data['same'][j, arms.index('binding')]),
                refine_changed=int(not np.array_equal(data['boxes'][j, 0],
                                                       data['boxes'][j, arms.index('refine')])),
                **outcome))
        score_input[direction] = dict(ious=np.stack(results), roles=data['flags'][:, :2],
                                      recordings=np.asarray(groups))
        print('Independently scored', direction, len(results), flush=True)
    out = HERE / 'results'
    out.mkdir()
    summary, paired = summarize(score_input, arms, SUBSETS)
    save_csv(out / 'endpoints.csv', summary)
    save_csv(out / 'paired.csv', paired)
    save_csv(out / 'coverage.csv', coverage_summary(oracles))
    secrets.SystemRandom().shuffle(outputs)
    save_csv(out / 'anonymous_rows.csv', outputs)
    save(out / 'recording_intervals.json', recording_intervals(score_input, arms))
    save(out / 'audit.json', dict(n=len(outputs), endpoints=len(outputs) * len(arms),
         candidate_boxes_scored=len(outputs) * 256, oracle_boundary_checks=checks,
         max_independent_absolute_iou_error=error, threshold_disagreements=0,
         baseline_bit_exact=seal['baseline_exact_matches'], prediction_seal_sha256=digest(path / 'SEALED.json'),
         source_training_runs=0, target_parameter_updates=0, FINAL_access=False))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=('smoke', 'predict', 'score'))
    options = cli.parse_args()
    protocol = read(HERE / 'protocol.json')
    rows, baseline = register(protocol)
    if options.phase == 'score':
        score(protocol, rows, baseline)
    else:
        predict(protocol, rows, baseline, options.phase == 'smoke')
