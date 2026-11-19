"""Recruit natural D-pool misses without revealing predictions to the visual reviewer."""
from argparse import ArgumentParser
from collections import Counter, defaultdict, deque
from pathlib import Path
from time import perf_counter
import fcntl
import os

import numpy as np
from PIL import Image

from predict import points_in, retrieve
from prepare_source import DIRECTION, RELATION_PRIORITY, eligible, source_row
from review import WIDTH, ROWS_PER_PAGE, render
from scope import HERE, PRIVATE, RESULTS, ROOT, atomic_json, former_frames, read, sha, stable, train_path
from score import ARMS, check_receipts, evaluate, fit_rank, fit_rejection, load_example, owner_coverage, physical_owners
from score import summarize, write_rows

WORK = PRIVATE / 'absence_extension'
ROWS = WORK / 'SCREEN_INPUT.json'
RECRUITED = WORK / 'RECRUITED.json'
PRED = WORK / 'predictions'
PLAN = HERE / 'extension_plan.json'
SOURCE = 'waymo'
MAX_FRAMES = 192
MAX_PER_RECORDING = 4
RECRUITMENT = (('potential_miss', 24), ('covered_control', 12))


def preparation():
    from select170 import parser

    WORK.mkdir(parents=True, mode=0o700, exist_ok=True)
    assert not ROWS.exists()
    locked = PRIVATE / 'prep' / (SOURCE + '.json')
    restricted = {r['recording'] for r in read(locked)}
    prior, prior_paths, prior_points, prior_images = former_frames()
    for group in (PRIVATE / 'prep').glob('*.json'):
        if group.name.endswith('_seal.json'):
            continue
        for e in read(group):
            r = e['row']
            prior.add(Path(r['metapath']).resolve())
            prior_paths.update(Path(r[k]).resolve() for k in ('pointpath','imagepath'))
            prior_points.add(r['point_sha256'])
            prior_images.add(r['image_sha256'])
    dataset = train_path(SOURCE)
    groups = defaultdict(list)
    for r in read(dataset):
        if not r['positive'] or not DIRECTION.search(r['caption']):
            continue
        meta = Path(r['metapath']).resolve()
        key = stable('source-ref-rec', meta.parent.parent)
        if key in restricted or meta in prior or any(Path(r[k]).resolve() in prior_paths for k in ('pointpath','imagepath')):
            continue
        groups[key].append(r)
    queues = {g: deque(sorted(items, key=lambda r: (not bool(RELATION_PRIORITY.search(r['caption'])),
                    stable('new-absence-screen', r['metapath'], r['caption'])))) for g, items in groups.items()}
    parse = parser()
    rows, counts, reasons = [], Counter(), Counter()
    while queues and len(rows) < MAX_FRAMES:
        rec = min(queues, key=lambda g: (counts[g], stable('screen-queue', g)))
        raw = queues[rec].popleft()
        if not queues[rec]:
            del queues[rec]
        meta = Path(raw['metapath']).resolve()
        if meta in prior:
            reasons['duplicate_frame'] += 1
            continue
        r = source_row(raw, meta.parent.parent)
        if r is None:
            reasons['no_exact_subject_or_peer'] += 1
            continue
        why, item = eligible(r, parse)
        reasons[why] += 1
        if item is None:
            continue
        scan, image = sha(Path(r['pointpath'])), sha(Path(r['imagepath']))
        if scan in prior_points or image in prior_images:
            reasons['duplicate_media'] += 1
            continue
        prior.add(meta)
        prior_points.add(scan)
        prior_images.add(image)
        assert item['recording'] == rec
        counts[rec] += 1
        r.update(point_sha256=scan, image_sha256=image)
        item['case'] = f'T{len(rows) + 1:03d}'
        rows.append(item)
        if counts[rec] == MAX_PER_RECORDING:
            queues.pop(rec, None)
    assert rows and not ({e['recording'] for e in rows} & restricted)
    atomic_json(ROWS, rows)
    atomic_json(WORK / 'SCREEN_INPUT_SEAL.json', dict(file=sha(ROWS), plan=sha(PLAN), program=sha(Path(__file__)),
                index=sha(dataset), n=len(rows), recordings=len(counts), screening_counts=dict(reasons),
                protected_data_access=False, screened_without_D=True))
    print('Supplement screen locked', len(rows), 'frames from', len(counts), 'recordings', flush=True)


def checked_screen():
    seal = read(WORK / 'SCREEN_INPUT_SEAL.json')
    assert not seal['protected_data_access'] and seal['screened_without_D']
    assert seal['file'] == sha(ROWS) and seal['plan'] == sha(PLAN)
    rows = read(ROWS)
    assert len(rows) == seal['n']
    return rows


def d_packet(model, item):
    from crossplatform.ground_single_frame import orient_box
    from predict165 import points_for
    from requery169 import native_donors, native_forward

    r = item['row']
    assert sha(Path(r['pointpath'])) == r['point_sha256'] and sha(Path(r['imagepath'])) == r['image_sha256']
    points, indices = points_for(r)
    d = native_forward(model, r, points, indices, None, item['views']['D'])
    ids, _, _ = native_donors(d)
    raw = d['z']['boxes'][ids]
    oriented = np.stack([orient_box(points, b) for b in raw])
    return np.unique(points, axis=0), oriented, np.c_[raw, np.zeros(len(raw))]


def screen_coverage(packet, item):
    from mechanisms162 import annotation_inventory

    points, oriented, raw = packet
    r = item['row']
    _, boxes = annotation_inventory(r['metapath'], SOURCE, int(r['suffix'] or 0), r['object_index'])
    owners_by_point, masks, owners = physical_owners(points, boxes, oriented)
    return dict(Counter(owner_coverage(j, boxes, points, owners_by_point, oriented, raw, masks, owners)[0]
                        for j in item['choices']))


def screen():
    import torch
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint

    inputs = checked_screen()
    result = WORK / 'INVENTORY_COVERAGE_PRIVATE.json'
    assert not result.exists()
    profiles = {}
    started = perf_counter()
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        model = EvidenceGrounder(SOURCE)
        before = fingerprint(model.base), fingerprint(model.matcher)
        for k, item in enumerate(inputs, 1):
            with torch.inference_mode():
                packet = d_packet(model, item)
            profiles[item['case']] = screen_coverage(packet, item)
            if k % 24 == 0:
                print('Screened', k, 'of', len(inputs), flush=True)
        assert before == (fingerprint(model.base), fingerprint(model.matcher))
    atomic_json(result, profiles)
    atomic_json(WORK / 'COVERAGE_SEAL.json', dict(file=sha(result), input=sha(ROWS), n=len(profiles),
                program=sha(Path(__file__)), used_only_unmodified_frozen_D=True, identities_available=False,
                weights_unchanged=True, protected_data_access=False, elapsed=perf_counter()-started))


def invite():
    inputs = checked_screen()
    profile_path = WORK / 'INVENTORY_COVERAGE_PRIVATE.json'
    cert = read(WORK / 'COVERAGE_SEAL.json')
    assert cert['file'] == sha(profile_path) and cert['input'] == sha(ROWS) and not cert['identities_available']
    profiles = read(profile_path)
    assert set(profiles) == {r['case'] for r in inputs}
    selected, quota, preference, by_group = [], Counter(), {}, {}
    for tier, limit in RECRUITMENT:
        chosen = Counter()
        available = [item for item in inputs if (
             (profiles[item['case']].get('natural_miss', 0) > 0) if tier == 'potential_miss' else
             (profiles[item['case']].get('natural_miss', 0) == 0 and profiles[item['case']].get('covered', 0) > 0))]
        quota[tier] = len(available)
        for _ in range(min(len(available), limit)):
            def rank(item):
                rec, case = item['recording'], item['case']
                ratio = profiles[case].get('natural_miss', 0) / len(item['choices'])
                return (int(tier == 'covered_control' and rec not in preference), chosen[rec], -ratio,
                        stable('supplement-invitation', case))
            best = min(available, key=rank)
            available.remove(best)
            chosen[best['recording']] += 1
            selected.append(best)
            by_group[best['case']] = tier
        if tier == 'potential_miss':
            preference = chosen
    home = WORK / 'review'
    home.mkdir(mode=0o700, exist_ok=True)
    assert not RECRUITED.exists()
    selected.sort(key=lambda x: stable('concealed-cohort-order', x['case']))
    pages, keys, strata, cards = {}, {}, {}, []
    for i, item in enumerate(selected, 1):
        original = item['case']
        case = f'S{i:02d}'
        item['case'] = case
        strata[case] = by_group[original]
        order = sorted(item['choices'], key=lambda j: stable('extension-blind-object', case, j))
        keys[case] = {f'P{k+1}': j for k,j in enumerate(order)}
        cards.append(render(item, keys[case]))
        if len(cards) == ROWS_PER_PAGE or i == len(selected):
            page = Image.new('RGB',(WIDTH,sum(c.height + 5 for c in cards)),'#d8dce8')
            y = 0
            for c in cards:
                page.paste(c,(0,y))
                y += c.height + 5
            p = home / f'sheet_{(i-1)//ROWS_PER_PAGE+1:02d}.jpg'
            page.save(p, quality=93)
            pages[p.name] = sha(p)
            cards.clear()
    assert selected
    atomic_json(RECRUITED, selected)
    atomic_json(home / 'KEY.json', keys)
    atomic_json(WORK / 'INVITATION_STRATA_PRIVATE.json', strata)
    atomic_json(home / 'PLATE_SEAL.json', dict(input=sha(RECRUITED), key=sha(home / 'KEY.json'), pages=pages,
                coverage=sha(WORK / 'COVERAGE_SEAL.json'), strata=sha(WORK / 'INVITATION_STRATA_PRIVATE.json'),
                eligible_by_stratum=dict(quota), invited_by_stratum=dict(Counter(strata.values())),
                protected_data_access=False, predictions_shown=False, strata_shown=False))
    print('Supplement review plates ready', len(selected), 'invitations', flush=True)


def seal_truth():
    home = WORK / 'review'
    choices = home / 'BLIND_CHOICES.json'
    answers = read(choices)
    seal = read(home / 'PLATE_SEAL.json')
    keys = read(home / 'KEY.json')
    assert sha(home / 'KEY.json') == seal['key'] and sha(RECRUITED) == seal['input']
    assert not seal['predictions_shown'] and not seal['strata_shown'] and not seal['protected_data_access']
    assert all(sha(home / file) == digest for file,digest in seal['pages'].items())
    entries = read(RECRUITED)
    assert {r['case'] for r in entries} == set(answers) == set(keys)
    for item in entries:
        case = item['case']
        a = answers[case]
        t, basis, foils = a['choice'], a['basis'], a['foils']
        assert basis in ('distinct_appearance', 'distinct_relation_or_position', 'ambiguous', 'unclear_or_contradictory')
        if t == 'UNKNOWN':
            assert basis in ('ambiguous', 'unclear_or_contradictory') and not foils
        else:
            assert basis in ('distinct_appearance', 'distinct_relation_or_position') and t in keys[case]
        assert len(foils) == len(set(foils)) and all(f in keys[case] and f != t for f in foils)
        item.update(reference_identity=keys[case].get(t,-1), verification_status=basis,
                    verified_foil_ids=[keys[case][f] for f in foils])
    dest = WORK / 'VERIFIED_ROSTER.json'
    assert not dest.exists()
    atomic_json(dest, entries)
    atomic_json(WORK / 'TRUTH_SEAL.json',dict(file=sha(dest), choices=sha(choices), plates=sha(home / 'PLATE_SEAL.json'),
                n=len(entries), one_internal_reviewer=True, prediction_blind_identities=True, protected_data_access=False))
    print('Supplement identities sealed', len(entries), flush=True)


def harvest():
    import torch
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict171 import read_npz

    PRED.mkdir(mode=0o700, exist_ok=True)
    certpath = PRED / 'SEAL.json'
    assert not certpath.exists()
    lock = PRED / 'LOCK.json'
    intended = dict(input=sha(RECRUITED),plan=sha(PLAN),producer=sha(HERE/'predict.py'),protected_data_access=False)
    if lock.exists():
        assert read(lock) == intended
    else:
        atomic_json(lock, intended)
    rows = read(RECRUITED)
    output = {}
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        model = EvidenceGrounder(SOURCE)
        before = fingerprint(model.base), fingerprint(model.matcher)
        for item in rows:
            file = PRED / (item['case'] + '.npz')
            if file.exists():
                assert str(read_npz(file)['pin']) == sha(lock)
            else:
                with torch.inference_mode():
                    packet = retrieve(model, item['row'], item['views'], item['case'])
                npz_atomic(file, **packet, pin=sha(lock))
            output[item['case']] = sha(file)
        assert before == (fingerprint(model.base), fingerprint(model.matcher))
    atomic_json(certpath,dict(pin=sha(lock), files=output, weights_unchanged=True, protected_data_access=False,
                selected_using_truth=False, truth_not_loaded=True))
    print('Sealed supplement prediction files',len(output),flush=True)


def assess():
    truth_path = WORK / 'VERIFIED_ROSTER.json'
    truth, pred = read(WORK / 'TRUTH_SEAL.json'), read(PRED / 'SEAL.json')
    assert truth['file'] == sha(truth_path) and pred['pin'] == sha(PRED / 'LOCK.json')
    assert pred['truth_not_loaded'] and pred['weights_unchanged'] and not truth['protected_data_access']
    assert all(sha(PRED / (k + '.npz')) == v for k,v in pred['files'].items())
    prior_rows, prior_pin = check_receipts()
    fixed = [load_example(e, prior_pin) for e in prior_rows if e['row']['platform'] == SOURCE]
    supplement = [load_example(e, pred['pin'], PRED) for e in read(truth_path)]
    assert len(supplement) == truth['n'] == len(pred['files'])
    assert not {e.recording for e in supplement} & {e.recording for e in fixed}
    for e in supplement:
        assert e.case in pred['files']
    by_recording = defaultdict(list)
    for e in supplement:
        by_recording[e.recording].append(e)
    outputs, folds = [], []
    for held, rows in sorted(by_recording.items()):
        train = fixed + [e for e in supplement if e.recording != held]
        assert held not in {e.recording for e in train}
        outputs.extend(evaluate(e, e.unary, None, 'D') for e in rows)
        for arm in ARMS[1:]:
            rejection, cover, absent, params = fit_rejection(train, arm)
            coefficients, n = fit_rank(train, arm) if arm != 'D_only' else (None, 0)
            folds.append(dict(arm=arm, held=held, calibration=[e.case for e in train],
                              evaluated=[e.case for e in rows], covered_train=cover, missing_train=absent,
                              hard_train=n, fully_fit=params is not None))
            for e in rows:
                scores = e.unary if coefficients is None else e.unary + e.design[arm] @ coefficients
                outputs.append(evaluate(e, scores, rejection(e), arm))
    assert len(outputs) == len(supplement)*len(ARMS)
    strata = read(WORK / 'INVITATION_STRATA_PRIVATE.json')
    sample = {e.case: f'B{i:02d}' for i,e in enumerate(sorted(supplement,key=lambda e: e.case),1)}
    groups = {r:f'H{i:02d}' for i,r in enumerate(sorted(by_recording),1)}
    public = [dict(o, case=sample[o['case']], recording=groups[o['recording']], invitation=strata[o['case']]) for o in outputs]
    tables = {k:{arm:summarize([r for r in public if r['arm']==arm and (k=='all' or r['invitation']==k)])
                for arm in ARMS} for k in ('all', *dict(RECRUITMENT))}
    all_metrics = tables['all']
    a, d, dcal, wrong = (all_metrics[k] for k in ('aligned', 'D', 'D_only', 'mismatched'))
    first = read(RESULTS / 'SUMMARY.json')['metrics'][SOURCE]
    enough_rec = a['miss_recordings'] > 1 and all(f['fully_fit'] for f in folds)
    added_rejection = (a['missed_rejected'] > max(dcal['missed_rejected'], wrong['missed_rejected'])
                       and a['covered_correct_accepted'] >= max(d['covered_correct_accepted'], dcal['covered_correct_accepted']))
    carried_rank = bool(first['aligned']['hard_concordance'] > max(first['D']['hard_concordance'],
                                                                first['mismatched']['hard_concordance']))
    first_accept = first['aligned']['covered_correct_accepted'] >= first['D']['covered_correct_accepted']
    gate = enough_rec and added_rejection and carried_rank and first_accept
    complete = RESULTS / 'EXTENSION_SUMMARY.json'
    assert not complete.exists()
    atomic_json(WORK / 'FOLDS_PRIVATE.json',folds)
    write_rows(RESULTS / 'enriched_reference_outcomes.csv', public)
    screen = read(WORK/'SCREEN_INPUT_SEAL.json')
    recruit = read(WORK/'review/PLATE_SEAL.json')
    atomic_json(complete,dict(screen_frames=screen['n'], screen_recordings=screen['recordings'],
        invited=truth['n'], invited_recordings=len(by_recording), eligible_by_stratum=recruit['eligible_by_stratum'],
        invited_by_stratum=recruit['invited_by_stratum'], statuses=dict(Counter(e.status for e in supplement)), metrics=tables,
        full_binary_folds={a:sum(f['arm']==a and f['fully_fit'] for f in folds) for a in ARMS[1:]},
        gate=dict(recording_support=enough_rec, rejection_increment=added_rejection, first_panel_ranking=carried_rank,
                  first_panel_correct_acceptance=first_accept, advance=gate),
        provenance=dict(truth=sha(WORK/'TRUTH_SEAL.json'), predicted=sha(PRED/'SEAL.json'), program=sha(Path(__file__)),
                        plan=sha(PLAN), weights_unchanged=True, protected_data_access=False)))
    print('Supplement scoring completed. Joint gate passed:', gate, flush=True)


if __name__ == '__main__':
    stages = dict(prepare=preparation, screen=screen, invite=invite, seal=seal_truth, predict=harvest, score=assess)
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=tuple(stages))
    stages[ap.parse_args().stage]()
