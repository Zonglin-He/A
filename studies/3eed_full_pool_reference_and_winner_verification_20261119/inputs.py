"""Read the previous exact source predictions and split manifests before any fit."""
from collections import Counter
from dataclasses import dataclass
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np


HOME = Path(__file__).resolve().parent
PRIVATE = HOME / 'private'
RESULTS = HOME / 'results'
PROTOCOL = HOME / 'protocol.json'
PREVIOUS = HOME.parent / '3eed_source_reference_verification_20261119'
SUPPLEMENT = PREVIOUS / 'private' / 'absence_extension'
sys.path.insert(0, str(PREVIOUS))
from scope import atomic_json, read, sha
import score as previous


@dataclass
class Input:
    example: previous.Example
    alias: str
    cohort: str
    rec: str
    has_c: np.ndarray
    has_noun: np.ndarray
    centered: np.ndarray
    shifted: np.ndarray
    owner_types: dict

    @property
    def group(self):
        return self.cohort, self.example.platform

    @property
    def features(self):
        return self.example.design['aligned']

    @property
    def owners(self):
        return self.example.owners


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    assert rows
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shift_indices(key, n):
    if n < 2:
        return np.arange(n)
    digest = hashlib.sha256(('reference-wrong-match-v1|' + key).encode()).digest()
    shift = 1 + int.from_bytes(digest[:8], 'little') % (n - 1)
    return np.roll(np.arange(n), shift)


def audit_owner_types(row, e):
    from calibrate168 import image_geometry
    from mechanisms162 import annotation_inventory
    from visual162 import projection

    if e.gold < 0:
        assert (e.owners < 0).all()
        return {-1: 'unassigned'}
    raw = row['row']
    names, boxes = annotation_inventory(raw['metapath'], e.platform, int(raw['suffix'] or 0), raw['object_index'])
    center = projection(image_geometry(raw), boxes[:, :3])
    computed = {-1: 'unassigned', 0: 'annotated_subject', e.gold: 'true_reference'}
    for i in set(e.owners) - set(computed):
        if i in e.verified_foils:
            category = 'verified_same_class_same_side_foil'
        elif names[i] != row['reference_class']:
            category = 'different_class_object'
        elif row['sign'] * (center[i, 0] - center[0, 0]) >= .014:
            category = 'other_same_class_same_side'
        else:
            category = 'same_class_other_or_unclear_side'
        computed[int(i)] = category
    return computed


def pack(rows, pin, folder, cohort, old_rows):
    from predict171 import read_npz

    cases = sorted(r['case'] for r in rows)
    aliases = {k: (f'A{i:03d}' if cohort == 'initial' else f'B{i:02d}') for i, k in enumerate(cases, 1)}
    recordings = sorted({r['recording'] for r in rows})
    recs = {k: ('G' if cohort == 'initial' else 'H') + f'{i:02d}' for i, k in enumerate(recordings, 1)}
    out = []
    for r in rows:
        e = previous.load_example(r, pin, folder)
        p = read_npz(folder / (e.case + '.npz'))
        c_present = np.isfinite(p['c'])
        noun_present = np.isfinite(p['noun'])
        assert np.all(e.design['aligned'][~c_present, 0] == 0.)
        assert np.all(e.design['aligned'][~(c_present & noun_present), 1] == 0.)
        x = e.design['aligned'].copy()
        if c_present.any():
            x[c_present, 0] -= np.median(x[c_present, 0])
        shuff = shift_indices(e.case, len(e.unary))
        np.testing.assert_array_equal(e.design['aligned'][shuff], e.design['mismatched'])
        example = Input(e, aliases[e.case], cohort, recs[e.recording], c_present, noun_present,
                        x, x[shuff], audit_owner_types(r, e))
        published = old_rows[(example.alias, 'D')]
        assert (published['recording'], published['status'], published['platform']) == (example.rec, e.status, e.platform)
        assert (published['raw_correct'] == 'True') == bool(e.gold >= 0 and e.owners[0] == e.gold)
        out.append(example)
    return out


def load():
    audited = read(PREVIOUS / 'results/AUDIT.json')
    assert audited['status'] == 'PASS' and not audited['protected_data_access']
    assert all(sha(PREVIOUS / 'results' / name) == expected for name, expected in audited['files'].items())
    release = read(PREVIOUS / 'publication/PUBLICATION_RECEIPT.json')
    assert release['commit'] == read(PROTOCOL)['prior_public_commit']
    truth, pin = previous.check_receipts()
    extra_truth_file = SUPPLEMENT / 'VERIFIED_ROSTER.json'
    truth_seal = read(SUPPLEMENT / 'TRUTH_SEAL.json')
    prediction_dir = SUPPLEMENT / 'predictions'
    prediction_seal = read(prediction_dir / 'SEAL.json')
    assert not truth_seal['protected_data_access'] and not prediction_seal['protected_data_access']
    assert prediction_seal['truth_not_loaded'] and prediction_seal['weights_unchanged']
    assert truth_seal['file'] == sha(extra_truth_file)
    assert prediction_seal['pin'] == sha(prediction_dir / 'LOCK.json')
    extra = read(extra_truth_file)
    assert len(extra) == truth_seal['n'] == 36 and len(truth) == 108
    assert set(prediction_seal['files']) == {row['case'] for row in extra}
    assert all(sha(prediction_dir / (k+'.npz')) == v for k, v in prediction_seal['files'].items())
    assert not {r['recording'] for r in extra} & {r['recording'] for r in truth}

    old_paths = [PREVIOUS / 'results' / f for f in ('reference_outcomes.csv', 'enriched_reference_outcomes.csv')]
    prior = {(r['case'], r['arm']): r for path in old_paths for r in csv_rows(path)}
    out = pack(truth, pin, previous.PRED, 'initial', prior)
    out.extend(pack(extra, prediction_seal['pin'], prediction_dir, 'enriched', prior))
    assert len({a.alias for a in out}) == 144
    assert len({a.example.recording for a in out}) == 58
    assert Counter(a.example.status for a in out) == {
        'covered': 55, 'unresolved_review': 81, 'natural_miss': 6, 'ambiguous_candidate_coverage': 2}
    seals = {
        'prior_audit': sha(PREVIOUS / 'results/AUDIT.json'),
        'initial_truth': sha(PREVIOUS / 'private/TRUTH_SEAL.json'),
        'initial_predictions': sha(previous.PRED / 'SEAL.json'),
        'enriched_truth': sha(SUPPLEMENT / 'TRUTH_SEAL.json'),
        'enriched_predictions': sha(prediction_dir / 'SEAL.json'),
    }
    return out, prior, seals


def outer_folds(cases):
    for cohort, platform in sorted({case.group for case in cases}):
        subset = [x for x in cases if x.group == (cohort, platform)]
        eligible = [x for x in cases if x.example.platform == platform and
                    (cohort == 'enriched' or x.cohort == 'initial')]
        for rec in sorted({x.rec for x in subset}):
            test = [x for x in subset if x.rec == rec]
            train = [x for x in eligible if x.rec != rec]
            assert not {x.example.recording for x in train} & {x.example.recording for x in test}
            yield train, test
