"""Independently check source membership, fold boundaries and public count arithmetic."""
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import subprocess

import numpy as np

from scope import HERE, PRIVATE, RESULTS, PLATFORMS, atomic_json, read, sha, stable, train_path

ARMS = ('D', 'D_only', 'aligned', 'mismatched')
EXT = PRIVATE / 'absence_extension'


def check_roster(path, expected, seen):
    rows = read(path)
    assert len(rows) == expected
    sources = set()
    def source_inventory(s):
        return {(r['metapath'], r['pointpath'], r['imagepath'], r['caption'].strip())
                for r in read(train_path(s)) if r['positive']}

    allowed = {s: source_inventory(s) for s in PLATFORMS}
    for e in rows:
        row = e['row']
        s = row['platform']
        sources.add(s)
        check = tuple(row[k] for k in ('metapath', 'pointpath', 'imagepath', 'caption'))
        assert check in allowed[s]
        real = Path(row['metapath']).resolve()
        assert stable('source-ref-rec', real.parent.parent) == e['recording']
        keys = {real, row['point_sha256'], row['image_sha256']}
        assert not (seen & keys)
        seen.update(keys)
        assert sha(Path(row['pointpath'])) == row['point_sha256']
        assert sha(Path(row['imagepath'])) == row['image_sha256']
        assert not e['verified_foil_ids'] or e['reference_identity'] in e['choices']
        assert set(e['verified_foil_ids']) <= set(e['choices']) - {e['reference_identity']}
    if path.parent == EXT:
        assert sources == {'waymo'}
    return {e['case']:e for e in rows}


def audit_prediction_seal(root, roster):
    s = read(root / 'SEAL.json')
    assert not s['protected_data_access'] and s['pin'] == sha(root / 'LOCK.json')
    assert set(s['files']) == set(roster) == {p.stem for p in root.glob('*.npz')}
    totals = Counter()
    for case, checksum in s['files'].items():
        file = root / (case + '.npz')
        assert sha(file) == checksum
        with np.load(file, allow_pickle=False) as data:
            assert data['pin'].item() == s['pin']
            wrong, right = data['mismatched'], data['evidence']
            assert len(wrong) == len(right) == len(data['raw']) == len(data['boxes'])
            assert Counter(map(tuple, wrong)) == Counter(map(tuple, right))
            if len(right) > 1:
                assert any(np.array_equal(wrong, np.roll(right, shift, axis=0)) for shift in range(1, len(right)))
            totals['cue_tuples'] += len(right)
            totals['changed_mismatch_frames'] += not np.array_equal(wrong,right)
            totals['all_frames'] += 1
    return dict(totals)


def audit_folds(fits, fit_roster, evaluation_roster, extension):
    k = set()
    for f in fits:
        hold = f['held'] if extension else f['holdout']
        calibration = f['calibration'] if extension else f['fit_cases']
        evaluation = f['evaluated'] if extension else f['held_cases']
        expected_test = {i for i,e in evaluation_roster.items() if e['recording'] == hold}
        assert set(evaluation) == expected_test and len(evaluation) == len(expected_test) and expected_test
        source = evaluation_roster[evaluation[0]]['row']['platform']
        expected_training = {i for i,e in fit_roster.items() if e['recording'] != hold and e['row']['platform'] == source}
        assert set(calibration) == expected_training and len(calibration) == len(expected_training)
        assert not (set(calibration) & set(evaluation))
        assert (hold, f['arm']) not in k and f['arm'] in ARMS[1:]
        k.add((hold, f['arm']))
    return len(k)


def read_csv(p):
    with p.open(newline='') as f:
        out = list(csv.DictReader(f))
    for r in out:
        for n in ('covered_correct_accepted','raw_correct','correct_accept','reject','all_foils_beaten'):
            if n in r:
                assert r[n] in ('True','False','')
        for n in ('case','recording'):
            assert re.fullmatch(r'[ABGH]\d{2,3}',r[n])
    return out


def no_duplicate_or_missing_outcomes(rows, roster, extension):
    units = defaultdict(set)
    for r in rows:
        k = r['case'], r['platform'], r['recording']
        assert r['arm'] not in units[k]
        units[k].add(r['arm'])
        if r['arm'] == 'D':
            assert r['reject'] == 'False'
        if r['correct_accept'] == 'True':
            assert r['raw_correct'] == 'True' and r['reject'] == 'False'
        if r['status'] != 'covered':
            assert r['rank'] == '' and r['raw_correct'] == 'False'
    assert len(units) == len(roster)
    for (_, p, _), arms in units.items():
        assert arms == set(ARMS if extension or p != 'quad' else ('D',))
    return len(units)


def check_metrics(rows, expected):
    cases = {r['case'] for r in rows}
    covered = [r for r in rows if r['status'] == 'covered']
    missing = [r for r in rows if r['status'] == 'natural_miss']
    ranked = [r for r in covered if r['hard_frame']]
    assert len(covered) == sum(bool(r['rank']) for r in rows)
    assert len(cases) == len(rows) == expected['n']
    assert len(covered) == expected['cover']
    assert len(missing) == expected['miss']
    assert len(ranked) == expected['hard_frames']
    assert len({r['recording'] for r in missing}) == expected['miss_recordings']
    assert len({r['recording'] for r in ranked}) == expected['hard_recordings']
    for col in ('pair_wins','pair_ties','pair_losses','geometric_wins','geometric_ties','geometric_losses'):
        src = col.removeprefix('pair_')
        assert sum(map(lambda row: int(row[src]), rows)) == expected[col]
    assert sum(r['reject'] == 'True' for r in missing) == expected['missed_rejected']
    assert sum(r['reject'] == 'True' for r in covered) == expected['covered_rejected']
    assert sum(r['correct_accept'] == 'True' for r in covered) == expected['covered_correct_accepted']
    assert sum(r['raw_correct'] == 'True' for r in covered) == expected['covered_top1']
    assert sum(r['reject'] == 'True' for r in rows) == expected['total_rejected']
    if ranked:
        frac = [(int(r['wins'])+.5*int(r['ties']))/(int(r['wins'])+int(r['ties'])+int(r['losses'])) for r in ranked]
        assert abs(sum(frac)/len(frac) - expected['hard_concordance']) < 1.e-10
    else:
        assert expected['hard_concordance'] is None


def main():
    import hashlib

    summary = read(RESULTS / 'SUMMARY.json')
    old = subprocess.check_output(['git','show','cd57f9b:score.py'],cwd=HERE)
    assert hashlib.sha256(old).hexdigest() == summary['provenance']['scorer']
    ext = read(RESULTS / 'EXTENSION_SUMMARY.json')
    assert sha(HERE/'extension.py') == ext['provenance']['program']
    assert sha(HERE/'scoring_plan.json') == summary['provenance']['plan']
    assert sha(HERE/'extension_plan.json') == ext['provenance']['plan']
    seen = set()
    first = check_roster(PRIVATE/'VERIFIED_ROSTER.json',108,seen)
    followup = check_roster(EXT/'VERIFIED_ROSTER.json',ext['invited'],seen)
    before_rec = {e['recording'] for e in first.values()}
    after_rec = {e['recording'] for e in followup.values()}
    assert not before_rec & after_rec
    prediction_audit = dict(first=audit_prediction_seal(PRIVATE/'predictions',first),
                            followup=audit_prediction_seal(EXT/'predictions',followup))
    folds = dict(first=audit_folds(read(PRIVATE/'evaluation/FIT.json'),first,first,False),
                 followup=audit_folds(read(EXT/'FOLDS_PRIVATE.json'),first|followup,followup,True))
    rows = read_csv(RESULTS/'reference_outcomes.csv')
    extension = read_csv(RESULTS/'enriched_reference_outcomes.csv')
    assert no_duplicate_or_missing_outcomes(rows,first,False) == 108
    assert no_duplicate_or_missing_outcomes(extension,followup,True) == ext['invited']
    for source, methods in summary['metrics'].items():
        for arm, m in methods.items():
            check_metrics([r for r in rows if r['platform'] == source and r['arm'] == arm],m)
    for tier, methods in ext['metrics'].items():
        for arm, m in methods.items():
            check_metrics([r for r in extension if r['arm'] == arm and (tier=='all' or r['invitation']==tier)],m)
    assert not summary['provenance']['protected_data_access'] and not ext['provenance']['protected_data_access']
    files = ('SUMMARY.json','EXTENSION_SUMMARY.json','reference_outcomes.csv',
             'recording_outcomes.csv','enriched_reference_outcomes.csv')
    out = dict(status='PASS', first_panel=len(first), extension_panel=len(followup),
               recordings_separated=True, calibration_fit_boundaries_checked=folds,
               prediction_permutation_counts=prediction_audit, independent_aggregate_arithmetic_checked=True,
               original_model_weights_changed=False, protected_data_access=False,
               files={f:sha(RESULTS/f) for f in files})
    atomic_json(RESULTS/'AUDIT.json',out)
    print('Source registration, recording isolation, permutations and arithmetic audited', flush=True)


if __name__ == '__main__':
    main()
