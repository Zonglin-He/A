"""Check the sealed scalar evidence and prepare the public R168 study."""
import argparse
from collections import Counter
from math import exp, fsum
from pathlib import Path
import shutil
import sys
import tempfile

from scope168 import ARMS, CAL, HERE, ORDER, OUT, ROOT, atomic_json, read, roster, sha, status
from score165 import table
from score168 import verify_seals

sys.path.insert(0, str(ROOT / 'research/round154'))
from package154 import scan_release, transport_csv

PREFIX = 'studies/3eed_selective_reference_binding_20261118'
CODE = ('scope168.py', 'select168.py', 'calibrate168.py', 'replay168.py', 'score168.py', 'panels168.py', 'release168.py')
METRICS = ('mIoU', 'Acc25', 'Acc50', 'subject_identity')
ABSTAIN = ('old_ref_abstain', 'new_ref_abstain', 'scrambled_ref_abstain')


def endpoint(row, arm):
    iou = float(row[arm + '_iou'])
    assert 0 <= iou <= 1
    return iou, int(iou > .25), int(iou > .50), int(row[arm + '_subject'] == '1')


def groups(rows, group):
    if group == 'pooled':
        return [rows]
    if group == 'six_equal':
        chosen = ORDER
    elif group.startswith('target_'):
        chosen = [(s, t) for s, t in ORDER if t == group[7:]]
    else:
        source, _, target = group.split('_')
        chosen = [(source, target)]
    return [p for s, t in chosen if (p := [x for x in rows if (x['source'], x['target']) == (s, t)])]


def audit():
    verify_seals()
    policy = read(HERE / 'ABSTENTION_RULE.json')
    scores = read(HERE / 'EVALUATION_AUDIT.json')
    panels = read(HERE / 'PANELS_AUDIT.json')
    visual = read(HERE / 'SOURCE_VISUAL_AUDIT.json')
    assert scores['sensor_ious'] == 18620 * len(ARMS) and scores['max_sensor_error'] < 1.e-7
    assert scores['exact_r167_max_error'] == 0. and not (scores['FINAL_access'] or panels['FINAL_access'] or visual['FINAL_access'])
    assert all(sha(OUT / p) == v for p, v in {**scores['files'], **panels['files']}.items())
    assert sha(OUT / 'source_calibration.csv') == policy['source_calibration_sha256']
    assert sha(OUT / 'source_visual_review.csv') == visual['scalar_file_sha256']
    rows = table(OUT / 'all_rows.csv')
    edges = table(OUT / 'relation_edges.csv')
    oracle = table(OUT / 'oracle_rows.csv')
    flips = table(OUT / 'counterfactual_rows.csv')
    pairs = table(OUT / 'identity_counterfactual_rows.csv')
    terms = table(OUT / 'term_deltas.csv')
    assert len(rows) == len({(x['source'], x['target'], x['expression']) for x in rows}) == 18620
    assert len(edges) == 11378 and len(flips) == 2526 * len(ARMS) and len(pairs) == panels['n'] == 400
    assert len(oracle) == scores['oracles'] == sum(x['independently_verified'] == '1' for x in edges)
    assert all(r['r167_joint_active_refusals'] == '0' for r in rows)
    assert sum(int(r['physical_roles']) for r in rows) == scores['old_nonempty_roles']
    assert min(float(r['legacy_worst_physical_advantage']) for r in rows if r['legacy_worst_physical_advantage']) == scores['old_null_min_dominated_margin']
    caps = {}
    for source, rule in policy['rules'].items():
        intercept, *weights = rule['coef']
        upper = intercept + sum(max(w * l, w * h) for w, l, h in zip(weights, rule['lo'], rule['hi'], strict=True))
        caps[source] = 1. / (1. + exp(-upper))
        observed = max(float(x['calibrated_availability']) for x in edges
                       if x['source'] == source and x['calibrated_availability'])
        assert observed <= caps[source] + 1.e-12 < .5
    assert all(r[f'{a}_active_refusals'] == r['physical_roles'] and r[f'{a}_iou'] == r['independent_iou']
               and r[f'{a}_subject_switch'] == '0' and r[f'{a}_physical_owner_unknown'] == '0'
               for a in ABSTAIN for r in rows)
    assert all(fsum(float(r[x]) for x in ('subject', 'reference', 'left', 'overlap')) >= -1.e-10 for r in terms)
    switches = Counter(r['arm'] for r in terms)
    assert all(switches[a] == sum(int(r[f'{a}_subject_switch']) for r in rows) for a in ARMS)

    subsets = {'all': rows}
    for scope in ('parsed_direct', 'camera_horizontal', 'active_factor'):
        subsets[scope] = [r for r in rows if r[scope] == '1']
    cached = {}
    checked, max_diff = 0, 0.
    for r in table(OUT / 'metrics.csv'):
        scope, group, arm = r['scope'], r['group'], r['arm']
        key = scope, group
        if key not in cached:
            cached[key] = groups(subsets[scope], group)
        split = cached[key]
        assert sum(map(len, split)) == int(r['n'])
        per = [[fsum(x) / len(part) for x in zip(*(endpoint(v, arm) for v in part))] for part in split]
        for metric, values in zip(METRICS, zip(*per), strict=True):
            max_diff = max(max_diff, abs(float(r[metric]) - 100 * fsum(values) / len(per)))
            checked += 1
    assert max_diff < 1.e-10
    for r in table(OUT / 'paired.csv'):
        split = cached[r['scope'], r['group']]
        arm, control = r['arm'], r['control']
        after = [(endpoint(v, arm), endpoint(v, control)) for part in split for v in part]
        for i, metric in enumerate(METRICS):
            differences = [a[i] - b[i] for a, b in after]
            assert int(r[metric + '_rescues']) == sum(x > 1.e-10 for x in differences)
            assert int(r[metric + '_harms']) == sum(x < -1.e-10 for x in differences)
            checked += 2
    for r in table(OUT / 'identity_counterfactual_summary.csv'):
        sample = [v for v in pairs if (v['scope'], v['arm']) == (r['scope'], r['arm'])]
        valid = [v for v in sample if v['valid_pair'] == '1']
        assert (len(sample), len(valid)) == (int(r['all_attempted']), int(r['valid_pairs']))
        for k in ('both_subjects', 'both_endpoints_twice'):
            assert int(r[k]) == sum(v[k] == '1' for v in valid)
            assert all(v[k] == '' for v in sample if v['valid_pair'] == '0')
            checked += 1
    for r in table(OUT / 'oracle.csv'):
        sample = oracle if r['scope'] == 'all_verified' else [x for x in oracle if x['reference_in_bank'] == '1']
        arm = r['arm']
        values = [float(s[arm + '_iou']) for s in sample if s[arm + '_iou'] != '']
        assert len(values) == int(r['served']) and len(sample) == int(r['all_verified'])
        assert abs(100 * fsum(values) / len(values) - float(r['mIoU'])) < 1.e-10
        assert sum(x > .25 for x in values) == int(r['acc25_correct'])
        assert sum(x > .50 for x in values) == int(r['acc50_correct'])
        checked += 5
    result = dict(full_rows=len(rows), evaluated_edges=len(edges), independent_oracle_edges=len(oracle),
                  paired_counterfactuals=len(flips)//len(ARMS), image_review_rows=len(pairs),
                  independent_scalar_elements=checked, max_scalar_error=max_diff,
                  calibrated_probability_upper_bound=caps, preserves_all_negative_and_ambiguous_records=True,
                  FINAL_access=False)
    atomic_json(HERE / 'SCALAR_AUDIT.json', result)
    return result


def private_literals():
    hidden = set()
    rows = [r for group in roster().values() for r in group]
    rows.extend(x['row'] for x in read(CAL / 'ROSTER.json'))
    for row in rows:
        for key in ('id', 'caption', 'scene', 'pointpath', 'imagepath', 'metapath'):
            value = row[key]
            if isinstance(value, str) and len(value) >= 12:
                hidden.add(value.encode())
        folder = Path(row['metapath']).parent.name
        if len(folder) >= 16:
            hidden.add(folder.encode())
    return hidden


def export():
    proof = audit()
    folder = HERE / 'publication'
    folder.mkdir(exist_ok=True)
    out = folder / 'public_export_attempt002'
    assert not out.exists()
    temp = Path(tempfile.mkdtemp(dir=folder))
    try:
        def copy(src, rel):
            dst = temp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

        for name in CODE:
            copy(HERE / name, 'own_code/research/round168/' + name)
        copy(HERE / 'experiment/PROTOCOL.json', 'experiment/PROTOCOL.json')
        for name in ('REPORT.md', 'REPORT_ZH.md', 'DECISION.json', 'INPUT_LOCK.json', 'SOURCE_INPUT_SEAL.json',
                     'SOURCE_FEATURE_SEAL.json', 'SOURCE_VISUAL_AUDIT.json', 'PILOT_SEAL.json',
                     'PREDICTION_SEAL.json', 'CONTROLLED_PREDICTION_SEAL.json', 'EVALUATION_LOCK.json',
                     'EVALUATION_AUDIT.json', 'PANELS_AUDIT.json', 'SCALAR_AUDIT.json'):
            copy(HERE / name, name)
        parts = {}
        for p in sorted(OUT.glob('*.csv')):
            rows = table(p)
            if p.name == 'source_calibration.csv':
                private = {'source_mass_enrichment', 'original_margin', 'full_fit_probability',
                           'leave_recording_out_probability'}
                rows = [{k: v for k, v in row.items() if k not in private} for row in rows]
            parts[p.name] = transport_csv(temp, 'results/' + p.name, rows, tuple(rows[0]))
        atomic_json(temp / 'PUBLIC_EXPORT_AUDIT.json', dict(validation=proof, tables=parts,
                    contains_fitted_calibrator_parameters_or_source_feature_inputs=False,
                    contains_raw_media_text_coordinates_instances_features_weights_or_credentials=False))
        items = [dict(path=p.relative_to(temp).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(temp.rglob('*')) if p.is_file()]
        atomic_json(temp / 'MANIFEST.json', dict(prefix=PREFIX, files=items))
        scan_release(temp, private_literals())
        temp.rename(out)
        print('R168 public files', len(items) + 1, flush=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def register():
    audit()
    path = HERE / 'publication/PUBLICATION_RECEIPT.json'
    receipt = read(path)
    assert receipt['repository'] == 'Zonglin-He/A' and receipt['prefix'] == PREFIX
    assert receipt['remoteByteVerified'] and receipt['immutableVerifiedBeforeRefUpdate']
    assert receipt['commit'] == receipt['mainRefObserved']
    commit = receipt['commit']
    remote = f'https://github.com/Zonglin-He/A/blob/{commit}/{PREFIX}/REPORT_ZH.md'
    verdict = read(HERE / 'DECISION.json')
    status('completed_published_verified', public_report=remote, public_commit=commit,
           report='round168/REPORT_ZH.md', decision='round168/DECISION.json',
           publication_receipt='round168/publication/PUBLICATION_RECEIPT.json')
    state = ROOT / 'research/STATE.json'
    s = read(state)
    s.update(latest_completed_round='R168', publication_state='R168_REMOTE_BYTES_VERIFIED',
             measurement_state='R168_FIXED_BANK_SELECTIVE_BINDING_AUDITED', claim_state=verdict['finding'],
             public_study=PREFIX, publication_receipt='round168/publication/PUBLICATION_RECEIPT.json',
             next_action=verdict['decision'])
    s['directions']['crossplatform'].update(latest_completed_round='R168', latest_completed_publication=commit,
                                           latest_public_report=remote, latest_public_commit=commit)
    atomic_json(state, s)
    archive = ROOT / 'research/ARCHIVE.md'
    start = f'R168 completed and published {commit}'
    if start not in archive.read_text():
        with archive.open('a', encoding='utf-8') as f:
            f.write(f'\n## R168 — Selective reference binding with fixed subject and geometry\n\n'
                    f'{start}. 18,620 fixed R165/R167 directional records. {verdict["finding"]} '
                    '[Report](round168/REPORT_ZH.md) · [Verified receipt](round168/publication/PUBLICATION_RECEIPT.json)\n')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('stage', choices=('audit', 'export', 'register'))
    stage = args.parse_args().stage
    result = {'audit': audit, 'export': export, 'register': register}[stage]()
    if result is not None:
        print(result)
