"""Recompute scalar claims and export only anonymized evidence and first-party code."""
import argparse
from math import fsum
from pathlib import Path
import shutil
import sys
import tempfile

from scope167 import ARMS, HERE, ORDER, OUT, ROOT, atomic_json, read, roster, sha, status
from score167 import eval_lock, table

sys.path.insert(0, str(ROOT / 'research/round154'))
from package154 import scan_release, transport_csv

PREFIX = 'studies/3eed_reference_retrieval_joint_binding_20261118'
CODE = ('scope167.py', 'retrieval167.py', 'predict167.py', 'audit167.py', 'score167.py', 'review167.py',
        'challenge167.py', 'release167.py')


def signal(row, arm):
    value = float(row[arm + '_iou'])
    assert 0 <= value <= 1
    return value, int(value > .25), int(value > .5), int(row[arm + '_subject'] == '1'), int(row[arm + '_subject_box'] == '1')


def metric_groups(items, group):
    if group == 'pooled':
        return [items]
    if group == 'six_equal':
        directions = ORDER
    elif group.startswith('target_'):
        directions = [(s, t) for s, t in ORDER if t == group[7:]]
    else:
        s, _, t = group.split('_')
        directions = [(s, t)]
    return [p for s, t in directions if (p := [r for r in items if r['source'] == s and r['target'] == t])]


def audit():
    eval_lock()
    e = read(HERE / 'EVALUATION_AUDIT.json')
    v = read(HERE / 'VISUAL_REVIEW_AUDIT.json')
    c = read(HERE / 'CONTROLLED_EVALUATION_AUDIT.json')
    assert e['n'] == 18620 and e['independently_scored_outputs'] == 93100 and e['max_sensor_iou_error'] < 1e-7
    assert e['max_r165_baseline_difference'] < 1e-11 and not e['FINAL_access'] and not v['FINAL_access']
    assert sha(HERE / 'PREDICTION_SEAL.json') == e['prediction_seal']
    assert c['rows'] == 40 and c['direction_pairs'] == 10 and not c['FINAL_access']
    assert all(sha(OUT / name) == digest for name, digest in {**e['files'], **v['files'], **c['files']}.items())
    rows = table(OUT / 'all_rows.csv')
    edges = table(OUT / 'relation_edges.csv')
    flip = table(OUT / 'counterfactual_rows.csv')
    assert len(rows) == len({(r['source'], r['target'], r['expression']) for r in rows}) == e['n']
    assert len(edges) == e['edges'] == 11378 and len(flip) == 4 * e['flip_pairs']
    assert sum(r['truth'] in ('annotation_proxy', 'visual_verified') for r in edges) == 750
    assert all(r['independent_iou'] == r['r158_pca_iou'] for r in rows)
    scopes = {'all': rows}
    scopes.update({name: [r for r in rows if r[name] == '1'] for name in ('parsed_direct', 'camera_horizontal', 'active_factor')})
    cache = {}
    checked, error = 0, 0.
    metrics = ('mIoU', 'Acc25', 'Acc50', 'subject_identity', 'subject_box_proxy')
    for row in table(OUT / 'metrics.csv'):
        key = row['scope'], row['group']
        if key not in cache:
            cache[key] = metric_groups(scopes[key[0]], key[1])
        groups = cache[key]
        assert sum(map(len, groups)) == int(row['n'])
        per = [[fsum(col) / len(p) for col in zip(*(signal(x, row['arm']) for x in p))] for p in groups]
        expected = [100 * fsum(values) / len(per) for values in zip(*per)]
        error = max(error, *(abs(float(row[m]) - val) for m, val in zip(metrics, expected, strict=True)))
        checked += len(metrics)
    assert error < 1e-10
    visual = table(OUT / 'counterfactual_visual.csv')
    assert len(visual) == v['panels'] * 2 * 4
    assert all(r['original_iou'] == '' and r['flipped_iou'] == '' for r in visual if r['valid_pair'] == '0')
    for arm in ARMS[1:]:
        labels = [r for r in visual if r['arm'] == arm]
        valid = [r for r in labels if r['valid_pair'] == '1']
        expected = next(r for r in table(OUT / 'counterfactual_visual_summary.csv') if r['target'] == 'all' and r['arm'] == arm)
        assert len(labels) == int(expected['all_attempted']) and len(valid) == int(expected['valid_pairs'])
        assert sum(r['both_endpoints_twice'] == '1' for r in valid) == int(expected['both_endpoints_twice'])
    challenge = table(OUT / 'controlled_counterfactual.csv')
    for result in table(OUT / 'controlled_counterfactual_summary.csv'):
        matches = [r for r in challenge if r['arm'] == result['arm']]
        assert len(matches) == int(result['n']) == 10
        for field in ('left_subject', 'right_subject', 'left_ref', 'right_ref', 'both_subjects', 'both_endpoints_twice'):
            assert sum(x[field] == '1' for x in matches) == int(result[field])
            checked += 1
    receipt = dict(all_rows=len(rows), relation_edges=len(edges), predicate_interventions=e['flip_pairs'],
                   blind_rows=len(visual), controlled_pairs=c['direction_pairs'],
                   independent_metric_elements=checked, max_scalar_error=error,
                   negative_and_unknown_rows_retained=True, FINAL_access=False)
    atomic_json(HERE / 'SCALAR_AUDIT.json', receipt)
    return receipt


def private_literals():
    hidden = set()
    for group in roster().values():
        for row in group:
            for key in ('id', 'caption', 'scene', 'pointpath', 'imagepath', 'metapath'):
                value = row[key]
                if isinstance(value, str) and len(value) >= 12:
                    hidden.add(value.encode())
            value = Path(row['metapath']).parent.name
            if len(value) >= 16:
                hidden.add(value.encode())
    return hidden


def export():
    validation = audit()
    folder = HERE / 'publication'
    folder.mkdir(exist_ok=True)
    out = folder / 'public_export_attempt001'
    assert not out.exists()
    temp = Path(tempfile.mkdtemp(dir=folder))
    try:
        def copy(src, rel):
            dst = temp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        for name in CODE:
            copy(HERE / name, f'own_code/research/round167/{name}')
        for p in (HERE / 'experiment').glob('*.json'):
            copy(p, 'experiment/' + p.name)
        for name in ('REPORT_ZH.md', 'REPORT.md', 'DECISION.json', 'PILOT_AUDIT.json', 'PREDICTION_SEAL.json',
                     'EVALUATION_LOCK.json', 'EVALUATION_AUDIT.json', 'REVIEW_PANEL_SEAL.json',
                     'VISUAL_REVIEW_AUDIT.json', 'CONTROLLED_LOCK.json', 'CONTROLLED_PREDICTION_SEAL.json',
                     'CONTROLLED_EVALUATION_AUDIT.json', 'SCALAR_AUDIT.json'):
            copy(HERE / name, name)
        parts = {}
        for p in sorted(OUT.glob('*.csv')):
            rows = table(p)
            parts[p.name] = transport_csv(temp, 'results/' + p.name, rows, tuple(rows[0]))
        atomic_json(temp / 'PUBLIC_EXPORT_AUDIT.json', dict(validation=validation, tables=parts,
                    contains_raw_media_text_coordinates_instances_features_weights_or_credentials=False))
        items = [dict(path=p.relative_to(temp).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(temp.rglob('*')) if p.is_file()]
        atomic_json(temp / 'MANIFEST.json', dict(prefix=PREFIX, files=items))
        scan_release(temp, private_literals())
        temp.rename(out)
        print('R167 verified public files', len(items) + 1, flush=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def register():
    audit()
    receipt_path = HERE / 'publication/PUBLICATION_RECEIPT.json'
    receipt = read(receipt_path)
    assert receipt['repository'] == 'Zonglin-He/A' and receipt['prefix'] == PREFIX
    assert receipt['remoteByteVerified'] and receipt['immutableVerifiedBeforeRefUpdate']
    assert receipt['commit'] == receipt['mainRefObserved']
    commit = receipt['commit']
    remote = f'https://github.com/Zonglin-He/A/blob/{commit}/{PREFIX}/REPORT_ZH.md'
    verdict = read(HERE / 'DECISION.json')['finding']
    status('completed_published_verified', public_report=remote, public_commit=commit,
           report='round167/REPORT_ZH.md', decision='round167/DECISION.json',
           publication_receipt='round167/publication/PUBLICATION_RECEIPT.json')
    path = ROOT / 'research/STATE.json'
    state = read(path)
    state.update(latest_completed_round='R167', publication_state='R167_REMOTE_BYTES_VERIFIED',
                 measurement_state='R167_JOINT_BINDING_AND_PREDICATE_REPLAY_AUDITED', claim_state=verdict,
                 public_study=PREFIX, publication_receipt='round167/publication/PUBLICATION_RECEIPT.json',
                 next_action=read(HERE / 'DECISION.json')['decision'])
    state['directions']['crossplatform'].update(latest_completed_round='R167', latest_completed_publication=commit,
                                             latest_public_report=remote, latest_public_commit=commit)
    atomic_json(path, state)
    archive = ROOT / 'research/ARCHIVE.md'
    entry = (f'\n## R167 — Reference retrieval and matched language–instance joint assignment\n\n'
             f'R167 completed and published {commit}. 18,620 R165-paired records. {verdict} '
             '[Report](round167/REPORT_ZH.md) · [Verified receipt](round167/publication/PUBLICATION_RECEIPT.json)\n')
    if f'R167 completed and published {commit}' not in archive.read_text():
        with archive.open('a', encoding='utf-8') as f:
            f.write(entry)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('audit', 'export', 'register'))
    {'audit': audit, 'export': export, 'register': register}[p.parse_args().stage]()
