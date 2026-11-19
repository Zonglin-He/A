"""Allowlist for publication of results and code, with immutable byte inventory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent

FILES = ('REPORT_ZH.md', 'DECISION.json', 'protocol.json', 'IMPLEMENTATION.md', 'common.py',
         'prepare.py', 'evidence.py', 'decoder.py', 'train.py', 'predict.py', 'score.py',
         'independent_audit.py', 'diagnostics.py', 'infer.py', 'check_implementation.py', 'report.py', 'build_export.py')
RESULTS = ('endpoints.csv', 'paired_deltas.csv', 'point_support.csv', 'recording_signs.csv',
           'source_diagnostics.csv', 'prior_verified_identity_panel.csv', 'identity_panel.json', 'slot_ceiling.csv',
           'training_curves.csv', 'independent_audit.json')
ARMS = ('r158_pca', 'joint', 'without_feedback', 'without_rgb', 'fixed_proposals')
FORBIDDEN = re.compile(r'/(?:home|mnt|media|tmp)/|(?i:gh[pousr]_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{12,}')


def checksum(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def shard(source, dest):
    with source.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames) == ('has_reference_mention', 'direct_left_right', *ARMS)
        rows = list(reader)
    for r in rows:
        assert all(r[k] in ('0', '1') for k in ('has_reference_mention', 'direct_left_right'))
        assert all(0 <= float(r[arm]) <= 1 for arm in ARMS)
    # Short immutable objects keep repository API fetch/verification below its content limit.
    for i in range(0, len(rows), 370):
        path = dest/f'{source.stem}_{i//370:02d}.csv'
        with path.open('x', newline='', encoding='utf-8') as out:
            writer = csv.DictWriter(out, reader.fieldnames, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows[i:i+370])
    return len(rows)


def export(dest):
    res = HERE/'results'
    audit = json.loads((res/'independent_audit.json').read_text())
    assert audit['n'] == 18620 and not audit['FINAL_access']
    assert all(v['threshold_disagreements'] == [0, 0] for v in audit['directions'].values())
    dest.mkdir(parents=True)
    agg = dest/'results'
    scalars = agg/'paired_scalar_rows'
    scalars.mkdir(parents=True)
    for filename in FILES:
        shutil.copyfile(HERE/filename, dest/filename)
    for filename in RESULTS:
        shutil.copyfile(res/filename, agg/filename)
    n = sum(shard(src, scalars) for src in sorted(res.glob('paired_rows_*.csv')))
    assert n == audit['n']
    files = []
    for f in sorted(dest.rglob('*')):
        if f.is_file():
            assert f.stat().st_size < 46_000
            assert f.suffix in ('.json', '.md', '.csv', '.py')
            assert not FORBIDDEN.search(f.read_text())
            files.append(dict(path=f.relative_to(dest).as_posix(), bytes=f.stat().st_size, sha256=checksum(f)))
    manifest = dict(schema='3eed_joint_role_support_v1', files=files, paired_rows=n,
                    sensors=False, private_text=False, sample_identifiers=False,
                    boxes_or_features=False, weights=False, third_party_snapshot=False, credentials=False)
    with (dest/'MANIFEST.json').open('x', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    print('EXPORT_COMPLETE', len(files)+1, n, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output', type=Path)
    export(p.parse_args().output)
