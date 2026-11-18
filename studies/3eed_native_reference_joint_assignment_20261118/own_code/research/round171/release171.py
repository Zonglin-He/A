"""Export only scalar data and first-party code, then register the verified release."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

from scope171 import HERE, INPUT, OUT, PRIOR_INPUT, ROOT, atomic_json, read, sha, status
from audit171 import audit
from score165 import table

sys.path.insert(0, str(ROOT / 'research/round154'))
from package154 import scan_release, transport_csv


PREFIX = 'studies/3eed_native_reference_joint_assignment_20261118'
PROGRAMS = ('scope171.py', 'select171.py', 'assignment171.py', 'predict171.py', 'review171.py',
            'score171.py', 'audit171.py', 'analysis171.py', 'release171.py')
RECEIPTS = ('SELECTION_LOCK.json', 'INPUT_SEAL.json', 'PREDICTION_LOCK.json', 'PREDICTION_SEAL.json',
            'VISUAL_REVIEW_LOCK.json', 'REVIEW_PANEL_SEAL.json', 'VISUAL_TRUTH_SEAL.json', 'SCORING_LOCK.json',
            'EVALUATION_AUDIT.json', 'SCALAR_AUDIT.json', 'MECHANISM_AUDIT.json')


def private_literals():
    tokens = set()
    for case in read(INPUT) + read(PRIOR_INPUT):
        row = case['row']
        words = [row[k] for k in ('id', 'caption', 'imagepath', 'pointpath', 'metapath', 'scene')]
        words.append(Path(row['metapath']).parent.name)
        words.extend(v['text'] for k, v in case['rewrite'].items() if k in ('B', 'C', 'D'))
        tokens.update(w.encode() for w in words if isinstance(w, str) and len(w) >= 12)
    return tokens


def export():
    result = audit()
    proof = read(HERE / 'MECHANISM_AUDIT.json')
    assert not proof['FINAL_access'] and proof['scoring'] == sha(HERE / 'EVALUATION_AUDIT.json')
    assert proof['code'] == sha(HERE / 'analysis171.py')
    assert all(sha(OUT / k) == v for k, v in proof['files'].items())
    parent = HERE / 'publication'
    parent.mkdir(exist_ok=True)
    dest = parent / 'public_export_attempt001'
    assert not dest.exists()
    temp = Path(tempfile.mkdtemp(dir=parent))
    try:
        def copy(name, relative):
            p = temp / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, p)
        for x in PROGRAMS:
            copy(x, 'own_code/research/round171/' + x)
        for x in (*RECEIPTS, 'REPORT.md', 'REPORT_ZH.md', 'DECISION.json', 'experiment/PROTOCOL.json'):
            copy(x, x)
        tables = {}
        for p in sorted(OUT.glob('*.csv')):
            rows = table(p)
            tables[p.name] = transport_csv(temp, 'results/' + p.name, rows, tuple(rows[0]))
        atomic_json(temp / 'PUBLIC_EXPORT_AUDIT.json', dict(summary=result, tables=tables,
                    raw_sensors_text_geometry_training_state_dataset_ids_or_credentials_exported=False))
        files = [dict(path=p.relative_to(temp).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                 for p in sorted(temp.rglob('*')) if p.is_file()]
        atomic_json(temp / 'MANIFEST.json', dict(prefix=PREFIX, files=files))
        scan_release(temp, private_literals())
        temp.rename(dest)
        status('local_audited_pending_publication', direction_records=result['direction_records'],
               report='round171/REPORT_ZH.md', decision='round171/DECISION.json')
        print('R171 public export files', len(files) + 1, flush=True)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def register():
    receipt = read(HERE / 'publication/PUBLICATION_RECEIPT.json')
    verdict = read(HERE / 'DECISION.json')
    assert receipt['repository'] == 'Zonglin-He/A' and receipt['prefix'] == PREFIX
    assert receipt['commit'] == receipt['mainRefObserved'] and receipt['immutableVerifiedBeforeRefUpdate'] and receipt['remoteByteVerified']
    commit = receipt['commit']
    remote = 'https://github.com/Zonglin-He/A/blob/' + commit + '/' + PREFIX + '/REPORT_ZH.md'
    relative = 'round171/publication/PUBLICATION_RECEIPT.json'
    status('completed_published_verified', direction_records=168, replay_frames=48, new_frames=36,
           public_report=remote, public_commit=commit, report='round171/REPORT_ZH.md',
           decision='round171/DECISION.json', publication_receipt=relative)
    path = ROOT / 'research/STATE.json'
    s = read(path)
    s.update(latest_completed_round='R171', measurement_state='R171_NATIVE_REFERENCE_AND_BLIND_PAIR_AUDITED',
             publication_state='R171_REMOTE_BYTES_VERIFIED', claim_state=verdict['finding'], next_action=verdict['decision'],
             public_study=PREFIX, publication_receipt=relative)
    s['directions']['crossplatform'].update(latest_completed_round='R171', latest_completed_publication=commit,
                                           latest_public_commit=commit, latest_public_report=remote)
    atomic_json(path, s)
    archive = ROOT / 'research/ARCHIVE.md'
    start = 'R171 completed and published ' + commit
    if start not in archive.read_text():
        with archive.open('a', encoding='utf-8') as stream:
            stream.write('\n## R171 — Native reference candidates and joint spatial assignment\n\n' + start + '. ' +
                         verdict['finding'] + ' [Report](round171/REPORT_ZH.md) · [Verified receipt](' + relative + ')\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('export', 'register'))
    {'export': export, 'register': register}[p.parse_args().phase]()
