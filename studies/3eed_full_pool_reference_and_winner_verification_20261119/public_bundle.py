"""Publish the locked code, aggregate evidence and anonymous per-case outcomes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
BASE = ('REPORT_ZH.md','EXPERIMENT_LOG.md','DECISION.json','protocol.json','inputs.py',
        'winner_audit.py','rank.py','verify_winner.py','run.py','audit.py','numerical_check.py','public_bundle.py')
SUMMARY = ('SUMMARY.json','WINNER_AUDIT.json','NUMERICAL_CHECK.json')
ROWS = ('winner_replay.csv','rank_rows.csv','verification_rows.csv')
SENSITIVE = re.compile(r'/(?:home|mnt|media|tmp)/|(?i:gh[pousr]_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{12,}')
SENSITIVE_COLUMNS = frozenset(('caption','image','metapath','identity','pointpath','coordinates','candidate_slot','risk_score'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(original, output):
    with original.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        safe = [key for key in reader.fieldnames if key not in SENSITIVE_COLUMNS]
        assert set(reader.fieldnames) - set(safe) <= {'risk_score'}
        groups = defaultdict(list)
        for r in reader:
            assert re.fullmatch(r'[AB]\d{2,3}',r['case'])
            assert re.fullmatch(r'[GH]\d{2,3}',r['recording'])
            assert r['platform'] in ('waymo','drone','quad')
            assert r['cohort'] in ('initial','enriched')
            groups[r['cohort']+'_'+r['platform']].append(r)
    if original.stem == 'winner_replay':
        groups = {'all':[r for group in groups.values() for r in group]}
    for g, rows in sorted(groups.items()):
        file = output/(original.stem+('_'+g if g!='all' else '')+'.csv')
        with file.open('x',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=safe,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)


def main(dest):
    results = HERE/'results'
    audit = json.loads((results/'AUDIT.json').read_text())
    assert audit['status']=='PASS' and audit['higher_budget_replay_verified'] and not audit['protected_data_access']
    assert set(audit['files'])==set(SUMMARY+ROWS)
    assert all(sha(results/name)==digest for name,digest in audit['files'].items())
    dest.mkdir(parents=True)
    out = dest/'results'
    out.mkdir()
    for n in BASE:
        shutil.copyfile(HERE/n,dest/n)
    for n in SUMMARY:
        shutil.copyfile(results/n,out/n)
    for n in ROWS:
        partition(results/n,out)
    metadata = {k:v for k,v in audit.items() if k!='files'}
    metadata['local_audit_sha256'] = sha(results/'AUDIT.json')
    metadata['public_bytes_manifest'] = 'MANIFEST.json'
    (out/'AUDIT.json').write_text(json.dumps(metadata,indent=2)+'\n')
    files = []
    for p in sorted(dest.rglob('*')):
        if not p.is_file():
            continue
        assert p.suffix in ('.csv','.md','.py','.json') and p.stat().st_size<48_000
        assert not SENSITIVE.search(p.read_text())
        files.append(dict(path=p.relative_to(dest).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    inventory = dict(schema='full_pool_source_public_v1',files=files,
                     private_data=False,weights=False,raw_identities=False,features=False,
                     coordinates=False,private_scores=False,credentials=False)
    (dest/'MANIFEST.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('Verified public artifacts',len(files)+1)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('dest',type=Path)
    main(cli.parse_args().dest)
