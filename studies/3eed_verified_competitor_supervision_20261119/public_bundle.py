"""Export the registered code, aggregate evidence and anonymous outcomes."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
FILES=('protocol.json','verifier_protocol.json','REPORT_ZH.md','EXPERIMENT_LOG.md','DECISION.json',
       'catalog.py','supervise.py','capture_features.py','compare.py','verify_added_labels.py',
       'validate_candidate.py','recording_intervals.py','audit.py','public_bundle.py')
TABLES=('all_rows.csv','seed_repeat_rows.csv','scalar_1200_rows.csv','label_coverage.csv',
        'feature_winner_verification.csv','winner_supervision.csv')
SUMMARY=('FEATURE_WINNER_VERIFICATION.json','RECORDING_INTERVALS.json','REPRESENTATION.json','SCALAR_1200.json',
         'SCALAR_240.json','SEED_REPEAT.json','SUPERVISION.json','WINNER_SUPERVISION.json')
FORBIDDEN=re.compile(r'/(?:home|mnt|media|tmp)/|(?i:gh[pousr]_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{12,}')


def checksum(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_table(src,dest):
    with src.open(newline='',encoding='utf-8') as f:
        data=csv.DictReader(f)
        allowed=tuple(data.fieldnames)
        assert not set(allowed)&{'score','risk','candidate_slot','identity','coordinates','caption','pointpath','imagepath'}
        rows=list(data)
    assert all(re.fullmatch(r'[AB]\d{2,3}',r['case']) and re.fullmatch(r'[GH]\d{2,3}',r['recording']) for r in rows)
    if src.stat().st_size>43_000:
        panels=defaultdict(list)
        for r in rows:
            panels[r['cohort']+'_'+r['platform']].append(r)
    else:
        panels={'all':rows}
    assert sum(map(len,panels.values()))==len(rows)
    for panel,part in sorted(panels.items()):
        file=dest/(src.stem+('_'+panel if panel!='all' else '')+'.csv')
        with file.open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=allowed)
            writer.writeheader()
            writer.writerows(part)


def main(dest):
    results=HERE/'results'
    a=json.loads((results/'AUDIT.json').read_text())
    assert a['status']=='PASS' and not a['protected_data_access']
    assert all(checksum(results/name)==digest for name,digest in a['files'].items())
    dest.mkdir(parents=True)
    out=dest/'results'
    out.mkdir()
    for key in FILES:
        shutil.copyfile(HERE/key,dest/key)
    for key in SUMMARY:
        shutil.copyfile(results/key,out/key)
    for key in TABLES:
        make_table(results/key,out)
    receipt={k:v for k,v in a.items() if k!='files'}
    receipt.update(local_receipt_sha256=checksum(results/'AUDIT.json'),public_hashes='MANIFEST.json')
    (out/'AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    files=[]
    for p in sorted(dest.rglob('*')):
        if not p.is_file():
            continue
        assert p.stat().st_size<48_000 and p.suffix in ('.md','.py','.csv','.json')
        assert not FORBIDDEN.search(p.read_text())
        files.append(dict(path=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=checksum(p)))
    manifest=dict(schema='verified_competitor_supervision_public_v1',files=files,raw_media=False,private_identifiers=False,
                  weights=False,coordinates=False,private_scores=False,features=False,credentials=False)
    (dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Exported verified results and methods',len(files)+1,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    main(p.parse_args().destination)
