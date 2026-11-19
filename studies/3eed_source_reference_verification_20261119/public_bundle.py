"""Export only source-verification methods, anonymous outcomes, and evidence digests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
SOURCES = ('protocol.json','scoring_plan.json','extension_plan.json','REPORT_ZH.md','EXPERIMENT_LOG.md','DECISION.json',
           'scope.py','prepare_source.py','review.py','predict.py','score.py','extension.py','audit.py','public_bundle.py')
METRICS = ('SUMMARY.json','EXTENSION_SUMMARY.json','AUDIT.json','reference_outcomes.csv','recording_outcomes.csv',
           'enriched_reference_outcomes.csv')
DROP_FIELDS = frozenset(('reference_points','missing_probability'))
PRIVATE_VALUES = re.compile(r'/(?:home|mnt|media|tmp)/|(?i:gh[pousr]_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{12,}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(dest):
    result = HERE / 'results'
    audit = json.loads((result/'AUDIT.json').read_text())
    assert audit['status'] == 'PASS' and not audit['protected_data_access']
    assert all(digest(result/name) == value for name,value in audit['files'].items())
    dest.mkdir(parents=True)
    out = dest / 'results'
    out.mkdir()
    for item in SOURCES:
        shutil.copyfile(HERE/item, dest/item)
    for item in METRICS:
        src = result/item
        if src.suffix == '.csv':
            with src.open(newline='',encoding='utf-8') as original, (out/item).open('x',newline='',encoding='utf-8') as new:
                dataset = csv.DictReader(original)
                fields = [f for f in dataset.fieldnames if f not in DROP_FIELDS]
                assert not {'image','points','caption','pointpath','metapath','identity','coordinates','model_features'} & set(fields)
                checked = csv.DictWriter(new,fieldnames=fields,extrasaction='ignore')
                checked.writeheader()
                checked.writerows(dataset)
        elif item == 'AUDIT.json':
            published = {k:v for k,v in audit.items() if k != 'files'}
            published['local_audit_sha256'] = digest(src)
            published['published_files_integrity'] = 'MANIFEST.json'
            (out/item).write_text(json.dumps(published,indent=2)+'\n')
        else:
            shutil.copyfile(src,out/item)
    manifest = []
    for p in sorted(dest.rglob('*')):
        if p.is_dir():
            continue
        content = p.read_text()
        assert not PRIVATE_VALUES.search(content)
        assert p.suffix in ('.py','.csv','.md','.json')
        manifest.append(dict(path=p.relative_to(dest).as_posix(),sha256=digest(p),bytes=p.stat().st_size))
    record = dict(schema='source_reference_study_public_v1',files=manifest,private_images=False,
                  raw_ids=False,coordinates=False,reference_scores_or_features=False,weights=False,secrets=False)
    (dest/'MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Export',len(manifest)+1,'files',flush=True)


if __name__=='__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('destination',type=Path)
    export(args.parse_args().destination)
