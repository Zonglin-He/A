"""Copy only public implementation, aggregate tables and anonymous paired IoUs."""
import argparse
import csv
from pathlib import Path
import re
import shutil

from study import ARMS, HERE, RESULTS, json_atomic, read, sha

CODE = ('protocol.json','reference_plan.json','IMPLEMENTATION.md','REPORT_ZH.md','DECISION.json','study.py',
        'factor_decoder.py','train_factorial.py','check_switches.py','replay_factorial.py',
        'batches.py','analyse_factorial.py','reference_truth.py','audit_factorial.py',
        'render_report.py','visual_review.py','visual_review_plan.json','build_export_factorial.py')
TABLES = ('endpoints.csv','factorial_summary.csv','reference_diagnostics.csv','reference_by_direction.csv',
          'identity_coverage.json','blind_switchable_review.json','recording_effects.csv','training_receipt.json','independent_audit.json')
FORBIDDEN = re.compile(r'/(?:home|mnt|media|tmp)/|(?i:gh[pousr]_|github_pat_|sk-(?:proj-)?)[A-Za-z0-9_-]{12,}')


def split_csv(source,folder,rows_per_file,outcomes=False):
    with source.open(newline='',encoding='utf-8') as f:
        p = csv.DictReader(f)
        header = p.fieldnames
        rows = list(p)
    if outcomes:
        assert tuple(header)==('parsed_reference','direct_left_right',*ARMS,
                               'best_subject_slot_iou','zero_pair_iou','best_reference_column_iou')
        for r in rows:
            assert all(r[k] in ('0','1') for k in ('parsed_reference','direct_left_right'))
            assert all(0<=float(r[k])<=1 for k in header[2:])
    for j in range(0,len(rows),rows_per_file):
        with (folder/f'{source.stem}_{j//rows_per_file:02d}.csv').open('x',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,header,lineterminator='\n')
            w.writeheader()
            w.writerows(rows[j:j+rows_per_file])
    return len(rows)


def export(dest):
    cert=read(RESULTS/'independent_audit.json')
    assert cert['n']==18620 and not cert['FINAL_access']
    assert all(x['threshold_disagreements']==0 for x in cert['directions'].values())
    visible=dest/'results'
    rows=visible/'paired_scalar_rows'
    contrasts=visible/'detailed_contrasts'
    rows.mkdir(parents=True)
    contrasts.mkdir()
    for f in CODE:
        shutil.copyfile(HERE/f,dest/f)
    for f in TABLES:
        shutil.copyfile(RESULTS/f,visible/f)
    total=sum(split_csv(f,rows,240,outcomes=True) for f in sorted(RESULTS.glob('rows_*.csv')))
    n_contrasts=split_csv(RESULTS/'factorial_contrasts.csv',contrasts,155)
    assert total==cert['n']
    manifest=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file():
            assert p.stat().st_size<46000 and p.suffix in ('.py','.json','.md','.csv')
            assert not FORBIDDEN.search(p.read_text(encoding='utf-8'))
            manifest.append(dict(path=p.relative_to(dest).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    json_atomic(dest/'MANIFEST.json',dict(files=manifest,paired_rows=total,contrast_rows=n_contrasts,
                raw_sensor_data=False,raw_text=False,private_ids=False,coordinates_or_features=False,
                weights=False,third_party_snapshot=False,credentials=False))
    print('PUBLIC_EXPORT',len(manifest)+1,total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('dest',type=Path)
    export(p.parse_args().dest)
