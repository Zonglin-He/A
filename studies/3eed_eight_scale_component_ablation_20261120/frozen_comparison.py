"""Join the pinned historical official validation metrics by source, target and population."""
import argparse
import csv
import hashlib
from pathlib import Path
from common import OUT
from score_roles import csv_file

EXPECTED="14e52420a616028816220141f99ca6911d95930d0c0fc09f6ae367cb9ca66105"


def run(path):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==EXPECTED
    with (OUT/'metrics.csv').open() as f:
        metrics={r['direction']:r for r in csv.DictReader(f) if r['arm']=='full' and r['subset']=='all'}
    output=[]
    for r in csv.DictReader(path.open()):
        if r['aggregation'] not in ('direction','six_direction_macro'):continue
        if r['arm'] not in ('official_bbf_axis','official_bbf_pca','native_text_axis','native_text_pca'):continue
        d='macro' if r['aggregation']=='six_direction_macro' else r['group']
        a=metrics[d]
        assert r['n']==a['n']
        result=dict(direction=d,records=int(r['n']),frozen=r['arm'])
        for key,old in [('acc25','Acc25'),('acc50','Acc50'),('miou','mIoU')]:
            result['frozen_'+key]=float(r[old]);result['main_'+key]=float(a[key])
            result['delta_'+key]=float(a[key])-float(r[old])
        output.append(result)
    assert len(output)==28
    csv_file(OUT/'frozen_comparison.csv',output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('published_metrics',type=Path);run(p.parse_args().published_metrics)
