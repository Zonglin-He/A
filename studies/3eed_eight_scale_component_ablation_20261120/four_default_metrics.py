"""Recount all paired metrics from the three public tables sealed at commit 4f6292f8."""
from collections import defaultdict
import csv
import gzip
from hashlib import sha256
from pathlib import Path
import numpy as np


OUT = Path(__file__).with_name('results')
INPUT_DIGESTS = {
    'compression_metrics.csv': '6949cdaa60d00e950a411f17c621e2fda25fdb97949f643ef71ef1fd51a15b82',
    'compression_rows.csv.gz': '45f92932da451223beb136994919531d47732e69201d318d5991d610cb4ff184',
    'ordinary_rows.csv.gz': '01ba1613068746a8b9e353e89869720e47c3cddbbe36a619f523a2bb22eac05d',
}


def metrics():
    groups = defaultdict(list)
    prior_names = ('full', 'direct_four_scales', 'slim_v1', 'terrain_v1')
    pooled = defaultdict(list)
    for name in ('ordinary_rows.csv.gz', 'compression_rows.csv.gz', 'compression_metrics.csv'):
        assert sha256((OUT/name).read_bytes()).hexdigest() == INPUT_DIGESTS[name]
    def key(row):
        return row['source'], row['target'], row['case']
    with gzip.open(OUT/'ordinary_rows.csv.gz','rt') as inp:
        original = {key(row):row for row in csv.DictReader(inp)}
    with gzip.open(OUT/'compression_rows.csv.gz','rt') as inp:
        for row in csv.DictReader(inp):
            previous = original.pop(key(row))
            assert (row['recording'],row['active']) == (previous['recording'],previous['active'])
            iou = float(row['four_without_reference_prior'])
            if row['active']=='0':
                assert iou == float(previous['full'])
            groups[row['source']+'_to_'+row['target']].append(
                [iou,float(previous['full']),float(row['direct_four_scales']),
                 float(previous['slim_v1']),float(previous['terrain_v1'])])
    assert not original and sum(map(len,groups.values()))==18620 and len(groups)==6
    rows = []
    for direction, records in groups.items():
        x = np.array(records)
        scores = 100*np.stack([(x>.25).mean(0),(x>.5).mean(0),x.mean(0)],axis=1)
        for j, arm in enumerate(prior_names,1):
            flips = []
            for t in (.25,.50):
                new, old = x[:,0]>t, x[:,j]>t
                flips.extend([int(sum(new & ~old)),int(sum(~new & old))])
            values = np.r_[scores[0],scores[0]-scores[j],flips]
            pooled[arm].append(values)
            rows.append([direction,len(x),arm,*values])
    for arm, v in pooled.items():
        a = np.stack(v)
        rows.append(['macro',18620,arm,*a[:,:6].mean(0),*a[:,6:].sum(0)])
    with (OUT/'compression_metrics.csv').open() as inp:
        registered = {r['direction']:r for r in csv.DictReader(inp) if r['arm']=='four_without_reference_prior'}
    for direction, n, arm, *v in rows:
        saved = registered[direction]
        np.testing.assert_allclose(v[:3],[float(saved[k]) for k in ('acc25','acc50','miou')],rtol=0,atol=1e-10)
        if arm=='full':
            np.testing.assert_equal(v[6:],[float(saved[k]) for k in ('rescue25','harm25','rescue50','harm50')])
    with (OUT/'four_default_comparisons.csv').open('w', newline='') as dest:
        writer = csv.writer(dest)
        writer.writerow(('direction','n','baseline','acc25','acc50','miou','delta25','delta50','delta_miou',
                         'rescue25','harm25','rescue50','harm50'))
        for row in rows:
            writer.writerow([*row[:9],*map(int,row[9:])])


if __name__ == '__main__':
    metrics()
