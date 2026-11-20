"""Recount public row metrics and all paired threshold crossings without any model or private data."""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

DIR = Path(__file__).resolve().parent/'results'


def rows(path):
    opener = gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt') as f:
        return list(csv.DictReader(f))


def by_direction(table):
    result=defaultdict(list)
    for row in table:
        result[row['direction']].append(row)
    return result


def run(stage):
    measurements=rows(DIR/f'{stage}_metrics.csv')
    paired=rows(DIR/f'{stage}_paired.csv')
    raw=rows(DIR/f'{stage}_rows.csv.gz')
    expected={'source':384,'pilot':3072,'ordinary':18620}[stage]
    assert len(raw)==len(set(row['row'] for row in raw))==expected
    scopes=by_direction(raw)
    assert len(scopes)==(3 if stage=='source' else 6)
    if stage=='pilot':
        assert all(len(group)==512 for group in scopes.values())
    maximum=0.
    for result in measurements:
        arm=result['arm']
        groups=scopes.values() if result['direction']=='macro' else [scopes[result['direction']]]
        assert sum(map(len,groups))==int(result['n'])
        for metric,threshold in (('Acc25',.25),('Acc50',.50),('mIoU',None)):
            per=[]
            for group in groups:
                values=[float(row[arm+'_iou']) for row in group]
                per.append(mean(v>threshold for v in values) if threshold is not None else mean(values))
            maximum=max(maximum,abs(100*mean(per)-float(result[metric])))
    for result in paired:
        group=raw if result['direction']=='macro' else scopes[result['direction']]
        a,b=result['arm'],result['baseline']
        for threshold,key in ((.25,'25'),(.50,'50')):
            delta=[int(float(row[a+'_iou'])>threshold)-int(float(row[b+'_iou'])>threshold) for row in group]
            assert delta.count(1)==int(result['rescue'+key]) and delta.count(-1)==int(result['harm'+key])
        magnitude=[float(row[a+'_iou'])-float(row[b+'_iou']) for row in group]
        assert sum(v>1e-10 for v in magnitude)==int(result['better'])
        assert sum(v< -1e-10 for v in magnitude)==int(result['worse'])
        if b=='v1':
            assert sum(int(row[a+'_switch']) for row in group)==int(result['switches'])
    assert maximum<1e-8
    result=dict(rows=expected, comparisons=len(paired), absolute_method_tables=len(measurements),
                max_metric_error_pp=maximum,paired_threshold_counts_exact=True)
    (DIR/f'{stage}_recount.json').write_text(json.dumps(result,indent=2)+'\n')
    print(stage,result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=('source','pilot','ordinary'))
    run(p.parse_args().stage)
