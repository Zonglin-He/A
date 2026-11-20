"""Reproduce task metrics, threshold switches, coverage and recording strata from public scalars."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parent/'results'


def records(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as f:
        return list(csv.DictReader(f))


def run(stage):
    table=records(OUT/(stage+'_rows.csv.gz'))
    expected={'source':384,'pilot':3072,'ordinary':18620}[stage]
    assert len(table)==len({r['row'] for r in table})==expected
    by_direction=defaultdict(list)
    by_recording=defaultdict(list)
    for r in table:
        by_direction[r['direction']].append(r)
        by_recording[r['direction'],r['recording']].append(r)
    assert len(by_direction)==(3 if stage=='source' else 6)
    if stage=='pilot':assert {len(v) for v in by_direction.values()}=={512}
    metrics=records(OUT/(stage+'_metrics.csv'))
    pairs=records(OUT/(stage+'_paired.csv'))
    coverage=records(OUT/(stage+'_coverage.csv'))
    max_error=0.
    for x in metrics:
        groups=list(by_direction.values()) if x['direction']=='macro' else [by_direction[x['direction']]]
        for field,tau in [('Acc25',.25),('Acc50',.5),('mIoU',None)]:
            averages=[]
            for rows in groups:
                a=[float(r[x['arm']+'_iou']) for r in rows]
                averages.append(mean(a) if tau is None else mean(v>tau for v in a))
            max_error=max(max_error,abs(float(x[field])-100*mean(averages)))
    for x in pairs:
        rows=table if x['direction']=='macro' else by_direction[x['direction']]
        a,b=x['arm']+'_iou',x['baseline']+'_iou'
        delta=[float(r[a])-float(r[b]) for r in rows]
        assert delta and sum(d>1e-10 for d in delta)==int(x['better']) and sum(d< -1e-10 for d in delta)==int(x['worse'])
        for t,tau in [('25',.25),('50',.5)]:
            transitions=[int(float(r[a])>tau)-int(float(r[b])>tau) for r in rows]
            assert transitions.count(1)==int(x['rescue'+t]) and transitions.count(-1)==int(x['harm'+t])
    for x in coverage:
        rows=table if x['direction']=='macro' else by_direction[x['direction']]
        arm=x['arm']
        for t,tau in [('25',.25),('50',.5)]:
            counts=defaultdict(int)
            for r in rows:
                a,b=float(r['v1_best4'])>tau,float(r[arm+'_best4'])>tau
                old,new=float(r['v1_iou'])>tau,float(r[arm+'_iou'])>tau
                pool=float(r['pool_best'])>tau
                outside=bool(int(r[arm+'_outside']))
                counts['gained_coverage']+=b and not a
                counts['lost_coverage']+=a and not b
                counts['outside_opportunity']+=pool and not a
                counts['rescued_no_old_good']+=new and not a
                counts['outside_rescue']+=new and not old and outside
                counts['outside_harm']+=old and not new and outside
                counts['four_covered']+=b
                counts['hits']+=new
                assert not old or a
                assert not new or b
                assert not b or pool
            for key,v in counts.items():assert v==int(x[key+t]),(key,x,v)
    arms=[x['arm'] for x in metrics if x['direction']=='macro']
    rec=[]
    for (direction,recording),rows in sorted(by_recording.items()):
        for arm in arms:
            entry=dict(direction=direction,recording=recording,n=len(rows),arm=arm)
            for t,tau in [('25',.25),('50',.5)]:
                d=[int(float(r[arm+'_iou'])>tau)-int(float(r['v1_iou'])>tau) for r in rows]
                entry.update({'hits'+t:sum(float(r[arm+'_iou'])>tau for r in rows),
                              'rescue'+t:d.count(1),'harm'+t:d.count(-1)})
            rec.append(entry)
    with (OUT/(stage+'_recordings.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rec[0].keys());w.writeheader();w.writerows(rec)
    assert max_error<1e-8
    result=dict(n=expected,paired_tables_checked=len(pairs),coverage_tables_checked=len(coverage),
                metric_max_error_pp=max_error,recording_by_direction_groups=len(by_recording))
    (OUT/(stage+'_recount.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(stage,result)


if __name__=='__main__':
    cmd=argparse.ArgumentParser();cmd.add_argument('stage',choices=('source','pilot','ordinary'))
    run(cmd.parse_args().stage)
