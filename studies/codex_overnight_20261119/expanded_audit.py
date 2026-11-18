"""Recompute the ordinary comparison and leave out one recording at a time."""
from collections import Counter, defaultdict
import csv
from math import fsum
from pathlib import Path

from expanded import ARMS, DATA, HERE, atomic_json, read, sha
from edge_score import METRICS, csv_atomic


MAIN='role_exchange_joint'
CONTROLS=tuple(a for a in ARMS if a!=MAIN)


def scalar_triplet(iou):
    return iou, float(iou>.25), float(iou>.50)


def summarized(items, key, omitted=None):
    by_direction=defaultdict(list)
    for x in items:
        if x['recording']==omitted:
            continue
        by_direction[(x['source'],x['target'])].append(x[key])
    means=[]
    for x in by_direction.values():
        means.append(tuple(fsum(col)/len(x) for col in zip(*x,strict=True)))
    average=tuple(100*fsum(v)/len(means) for v in zip(*means,strict=True)) if means else (0.,)*3
    return average,len(means),sum(map(len,by_direction.values()))


def records():
    root=HERE/'results/expanded'
    cert=read(root/'SCORE_SEAL.json')
    assert all(sha(root/f)==dig for f,dig in cert['files'].items())
    with (root/'rows.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==cert['rows']*len(ARMS) and cert['physical_frames']*2==cert['rows']
    d={}
    for r in rows:
        slot = str(r['frame']), r['source']
        arr=d.setdefault(slot,{})
        assert r['arm'] not in arr
        arr[r['arm']]=r
    assert len(d)==cert['rows'] and all(Counter(a.keys())==Counter(ARMS) for a in d.values())
    groups=defaultdict(set)
    for f,s in d:
        groups[f].add(s)
    assert len(groups)==cert['physical_frames'] and all(len(x)==2 for x in groups.values())
    return root,cert,d


def recheck():
    root,cert,d=records()
    assert not (root/'INDEPENDENT_AUDIT.json').exists()
    base=[]
    for entry in d.values():
        m=entry[MAIN]
        expected=scalar_triplet(float(m['official_iou']))
        score_fields={k:scalar_triplet(float(v['official_iou'])) for k,v in entry.items()}
        if m['rewrite']=='0':
            assert all(x==expected for x in score_fields.values())
        base.append(dict(frame=m['frame'],recording=m['recording'],source=m['source'],target=m['target'],
                         rewrite=int(m['rewrite']), prior_recording=int(m['prior_mechanism_recording']),
                         **score_fields,**{a+'_delta':tuple(x-y for x,y in zip(expected,score_fields[a],strict=True))
                                          for a in CONTROLS}))
    scopes={'all':base,'relation':[x for x in base if x['rewrite']],
            'novel_recording':[x for x in base if not x['prior_recording']]}
    old=list(csv.DictReader((root/'metrics.csv').open()))
    expected={(x['scope'],x['arm']):x for x in old if x['route']=='direction_macro'}
    comparables={'all':'all','relation':'relation','novel_recording':'new_recordings'}
    audits=[]
    for name,group in scopes.items():
        for arm in ARMS:
            measured,n,count=summarized(group,arm)
            prior=expected[comparables[name],arm]
            assert n==int(prior['directions']) and count==int(prior['n'])
            assert all(abs(a-float(prior[k]))<1e-10 for k,a in zip(METRICS,measured,strict=True))
        for rec in sorted({x['recording'] for x in group}):
            for arm in CONTROLS:
                values,directions,size=summarized(group,arm+'_delta',omitted=rec)
                audits.append(dict(scope=name,excluded_recording=rec,records=size,remaining_directions=directions,
                                   arm=MAIN,comparison=arm,**dict(zip(METRICS,values,strict=True))))
    csv_atomic(root/'recording_sensitivity.csv',audits)
    stamp=read(DATA/'ROSTER_SEAL.json')
    report=dict(rows=cert['rows'],frames=cert['physical_frames'],paired_metrics_all_rechecked=True,
                unique_recordings=stamp['historical_recordings'],new_mechanism_recordings=stamp['new_recordings'],
                record_omission_is_descriptive_sensitivity_only=True,
                source_metric_seal_sha256=sha(root/'SCORE_SEAL.json'),recording_file_sha256=sha(root/'recording_sensitivity.csv'),
                iou_changes_vs_additive=sum(x[MAIN][0]!=x['same_evidence_geometric'][0] for x in base),
                protected_data_access=False)
    atomic_json(root/'INDEPENDENT_AUDIT.json',report)
    print(report,flush=True)


if __name__=='__main__':
    recheck()
