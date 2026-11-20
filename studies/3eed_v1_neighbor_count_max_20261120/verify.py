"""Exact subset enumeration and an independent recount of all released source decisions."""
from collections import Counter,defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
import csv
import gzip
import hashlib
import json
from statistics import mean,median


OUT=Path(__file__).resolve().parent/'results'


def check_order_statistics():
    worst=0.
    tests=0
    for a in ((.2,0.,.4,.4,1.1,3.),(0.,0.,0.,.5,.5,1.),(4.,0.,.2,.2,.2,3.)):
        center=a[0]
        others=sorted(a[1:])
        for n in range(2,len(a)+1):
            exact=Counter(max(center,*selected) for selected in combinations(others,n-1))
            terms=Counter()
            for rank in range(n-1,len(a)):
                terms[max(center,others[rank-1])]+=comb(rank-1,n-2)
            assert dict(exact)==dict(terms)
            assert sum(terms.values())==comb(len(others),n-1)
            tests+=1
    # A finite-source risk identity with tied ReLU values, mixed neighborhoods and bins.
    frames=[[.5,0.,1.,1.,3.],[0.,.1,.7,1.3,2.], [1.,0.,1.,1.,4.5]]
    for n in (2,3,4,5):
        by=defaultdict(list)
        for x in frames:
            for sub in combinations(x[1:],n-1):
                m=max(x[0],*sub)
                by[int(m>1.)].append(max(x)-m)
        residuals=[(r,mean(v)) for v in by.values() for r in v]
        benefit=mean(r*r-(r-d)**2 for r,d in residuals)
        optimum=mean(d*d for r,d in residuals)
        worst=max(worst,abs(benefit-optimum))
    assert worst<1e-12
    return dict(exact_thinning_distributions=tests,max_empirical_risk_identity_error=worst)


def run():
    path=OUT/'deletion_all_rows.csv.gz'
    with gzip.open(path,'rt') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==384*7 and len({(r['source'],r['frame'],r['arm']) for r in rows})==len(rows)
    case_map=defaultdict(dict)
    for r in rows:
        case_map[r['arm']][r['source'],r['frame']]=r
    for arm,case in case_map.items():
        assert set(case)==set(case_map['native'])
    with (OUT/'deletion_metrics.csv').open() as f:
        for metric in csv.DictReader(f):
            groups=defaultdict(list)
            for r in case_map[metric['arm']].values():
                if metric['source'] in (r['source'],'macro'):
                    groups[r['source']].append(r)
            for threshold in (25,50):
                rate=mean(mean(float(r['iou'])>threshold/100 for r in group) for group in groups.values())*100
                assert abs(rate-float(metric['acc'+str(threshold)]))<1e-10
                pairs=[(float(r['frozen_iou'])>threshold/100,float(r['iou'])>threshold/100)
                       for group in groups.values() for r in group]
                assert pairs.count((False,True))==int(metric['rescue'+str(threshold)])
                assert pairs.count((True,False))==int(metric['harm'+str(threshold)])
    with gzip.open(OUT/'source_neighborhoods.csv.gz','rt') as f:
        hist=list(csv.DictReader(f))
    assert len(hist)==384*16*3
    assert sum(int(v['groups']) for v in hist if v['region']=='scene')==384*2048
    gates=json.loads((OUT/'deletion_decision.json').read_text())
    assert gates['advance_to_source_calibration']==all(any(float(r['iou'])<=.5<float(r['frozen_iou'])
           for r in case_map[arm].values()) for arm in ('n4_211','n4_947'))
    details=[]
    for arm,bank in case_map.items():
        if arm=='v1':
            continue
        for name,rr in [('all',list(bank.values())),('old192',[r for r in bank.values() if r['exposed192']=='1']),
                        ('other192',[r for r in bank.values() if r['exposed192']=='0'])]:
            misses=[r for r in rr if r['opportunity50']=='1']
            rescue=sum(float(r['iou'])>.5 for r in misses)
            narrow=sum(float(r['score_change_span'])>=-float(r['frozen_margin50']) for r in misses)
            helpful=sum(float(r['margin50'])>float(r['frozen_margin50']) for r in misses)
            localized=[r for r in rr if int(r['winner_eligible'])>0]
            details.append(dict(arm=arm,panel=name,n=len(rr),opportunities=len(misses),rescued=rescue,
                       potentially_crossed_by_score_span=narrow,misses_with_improved_margin=helpful,
                       frames_with_full_center_inside_winner=len(localized),
                       eligible_mse=sum(float(r['scene_sse']) for r in rr)/sum(int(r['scene_eligible']) for r in rr),
                       score_change_span_median=median(float(r['score_change_span']) for r in rr),
                       frozen_missed_margin_median=median(-float(r['frozen_margin50']) for r in misses) if misses else None))
    with (OUT/'score_propagation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,details[0]);writer.writeheader();writer.writerows(details)
    receipt=dict(rows=len(rows),centers=384*2048,unique_source_records=384,recounted_pairs=True,
                 row_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**check_order_statistics())
    (OUT/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)


if __name__=='__main__':
    run()
