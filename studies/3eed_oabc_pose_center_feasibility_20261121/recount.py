"""Cross-check the exported scalar tables without loading any private data."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median


OUT = Path(__file__).resolve().parent/'results'


def table(n):
    with (OUT/f'{n}.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def group(records,row):
    def eligible(r):
        if row['panel']=='all':
            return True
        if row['panel']=='all_recording_separated':
            return r['separation']=='new_recording'
        return r['panel']==row['panel']
    return [r for r in records if row['target'] in ('macro',r['target']) and eligible(r)]


def readout(g,arm):
    values=defaultdict(list)
    for r in g:
        values[r['target']].append(float(r[arm]))
    triplet=[(mean(v>.25 for v in x),mean(v>.50 for v in x),mean(x)) for x in values.values()]
    return tuple(100.*mean(v) for v in zip(*triplet))


def verify():
    rows,poses,rays = table('rows'),table('poses'),table('ray_rows')
    seal=json.loads((OUT/'prediction_seal.json').read_text())
    assert len(rows)==len({r['uid'] for r in rows})==seal['frames']==416
    matched = {r['uid'] for r in rows if r['fit_node']=='True'}
    assert len(matched)==seal['fit_node_exact']==322 and set(Counter(r['uid'] for r in poses).values())=={2}
    assert {r['uid'] for r in poses}==matched
    by_frame=defaultdict(list)
    for p in poses:
        by_frame[p['uid']].append(p)
        if p['robust_truth']=='True':
            assert p['robust_feasible']=='True'
        if p['near_truth']=='True':
            assert p['robust_truth']=='True' and p['solved']=='True'
        if p['solved']=='True':
            assert float(p['fit_oabc_loss_m']) <= float(p['fit_geometry_loss_m'])+.05+1.e-9
    for r in rows:
        if r['fit_node']=='True':
            assert r['robust_truth']==str(any(p['robust_truth']=='True' for p in by_frame[r['uid']]))
            assert r['near_truth']==str(any(p['near_truth']=='True' for p in by_frame[r['uid']]))
            assert r['gt_both']==str(any(p['gt_both']=='True' for p in by_frame[r['uid']]))
    e=0.
    m,b,c,a = table('metrics'),table('paired'),table('coverage'),table('ray_audit')
    for r in m:
        met=[x for x in group(rows,r) if x[r['arm']]]
        assert len(met)==int(r['n'])
        y=readout(met,r['arm'])
        e=max(e,*(abs(x-float(r[k])) for k,x in zip(('acc25','acc50','miou'),y)))
        cen=[float(x[r['arm']+'_center_m']) for x in met if x.get(r['arm']+'_center_m')]
        if cen:
            assert abs(median(cen)-float(r['center_median_m']))<1.e-10
    for r in b:
        arm,ref=r['arm'],r['reference']
        g=group(rows,r)
        xs=[(float(x[arm]),float(x[ref])) for x in g]
        assert len(g)==int(r['n'])
        for t,s in ((.25,'25'),(.5,'50')):
            assert int(r['rescue'+s])==sum(a>t>=b for a,b in xs)
            assert int(r['harm'+s])==sum(b>t>=a for a,b in xs)
    for r in c:
        g=group(rows,r)
        assert int(r['n'])==len(g)
        assert int(r['fit_node'])==sum(x['fit_node']=='True' for x in g)
        for k in ('strict_truth','robust_truth','near_truth','robust_feasible','gt_both'):
            assert int(r[k])==sum(x[k]=='True' for x in g)
        wrong=[x for x in g if x['fit_node']=='True' and float(x['D'])<=.5]
        assert int(r['bad_D_at_fit_node'])==len(wrong)
        assert int(r['missed_inside_domain'])==sum(x['robust_truth']=='True' for x in wrong)
        assert int(r['domain_can_rescue'])==sum(x['robust_oracle']!='' and float(x['robust_oracle'])>.5 for x in wrong)
    for r in a:
        source=[x for x in group(rays,r) if x['arm']==r['arm']]
        n=sum(int(x['n']) for x in source)
        assert n==int(r['n']) and len(source)==int(r['frames'])
        assert abs(sum(float(x['sum_cost_m']) for x in source)/n-float(r['clipped_mae_m']))<1.e-10
    assert e<1.e-10 and seal['ray_isolation']['retained_withheld_raw']==0
    return dict(frames=len(rows),pose_records=len(poses),metrics=len(m),pairs=len(b),coverage=len(c),
                ray_audits=len(a),max_metric_gap_pp=e)


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
