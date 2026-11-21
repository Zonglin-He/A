"""Independently reconstruct the reported task and transition statistics from public scalars."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / 'results'
METRICS = ('acc25','acc50','miou')


def table(name):
    with (OUT / f'{name}.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def subset(rows, panel, target):
    def admitted(row):
        return row['separation']=='new_recording' if panel=='all_recording_separated' else row['panel']==panel

    return [r for r in rows if admitted(r) and target in ('macro',r['target'])]


def readout(rows, arm):
    groups = defaultdict(list)
    for r in rows:
        groups[r['target']].append(float(r[arm]))
    scores = [[100*mean(v>.25 for v in group),100*mean(v>.5 for v in group),100*mean(group)] for group in groups.values()]
    return tuple(mean(v) for v in zip(*scores))


def check():
    rows = table('rows')
    source = table('source_rows')
    seal = json.loads((OUT/'panel_seal.json').read_bytes())
    assert len({r['uid'] for r in rows})==len(rows)==seal['frame_count']
    by_id = {r['uid']:r for r in rows}
    assert len(source)==2*len(rows) and len({(r['uid'],r['source']) for r in source})==len(source)
    assert Counter(r['uid'] for r in source) == {uid:2 for uid in by_id}
    assert all(r['source']!=r['target'] and all(r[k]==by_id[r['uid']][k] for k in (
        'A','H','target','panel','recording','separation')) for r in source)
    assert all(0<=float(r[k])<=1 for r in rows for k in ('A','H','H_yaw_only','H_qwen_reanchored'))
    assert all(r['selected']=='True' or r['H']==r['A']==r['H_yaw_only']==r['H_qwen_reanchored'] for r in rows)
    assert all(r['fit_available']=='True' or r['Fit_window']==r['Fit_region']==r['Fit_mismatch']==r['A'] for r in rows)
    for key, summary in seal['coverage'].items():
        panel,target = key.split('/')
        group = subset(rows,panel,target)
        assert len(group)==summary['frames'] and len({r['recording'] for r in group})==summary['recordings']

    max_error = 0.
    metrics, pairs = table('metrics'),table('paired')
    for summary in metrics:
        arm = summary['arm']
        group = subset(source if arm=='B' else rows,summary['panel'],summary['target'])
        assert len(group)==int(summary['scored_rows'])
        assert len({r['uid'] for r in group})==int(summary['frames'])
        calc = readout(group,arm)
        max_error = max(max_error,*(abs(float(summary[k])-v) for k,v in zip(METRICS,calc)))
    for comparison in pairs:
        a,b = comparison['arm'],comparison['reference']
        group = subset(source if b=='B' else rows,comparison['panel'],comparison['target'])
        assert len(group)==int(comparison['rows'])
        values = [(float(r[a]),float(r[b])) for r in group]
        for threshold,k in ((.25,'25'),(.5,'50')):
            assert sum(x>threshold>=y for x,y in values)==int(comparison['rescue'+k])
            assert sum(y>threshold>=x for x,y in values)==int(comparison['harm'+k])
        assert sum(x>y+1.e-10 for x,y in values)==int(comparison['improved'])
        assert sum(y>x+1.e-10 for x,y in values)==int(comparison['worsened'])
        max_error = max(max_error,*(abs(float(comparison[k])-(x-y)) for k,x,y in zip(
            ('delta25_pp','delta50_pp','delta_miou_pp'),readout(group,a),readout(group,b))))

    surfaces = table('surface')
    detailed = table('surface_rows')
    for r in surfaces:
        group = subset(detailed,r['panel'],'macro' if r['target']=='pooled_rays' else r['target'])
        group = [g for g in group if g['scope']==r['scope'] and g['reference']==r['reference']]
        assert len(group)==int(r['frames'])
        for name in ('rays','both_hits','region_hits','reference_hits'):
            assert sum(int(g[name]) for g in group)==int(r[name])
        for k in ('region','reference'):
            if int(r['both_hits']):
                m = sum(float(g[f'{k}_common_abs_sum']) for g in group)/int(r['both_hits'])
                assert abs(m-float(r[f'{k}_common_mae_m']))<1.e-12

    assert max_error<1.e-10
    return dict(physical_frames=len(rows), directional_rows=len(source), method_metric_rows=len(metrics),
        paired_transition_rows=len(pairs), ray_summary_rows=len(surfaces), maximum_error_pp=max_error)


if __name__=='__main__':
    print(json.dumps(check(),indent=2))
