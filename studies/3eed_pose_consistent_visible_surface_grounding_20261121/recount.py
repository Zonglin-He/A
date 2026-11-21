"""Independently recount paired and audited results from the exported scalars."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent / 'results'


def table(name):
    with (OUT/f'{name}.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def select(rows, panel, target):
    return [r for r in rows if target in ('macro',r['target']) and
            (r['separation']=='new_recording' if panel=='all_recording_separated' else r['panel']==panel)]


def metrics(rows, arm):
    by_platform = defaultdict(list)
    for r in rows:
        by_platform[r['target']].append(float(r[arm]))
    return tuple(100*mean(x) for x in zip(*[(mean(v>.25 for v in xs),mean(v>.5 for v in xs),mean(xs))
                                             for xs in by_platform.values()]))


def audit():
    rows=table('rows')
    seal=json.loads((OUT/'prediction_seal.json').read_text())
    assert len({r['uid'] for r in rows})==len(rows)==seal['frames']==416
    assert seal['isolation']['retained_withheld_raw']==0
    for r in rows:
        assert r['P_oracle'] in (r['P'],r['Q_joint']) and float(r['P_oracle'])==max(float(r['P']),float(r['Q_joint']))
        assert r['P_transfer']==r['Q_joint' if r['transfer']=='True' else 'P']
        if r['transfer']=='True':
            assert float(r['fit_match'])>=.5 and r['F_qwen_selected']=='True'
        for family in ('F','C'):
            p,q,runner,oracle=(r[f'{family}_{k}'] for k in ('pca','qwen','ray','oracle'))
            assert float(oracle)==max(float(p),float(q))
            chooses_q=r[family+'_qwen_selected']=='True'
            allowed=int(r['selection_rays'])>=5
            pref=r[family+'_selection_preference']
            assert (pref!='')==allowed
            assert chooses_q==(allowed and float(pref)>1.e-12)
            assert runner==(q if chooses_q else p)

    original, pairs, selection = table('metrics'),table('paired'),table('selection')
    peak_error=0.
    for x in original:
        group=select(rows,x['panel'],x['target'])
        assert len(group)==int(x['n'])
        values=metrics(group,x['arm'])
        peak_error=max(peak_error,*(abs(float(x[k])-v) for k,v in zip(('acc25','acc50','miou'),values)))
    for x in pairs:
        arm,base=x['arm'],x['reference']
        group=select(rows,x['panel'],x['target'])
        if x['scope']=='both_supported':
            group=[r for r in group if r['full']=='True' and r[arm+'_used']=='True']
        assert len(group)==int(x['n'])
        compared=[(float(r[arm]),float(r[base])) for r in group]
        for th,tag in ((.25,'25'),(.5,'50')):
            assert int(x['rescue'+tag])==sum(a>th>=b for a,b in compared)
            assert int(x['harm'+tag])==sum(b>th>=a for a,b in compared)
        ar,ref=metrics(group,arm),metrics(group,base)
        peak_error=max(peak_error,*(abs(float(x[k])-a+b) for k,a,b in zip(
            ('delta25_pp','delta50_pp','delta_miou_pp'),ar,ref)))
    for x in selection:
        name=x['family']
        group=[r for r in select(rows,x['panel'],x['target']) if int(r['selection_rays'])>=5]
        assert len(group)==int(x['eligible'])
        assert sum(r[name+'_qwen_selected']=='True' for r in group)==int(x['selects_qwen'])
        decisive=[r for r in group if abs(float(r[name+'_pca'])-float(r[name+'_qwen']))>1.e-10]
        assert len(decisive)==int(x['decisive'])
        assert sum(abs(float(r[name+'_ray'])-float(r[name+'_oracle']))<1.e-10 for r in decisive)==int(x['chosen_correct'])

    raw,summary=table('ray_audit_rows'),table('ray_audit')
    for x in summary:
        group=[r for r in select(raw,x['panel'],x['target']) if r['scope']==x['scope'] and
               r['family']==x['family'] and r['scored_selection']=='True']
        assert len(group)==int(x['frames'])
        total=sum(int(r['rays']) for r in group)
        assert total==int(x['rays']) and int(x['both_hits'])==sum(int(r['both_hits']) for r in group)
        for k in ('pca','qwen','chosen'):
            assert abs(sum(float(r[k+'_loss']) for r in group)/total-float(x[k+'_cost']))<1.e-12
    assert peak_error<1.e-10
    return dict(frames=len(rows),metric_rows=len(original),paired_rows=len(pairs),selection_rows=len(selection),
                audit_rows=len(summary),max_metric_discrepancy_pp=peak_error)


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
