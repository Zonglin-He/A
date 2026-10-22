"""Read already-published metric scalars; no labels, models or new predictions.

Usage: python recompute_saved_scalars.py FOURARM_ROWS FOURARM_SUMMARY RAW_RESULTS T2_RESULTS
Inputs are the exact historical public JSON files referenced in PROVENANCE.json.
This checks aggregation, not the original GPU execution or metric correctness.
"""
import json, math, statistics, sys
from collections import defaultdict

def mean(xs):
    return statistics.fmean(xs) if xs else None

def audit(four_rows, four_summary, raw, t2):
    checks = []
    def same(a, b, name):
        assert (a is None and b is None) or (a is not None and b is not None and abs(a-b) < 2e-12), name
        checks.append(abs(a-b) if a is not None else 0.)
    assert len(four_rows) == 256 and len({r['parent'] for r in four_rows}) == 16
    groups = defaultdict(list)
    for r in four_rows:
        if r['condition'] != 'clean': groups[r['parent']].append(r)
    assert all(len(v) == 15 for v in groups.values())
    four = {}
    for arm in ('Frozen','Rerank','Hard','OPD'):
        four[arm] = {}
        for field in ('sIoU','tIoU','vIoU_corrected'):
            value = mean([mean([r['arms'][arm][field] for r in rr]) for rr in groups.values()])
            same(value, four_summary['transient']['arms'][arm][field]['mean'], 'fourarm/'+arm+'/'+field)
            four[arm][field] = 100*value
    loss = {arm:sum(r['diagnostics'][arm]['final_loss'] < r['diagnostics'][arm]['initial_loss'] for r in four_rows) for arm in ('Hard','OPD')}
    assert loss == {'Hard':256,'OPD':256}
    assert len(raw['rows']) == 128 and raw['parents'] == 128
    raw_means = {}
    for arm, values in raw['means'].items():
        raw_means[arm] = {}
        for field, target in values.items():
            if field not in ('v','t','s','boundary_MAE'): continue
            value=mean([r['values'][arm][field] for r in raw['rows']])
            same(value,target,'raw/'+arm+'/'+field); raw_means[arm][field]=value
    for r in raw['rows']:
        v=r['values']
        for field in ('v','t','s','boundary_MAE'):
            for arm in ('Fixed','Live'):
                same(v['Mean_'+arm][field],mean([v[f'Live{s}_{arm}{s}'][field] for s in raw['seeds']]),'raw matched seed mean')
    dv=[100*(r['values']['Mean_Live']['v']-r['values']['Mean_Native']['v']) for r in raw['rows']]
    tails=dict(positive=sum(x>1e-10 for x in dv),negative=sum(x< -1e-10 for x in dv),unchanged=sum(abs(x)<=1e-10 for x in dv),below_minus5pp=sum(x< -5 for x in dv))
    assert tails == dict(positive=41,negative=74,unchanged=13,below_minus5pp=34)
    assert len(t2['rows']) == 768 and t2['count'] == 256
    by=defaultdict(list)
    for r in t2['rows']: by[(r['dataset'],r['arm'])].append(r)
    t2_means={}
    for cell in t2['summary']:
        xs=[r['metrics'].get(cell['metric']) for r in by[(cell['dataset'],cell['arm'])]]
        valid=[x for x in xs if x is not None and math.isfinite(x)]
        assert len(xs)==128 and len(valid)==cell['valid'] and 128-len(valid)==cell['unknown']
        value=mean(valid);same(value,cell['mean'],'t2/'+str(cell))
        if cell['metric'] in ('m_vIoU','m_tIoU'):
            t2_means.setdefault(cell['dataset'],{}).setdefault(cell['arm'],{})[cell['metric']]=100*value
    return dict(status='passed_saved_scalar_aggregation_only',fourarm=dict(rows=256,independent_parents=16,transient_cells=240,means_percent=four,loss_decrease_counts=loss),raw=dict(rows=128,means=raw_means,metric_units='v/t/s are fractions; boundary_MAE preserves historical frame units, not seconds',Live_minus_Native_tails=tails),T2_T4=dict(rows=768,independent_parents=256,means_percent=t2_means),scalar_identity_checks=len(checks),max_abs_error=max(checks),new_GT_or_predictions_or_fits_or_models=0)

if __name__=='__main__':
    assert len(sys.argv)==5, __doc__
    print(json.dumps(audit(*[json.load(open(p)) for p in sys.argv[1:]]),indent=2))
