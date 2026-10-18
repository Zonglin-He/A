"""Independent reaggregation of anonymous sealed metrics; no GT/data readers."""
from pathlib import Path
import hashlib, json
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/stvg_posterior_decision_tree_v1'
P = ROOT/'results/stvg_posterior_decision/2026-10-18/cached64'
checks = 0

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def equal(a, b):
    global checks
    checks += 1
    if a is None or b is None: assert a is b, (a, b)
    else: assert abs(float(a)-float(b)) < 2e-12, (a, b)

def pair(v, saved):
    v = np.asarray(v, dtype=float)
    equal(len(v), saved['parents'])
    if not len(v):
        assert saved['mean'] is None and saved['CI95'] is None
        return
    equal(np.mean(v), saved['mean'])
    for key, value in [('positive', np.sum(v>1e-12)), ('negative', np.sum(v< -1e-12)),
                       ('neutral', np.sum(np.abs(v)<=1e-12))]: equal(value, saved[key])
    assert np.array_equal(v, np.asarray(saved['per_parent']))
    sample = np.random.default_rng(20261018).integers(len(v), size=(10000, len(v)))
    ci = np.quantile(v[sample].mean(1), [.025, .975])
    for a, b in zip(ci, saved['CI95']): equal(a, b)

def metrics(rows, saved, section, geometry=None):
    for arm, table in saved.items():
        for metric, value in table.items():
            vals = [r[section][geometry][arm][metric] if geometry else r[section][arm][metric]
                    for r in rows]
            vals = [v for v in vals if v is not None]
            equal(np.mean(vals) if vals else None, value)

def audit_summary(rows, summary, geometry=None):
    equal(len(rows), summary['queries']); equal(len(rows), summary['parents'])
    metrics(rows, summary['metrics'], 'temporal' if geometry else 'coverage', geometry)
    for name, saved in summary['paired'].items():
        if geometry:
            arm = name.removeprefix('final_MBR_minus_')
            vals = [100*(r['temporal'][geometry]['final_MBR']['v']-
                         r['temporal'][geometry][arm]['v']) for r in rows]
        else:
            control = name[-1]; metric = 'zero_hit' if '_zero_' in name else 'score'
            vals = [r['coverage']['P'][metric]-r['coverage'][control][metric] for r in rows]
        pair(vals, saved)
    for key, bins in summary['strata'].items():
        assert set(bins) == {r[key] for r in rows}
        for label, group in bins.items():
            sub = [r for r in rows if r[key] == label]
            equal(len(sub), group['queries']); equal(len(sub), group['parents'])
            metrics(sub, group['metrics'], 'temporal' if geometry else 'coverage', geometry)
            for control in (['Native', 'projection'] if geometry else ['N', 'U']):
                vals = [100*(r['temporal'][geometry]['final_MBR']['v']-
                             r['temporal'][geometry][control]['v']) if geometry else
                        r['coverage']['P']['score']-r['coverage'][control]['score'] for r in sub]
                pair(vals, group['paired_vs_'+control])

def main():
    x=read(P/'RESULTS_INITIAL.json'); rows=x['rows']; decision=read(D/'INITIAL_DECISION.json')
    assert len(rows)==64 and len({(r['dataset'],r['alias'])for r in rows})==64
    assert x['GT_after_all_predictions'] and x['parent_macro_equals_query_macro']
    old_checks=0; domains={}
    for ds in ['hc2','vidstg']:
        rr=[r for r in rows if r['dataset']==ds]; assert len(rr)==32
        for geometry in ['Source','Before','After']: audit_summary(rr,x['temporal'][ds][geometry],geometry)
        audit_summary(rr,x['coverage'][ds])
        old=read(ROOT/f'results/stvg_interval_decision/2026-10-18/components64/RESULTS_{ds}.json')
        lookup={(r['alias'],r['geometry']):r for r in old['rows']if r['support']=='fused'}
        for r in rr:
            for geometry in ['Source','Before','After']:
                for new,prior in [('Native','Native'),('projection','full_projection'),('per_offset_MBR','native_MBR')]:
                    previous=lookup[(r['alias'],geometry)]['values'][prior]
                    for key in ['v','t','s','R03','R05','added_known_frames','lost_known_frames',
                                'added_unknown_frames','lost_unknown_frames','added_known_IoU','lost_known_IoU']:
                        assert r['temporal'][geometry][new][key] == previous[key]
                        old_checks+=1
            for layout, c in r['coverage'].items():
                assert c['hits'] in range(5)
                equal(c['score'],min(1,c['hits']/2));equal(c['zero_hit'],1-c['hit1'])
                equal(c['score'],(c['hit1']+c['hit2'])/2)
        for kind in ['T0','S0']:
            risks={}
            for name,key,binname in [('short','event_length_bin','Q1'),('fast','motion_bin','Q4'),('small','target_area_bin','Q1')]:
                sub=[r for r in rr if r[key]==binname]; ss=decision[kind][ds]['subgroups'][name]
                equal(len(sub),ss['parents']); vals=[]
                for control in (['Native','projection']if kind=='T0'else['N','U']):
                    v=[100*(r['temporal']['After']['final_MBR']['v']-r['temporal']['After'][control]['v'])if kind=='T0'else r['coverage']['P']['score']-r['coverage'][control]['score']for r in sub]
                    pair(v,ss['comparisons'][control]);vals.append(np.mean(v)if v else None)
                state='UNKNOWN'if len(sub)<8 else'RISK'if any(v<(-1 if kind=='T0'else-.05)for v in vals)else'PASS'
                assert state==ss['status'];risks[name]=state
        p=x['temporal'][ds]['After']['paired'];c=x['coverage'][ds]['paired']
        means=p['final_MBR_minus_Native']['mean']>=.5 and p['final_MBR_minus_projection']['mean']>=-.5
        assert means==decision['T0'][ds]['mean_gate']
        strict=c['P_minus_N']['mean']>=.05 and c['P_minus_U']['mean']>0
        nonstrict=c['P_minus_N']['mean']>=.05 and c['P_minus_U']['mean']>=0
        assert strict==decision['S0'][ds]['mean_gate'] and strict==nonstrict
        zero=all(c['P_zero_minus_'+a]['mean']<=.05 for a in ['N','U'])
        assert zero==decision['S0'][ds]['zero_hit_risk_pass']
        domains[ds]=dict(T0_mean_gate=means,S0_mean_gate=strict,S0_zero_risk_pass=zero,
                         strict_vs_nonnegative_uniform_gate_same=True)
    assert not decision['T0_passed'] and not decision['S0_passed']
    assert not decision['S0_extension_required'] and not decision['S1_authorized_to_dispatch']
    lock=read(D/'EVALUATION_LOCK.json')
    assert sha(ROOT/'scripts/score_stvg_posterior_decision_tree_v1.py')==lock['scorer_sha256']
    assert sha(D/'ANALYSIS_GATE_LOCK.json')==lock['analysis_gate_sha256']
    result=dict(status='pass',count=64,aggregate_CI_count_checks=checks,
                old_three_readout_scalar_bit_exact_checks=old_checks,
                anonymous_results_only=True,GT_coordinates_read=False,new_scoring=False,
                numerical_audit=read(D/'ROOT_NUMERICAL_AUDIT_NO_GT.json')['maximum_errors'],
                scorer_and_preGT_analysis_lock_exact=True,domains=domains,
                both_gates_stopped=True,no_expansion_or_DINO=True,
                results_sha256=sha(P/'RESULTS_INITIAL.json'),script_sha256=sha(__file__))
    target=P/'ROOT_AGGREGATE_AUDIT.json';assert not target.exists()
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
