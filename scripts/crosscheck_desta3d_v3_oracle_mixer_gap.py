"""Second, independent CPU reduction of gap-audit metrics and per-parent CIs."""
import json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'artifacts/desta3d_v3/latent_oracle_v1/oracle_mixer_gap_v1'
E=D
ARMS=('B1','seed20260928','seed20260929','oracle')
METRICS=('vIoU','sIoU','tIoU')
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2)

def summary():
    import numpy as np
    start = time.monotonic()
    dest = E / 'independent_readback_v1'
    out = dest / 'ROOT_SUMMARY_CROSSCHECK.json'
    assert not out.exists()
    report = read(dest / 'REPORT.json')
    assert read(dest / 'COMPLETE.json')['report_sha'] == sha(dest / 'REPORT.json')
    assert read(dest / 'PRE_SCORE_AUDIT.json')['status'] == 'passed'
    rows = {a: report['arms'][a]['rows'] for a in ARMS}
    parents = sorted({r['source'] for r in rows['B1']})
    assert len(parents) == 31
    errors = []

    def close(x, y):
        err = float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        errors.append(err)
        assert err < 1e-8, (x, y, err)

    matrices = {}
    for arm in ARMS:
        assert [(r['key'], r['source']) for r in rows[arm]] == [(r['key'], r['source']) for r in rows['B1']]
        matrices[arm] = np.asarray([[sum(r['metrics'][m] for r in rows[arm] if r['source'] == p) /
                                    sum(r['source'] == p for r in rows[arm]) for m in METRICS] for p in parents])
        for j, m in enumerate(METRICS):
            close(matrices[arm][:, j].mean(), report['arms'][arm]['summary']['parent_macro'][m])
            close(sum(r['metrics'][m] for r in rows[arm])/447, report['arms'][arm]['summary']['query_macro'][m])
        close(sum(not r['format_ok'] for r in rows[arm]), report['arms'][arm]['summary']['format_failures'])
    parent_deltas, retained, gates = {}, {}, {}
    for arm in ARMS[1:]:
        delta = matrices[arm] - matrices['B1']
        parent_deltas[arm] = {p: {m: float(100*delta[i, j]) for j, m in enumerate(METRICS)} for i, p in enumerate(parents)}
        for j, m in enumerate(METRICS):
            d = delta[:, j]; c = report['comparisons'][arm][m]
            draws = np.random.default_rng(20260927).integers(0, 31, (10000, 31))
            ci = np.quantile(d[draws].mean(axis=1), [.025, .975])*100
            close(d.mean()*100, c['mean_delta_pp']); close(ci, c['bootstrap_ci95_pp'])
            for p, value in zip(parents, d): close(value*100, c['parent_delta_pp'][p])
            close((d > 1e-10).sum(), c['positive_parents']); close((d < -1e-10).sum(), c['negative_parents'])
            close((np.abs(d) <= 1e-10).sum(), c['zero_parents']); close((d < -.05).sum(), c['severe_loss_below_minus5pp'])
            close(d.min()*100, c['worst_delta_pp']); close(d.max()*100, c['best_delta_pp'])
        retained[arm] = {}
        for m in ('vIoU', 'tIoU'):
            good = {r['key'] for r in rows['B1'] if r['metrics'][m] > .5}
            kept = {r['key'] for r in rows[arm] if r['key'] in good and r['metrics'][m] > .5}
            retained[arm][m] = dict(eligible=len(good), retained=len(kept), lost_keys=sorted(good-kept))
            for name in ('eligible', 'retained'): close(retained[arm][m][name], report['retention'][m][arm][name])
        gates[arm] = dict(v_positive=bool(delta[:, 0].mean()>0),
                         v_lower_CI_positive=bool(np.quantile(delta[draws, 0].mean(axis=1), .025)>0),
                         s_nonnegative=bool(delta[:, 1].mean()>=0), t_nonnegative=bool(delta[:, 2].mean()>=0),
                         no_severe_parent_v_harm=bool(not (delta[:, 0]<-.05).any()),
                         native_good_retained=all(r['eligible']==r['retained'] for r in retained[arm].values()))
        assert gates[arm] == report['gate'][arm]
    qualified = all(gates['oracle'].values())
    assert qualified == report['oracle_qualified']
    write(out, dict(status='passed', checks=len(errors), max_abs=max(errors), parent_deltas=parent_deltas,
                    retention=retained, gate=gates, qualified=qualified, report_sha=sha(dest/'REPORT.json'),
                    auditor_sha=sha(Path(__file__)), CPU_seconds=time.monotonic()-start,
                    scope='Independent saved scalar aggregation; no new label read, inference or selection.'))
    print(json.dumps({k:v for k,v in read(out).items() if k not in ('parent_deltas','retention')}))


if __name__=='__main__':summary()
