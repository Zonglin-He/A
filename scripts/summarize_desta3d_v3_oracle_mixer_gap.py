"""Write the source gap's final readout and explicitly anonymous public aggregates."""
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/'artifacts/desta3d_v3/latent_oracle_v1/oracle_mixer_gap_v1'
P = D.parent/'gap_followups_v1/decision_readback_v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def main():
    rdir = D/'independent_readback_v1'
    r = read(rdir/'REPORT.json')
    raw = read(D/'ROOT_ALL_RAW_READBACK.json')
    root = read(rdir/'ROOT_SUMMARY_CROSSCHECK.json')
    decision = read(P/'DECISION_TABLE.json')
    dc = read(P/'ROOT_DECISION_CROSSCHECK.json')
    resources = read(D/'ROOT_GPU_COMPLETION_RECEIPTS.json')
    assert raw['status'] == root['status'] == dc['status'] == 'passed'
    assert dc['decision_table_sha'] == sha(P/'DECISION_TABLE.json')
    assert dc['report_sha'] == root['report_sha'] == sha(rdir/'REPORT.json')
    arms = ('B1', 'seed20260928', 'seed20260929', 'oracle')
    metrics = ('tIoU', 'sIoU', 'vIoU')
    parents = sorted({x['source'] for x in r['arms']['B1']['rows']})
    parent_rows = []
    for i, p in enumerate(parents, 1):
        entry = dict(parent=f'P{i:02}', queries=sum(x['source'] == p for x in r['arms']['B1']['rows']), arms={})
        for arm in arms:
            rows = [x for x in r['arms'][arm]['rows'] if x['source'] == p]
            entry['arms'][arm] = {m: sum(x['metrics'][m] for x in rows)/len(rows) for m in metrics}
        parent_rows.append(entry)
    def anonymous_comparison(c):
        return {k: v for k, v in c.items() if k != 'parent_delta_pp'}
    pub = dict(status='completed_independently_audited', queries=447, parents=31, predictions=1788,
               data_use='Previously scored source diagnosis, GT oracle and privileged evidence; not held-out or target evaluation.',
               configuration='Frozen PTD4B/B1/union256/two final mixers; one fixed pass-matched analytic Joint correction; zero optimizer.',
               absolute={a: {k: v for k, v in r['arms'][a]['summary'].items() if k != 'parent_rows'} for a in arms},
               comparisons={a: {m: anonymous_comparison(c) for m, c in cs.items()} for a, cs in r['comparisons'].items()},
               oracle_vs_learned={a: {m: anonymous_comparison(c) for m, c in cs.items()} for a, cs in r['oracle_vs_learned'].items()},
               parent_rows=parent_rows, retention=r['retention'], query_tails=r['query_tails'],
               practical_gate=r['gate'], oracle_qualified=r['oracle_qualified'],
               geometry_groups=r['geometry_groups'], decision=decision['decision'],
               decision_table=decision['table'], quadrants=dc['quadrants'],
               validation=dict(raw_episodes=raw['episodes'], backwards=raw['backwards'],
                               norm_dot_max_abs=raw['norm_dot_max_abs'], field_relative_L2_max=raw['field_relative_L2_max'],
                               CE_FP64_max_abs=raw['CE_FP64_max_abs'], scalar_tensor_max_abs=r['metric_scalar_tensor_max_abs'],
                               reused_metric_max_abs=r['reused_metric_max_abs'], summary_checks=root['checks'],
                               summary_max_abs=root['max_abs'], decision_checks=dc['checks'], decision_max_abs=dc['max_abs']),
               resource=dict(stage_GPU_seconds=resources['stage_seconds'], cumulative_GPU_seconds=resources['cumulative_seconds'],
                             cap=None, raw_CPU_seconds=raw['CPU_seconds'], score_CPU_seconds=r['CPU_seconds'],
                             decision_CPU_seconds=decision['CPU_seconds']),
               pending='A/B/C only CPU modules/protocols; no candidate training or rescue measurement, no external/OPD/target run.',
               limits='Descriptive unadjusted parent bootstrap CI; two learned seeds share cases; analytic one-step GT oracle is not an upper bound. Local alignment is not finite native causality. Fresh31/388 remains metadata-only.')
    write(D/'PUBLIC_REPORT.json', pub)
    lines = ['# Oracle–Mixer Gap Audit: complete source diagnosis', '', pub['data_use'], '',
             '**DECISION = '+pub['decision']['DECISION']+'**; recommended branch = '+str(pub['decision']['recommendation'])+'.',
             pub['decision']['reason'], '', '## Native policy results', '',
             '|Arm|tIoU %|sIoU %|vIoU %|delta v vs B1, pp|95% parent CI, pp|',
             '|---|---:|---:|---:|---:|---|']
    for arm in arms:
        s = r['arms'][arm]['summary']['parent_macro']
        c = r['comparisons'].get(arm, {}).get('vIoU')
        lines.append('|'+arm+'|'+'|'.join(f'{100*s[m]:.6f}' for m in metrics)+'|'+
                     (f"{c['mean_delta_pp']:+.6f}|[{c['bootstrap_ci95_pp'][0]:+.6f}, {c['bootstrap_ci95_pp'][1]:+.6f}]|" if c else '—|—|'))
    lines += ['', 'Oracle complete practical qualification: **'+str(pub['oracle_qualified'])+'**. Individual gates: '+json.dumps(r['gate']['oracle'])+'.',
              '', '## Direction and finite outcome', '',
              '|Readout|Seed1|Seed2|', '|---|---:|---:|']
    seeds = arms[1:3]
    for title, getter in [
            ('Cosine to oracle: mean / median', lambda x: f"{x['cosine']['mean']:.6f} / {x['cosine']['median']:.6f}"),
            ('Positive temporal local descent fraction', lambda x: f"{x['screen_features']['learned_descent']['event']:.6f}"),
            ('Positive spatial local descent fraction', lambda x: f"{x['screen_features']['learned_descent']['spatial']:.6f}"),
            ('Fraction at >=99% norm cap', lambda x: f"{x['screen_features']['saturated_fraction']:.6f}"),
            ('B1 v>.5 group delta-v: query mean pp', lambda x: f"{x['B1_good_learned_delta_v']['mean']:+.6f}"),
            ('Other group delta-v: query mean pp', lambda x: f"{x['B1_other_learned_delta_v']['mean']:+.6f}")]:
        lines.append('|'+title+'|'+'|'.join(getter(decision['table'][a]) for a in seeds)+'|')
    lines += ['', 'The stratified query means above are not parent-macro effects. A positive local descent dot is not a finite native gain.',
              '', '## All nine oracle / learned outcome combinations', '', '|Oracle / learned|Seed1 count|Seed2 count|', '|---|---:|---:|']
    for q in dc['quadrants'][seeds[0]]:
        lines.append('|'+q+'|'+'|'.join(str(dc['quadrants'][a][q]) for a in seeds)+'|')
    lines += ['', '## Retention, tails and evidence limits', '',
              'Native-good counts: '+json.dumps(pub['retention'])+'.', '',
              'Query changes exceeding5pp (both losses and gains): '+json.dumps(pub['query_tails'])+'.', '',
              'All31 anonymous parent results and all group summaries are included in PUBLIC_REPORT.json. No case, invalid geometry or neutral outcome was removed.', '',
              'Independent validation: '+json.dumps(pub['validation'])+'.', '', pub['limits'], '',
              '## Execution and next boundary', '', pub['pending'], '',
              f"Measured current-stage GPU receipts total {resources['stage_seconds']:.6f}s; all-history total {resources['cumulative_seconds']:.6f}s, cap=null. All CPU audit times are recorded separately.", '',
              'A/B support would motivate a separately registered source test. An INCONCLUSIVE/C recommendation is not a C1/C2/C3 result. No new scientific experiment starts from this report.']
    with (D/'DECISION_AND_NEXT.md').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    complete = dict(status='completed_independently_audited', time=time.time(), queries=447, parents=31,
                    predictions=1788, actual_backwards=raw['backwards'], optimizer_steps=0,
                    DECISION=pub['decision'], oracle_qualified=pub['oracle_qualified'],
                    stage_GPU_seconds=resources['stage_seconds'], cumulative_GPU_seconds=resources['cumulative_seconds'], cap=None,
                    disk_free_bytes=shutil.disk_usage(ROOT).free,
                    evidence={str(p.relative_to(ROOT)): sha(p) for p in [D/'COMPLETE.json', D/'PREDICTIONS_SEAL.json',
                              D/'ROOT_ALL_RAW_READBACK.json', rdir/'REPORT.json', rdir/'ROOT_SUMMARY_CROSSCHECK.json',
                              P/'DECISION_TABLE.json', P/'ROOT_DECISION_CROSSCHECK.json', D/'PUBLIC_REPORT.json', D/'DECISION_AND_NEXT.md']},
                    scientific_scope='Exposed source diagnosis only. No A/B/C GPU or additional data use.',
                    raw_limit=raw['limits'])
    write(D/'ROOT_COMPLETION_SUMMARY.json', complete)
    print(json.dumps({k: v for k, v in complete.items() if k != 'evidence'}))


if __name__ == '__main__':
    main()
