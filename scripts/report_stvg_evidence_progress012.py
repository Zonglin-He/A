"""Export completed controlled rounds with exact anonymous scalar readback."""
import sys, math, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_evidence_common_v1 import BASE, read, write, sha
OUT = ROOT / 'results/stvg_evidence_constrained/2026-10-15/round2'

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    phases = ['round2', 'round2_prompt009', 'preservation', 'preservation010']
    audits = []
    for phase in phases:
        b = BASE / phase
        rows = [dict(case_id=x['case_id'], dataset=x['dataset'], arm=x['arm'], metrics=x['metrics'])
                for x in read(b / 'POSTSEAL_ROWS.json')['rows']]
        summary = read(b / 'POSTSEAL_SUMMARY.json')
        write(OUT / (phase + '_ALL_ROWS.json'), dict(status='complete', rows=rows, development=True))
        write(OUT / (phase + '_SUMMARY.json'), summary)
        write(OUT / (phase + '_PAIRS.json'), read(b / 'PARENT_PAIRED_DELTAS.json'))
        for r in summary['summary']:
            values = [x['metrics'][r['metric']] for x in rows
                      if x['dataset'] == r['dataset'] and x['arm'] == r['arm']
                      and x['metrics'][r['metric']] is not None]
            assert len(values) == r['count']
            assert r['mean'] is None if not values else abs(math.fsum(values) / len(values) - r['mean']) < 1e-12
        roots = {}
        for name in ['ROOT_INPUT_READBACK.json', 'ROOT_TEMPORAL_READBACK.json', 'ROOT_SPATIAL_READBACK.json']:
            r = read(b / name)
            roots[name] = {k: v for k, v in r.items() if isinstance(v, (bool, int, float)) or k in ['status', 'scope']}
            roots[name]['private_receipt_sha256'] = sha(b / name)
        write(OUT / (phase + '_ROOTS.json'), dict(status='pass', receipts=roots))
        audits.append(dict(phase=phase, row_count=len(rows), summary_scalar_count=len(summary['summary']),
                           positive_negative_neutral_null_preserved=True))
    for name in ['FROZEN_METHOD_LOCK.json', 'TEMPORAL_SELECTION_LOCK.json', 'SPATIAL_SELECTION_LOCK009.json',
                 'PRESERVATION_SELECTION_LOCK010.json']:
        write(OUT / name, read(BASE / name))
    m = {(x['dataset'], x['arm']): 100*x['mean']
         for x in read(BASE / 'round2_prompt009/POSTSEAL_SUMMARY.json')['summary'] if x['metric'] == 'm_vIoU'}
    old = {(x['dataset'], x['arm']): 100*x['mean']
           for x in read(BASE / 'round2/POSTSEAL_SUMMARY.json')['summary'] if x['metric'] == 'm_vIoU'}
    pp = {(x['dataset'], x['arm']): 100*x['mean']
          for x in read(BASE / 'preservation010/POSTSEAL_SUMMARY.json')['summary'] if x['metric'] == 'm_vIoU'}
    lines = ['# Evidence-Constrained STVG: final controlled development rounds', '',
             'Status: algorithm frozen; fresh32 confirmation and official13785-query Table1 are not yet complete. CURRENT is unchanged.', '',
             '## Matched event clock × visual presentation', '',
             'All32 historically exposed development parents (16/domain), same offset-trust temporal objective, fixed domain-specific Spatial configurations. Every phase sealed all predictions before scoring. No GT generates references or accepts parameter steps; development GT selects one global recipe. No Bootstrap.', '',
             '|Spatial input|HC T+S vIoU (%)|Vid T+S vIoU (%)|', '|---|---:|---:|']
    for v in ['old_crop', 'new_crop', 'old_scene', 'new_scene']:
        lines.append(f"|{v}|{m['hc2','TS_'+v]:.6f}|{m['vidstg','TS_'+v]:.6f}|")
    lines += ['', 'The original factorial changed crop prompt JSON whitespace together with the clock. It is retained in full as an implementation-confounded measurement, not a clean clock contrast. A separate correction changed only128 new_crop prompts to the exact original serialization: pixels, geometry, clock, model and parameters stayed fixed; other384 logical teacher records reused exact original responses. Seven changed-reference cases received actual new Spatial fits and full Student readouts; the remaining outputs were verified exact reference/config/seed reuse. The corrected global equal-domain mean selects new_crop. The improvement is small and is not an independent generalization estimate.', '',
              'Original-matrix new_crop HC/Vid was %.6f/%.6f%%; corrected is %.6f/%.6f%%. Whitespace was not assumed semantically harmless.' % (old['hc2','TS_new_crop'],old['vidstg','TS_new_crop'],m['hc2','TS_new_crop'],m['vidstg','TS_new_crop']), '',
              'Original512 logical WHO jobs comprised128 exact historical response reuses and384 new calls;58 actual new Spatial fits/full redecodes. The matched correction added128 calls and7 actual fits. These are not512 new Student fits. Proposal reconstruction is not universally deterministic: exact matching cached centers are reused, while new centers use the unchanged locked algorithm.', '',
              '## One prelocked output-preservation ablation', '',
              '|Fixed reference|HC T+S (%)|Vid T+S (%)|', '|---|---:|---:|',
              f"|new_crop original update|{pp['hc2','TS_new_crop']:.6f}|{pp['vidstg','TS_new_crop']:.6f}|",
              f"|new_crop + fixed lambda1 preservation|{pp['hc2','TS_preserved']:.6f}|{pp['vidstg','TS_preserved']:.6f}|", '',
              'Exactly4 corrected-winner cases met the prelocked reference-improves/unsupervised-frame-harms trigger. The single fixed lambda1 penalty received23 actual fits/full redecodes and no new teacher search. Its equal-domain mean is lower; it is not selected. The earlier old_crop preservation run also remains exported with its negative result, but cannot select the corrected method. No lambda sweep follows.', '',
              '## Frozen method and next execution', '',
              'One global recipe: offset-aware full-grid-trust temporal adaptation; spatial observation centers conditioned on its adapted interval; original crop-contact Object-ID prompt; unchanged Selective Search256 / GrabCut24 / DIS candidate algorithm and up-to8 new candidates plus Source; existing domain-specific1792-parameter validated IoU-OPD. Output preservation is off. No new detector, SAM, DINO or method search.', '',
              'Temporal objective decline is not teacher correctness. All28 usable development trust fits made nonzero512-parameter updates and hit the50-iteration cap; no convergence claim. Four explicit missing temporal inputs fell back. Wrong teacher preferences and remaining full-Tube negative cases are preserved.', '',
              'The fresh32 roster excludes enumerated prior64 coordinate parents, original8 and additional recorded teacher/development source registries. This is a registered-roster exclusion, not a globally unseen-data claim. Its method lock precedes new teacher/adaptation/GT; results cannot choose another recipe. The official episodic Table1 is HC validation3482 + Vid test10303 queries. It will seal Frozen/Temporal/Spatial/T+S before final GT scoring; old canceled baselines are not called complete or restarted.', '',
              'Input root independently checked source RGB hashes, serialized tensor bytes, masks, geometry and marked-scene digests. Full-scene images were not independently rerendered. Eight private panels from two fixed cases were actually viewed; partial/overlapping candidates and edge-positioned ID labels remain limitations. Historical E mask NPZ files were absent from its old candidate-barrier SHA list; exact copied current mask bytes were checked, without claiming a nonexistent historical seal.', '',
              'All positive, negative, neutral and null scalar rows are public. Private video/query/mask/annotation/weights/hidden-state payloads are excluded. Frozen external R1 training-data overlap remains unresolved. No paper-wide or Table1 completion is claimed.']
    report = '\n'.join(lines) + '\n'
    (ROOT / 'docs/STVG_EVIDENCE_CONSTRAINED_PROGRESS.md').write_text(report)
    (OUT / 'REPORT.md').write_text(report)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for a, ds, title in zip(ax, ['hc2','vidstg'], ['HC-STVG-v2','VidSTG']):
        names = ['old_crop','new_crop','old_scene','new_scene']
        vals = [m[ds,'TS_'+v] for v in names]
        a.bar(names, vals, color=['#8d9db0','#3e6d9d','#b2b8c2','#77a898'])
        a.set(title=title+' / exposed development16', ylabel='T+S vIoU (%)')
        a.tick_params(axis='x', rotation=20)
        for i, v in enumerate(vals): a.text(i, v+.03, f'{v:.3f}', ha='center', fontsize=9)
    fig.savefig(OUT / 'MATCHED_ROUND2.png', dpi=160)
    plt.close(fig)
    write(BASE / 'ROUND2_ANONYMOUS_SCALAR_AUDIT012.json', dict(status='pass', phases=audits,
          CPU_only=True, all_positive_negative_neutral_null_preserved=True))
    print('completed controlled phases exported', sum(x['row_count'] for x in audits))

if __name__ == '__main__': main()
