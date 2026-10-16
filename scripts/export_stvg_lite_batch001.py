"""Export a fixed completed-stage batch while later experiments keep running.

Reads saved sealed scalar results and code only; never loads models, tensors,
videos, annotations, prompts containing real queries, or raw teacher responses.
"""
import ast
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_lite_event_support_v1'
OUT = ROOT / 'results/stvg_lite_event_support/2026-10-16/batch001'
DOC = ROOT / 'docs/STVG_LITE_EVENT_SUPPORT_PROGRESS001.md'
TRIALS = ('L000', 'L001', 'L002', 'SCOPE_FG', 'SCOPE_ES', 'LR_080', 'LR_090', 'LR_110')
METRICS = ('m_vIoU', 'm_tIoU', 'm_sIoU_dense_GT', 'm_sIoU_supervised_GT', 'm_sIoU_unsupervised_GT')
SOURCES = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    SOURCES[str(p.relative_to(ROOT))] = sha(p)
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def public_source_text(raw):
    text = re.sub(r'/home/[^/]+/\.codex/attachments(?:/[^\s\'\"<>]+)?',
                  'PRIVATE_AUTHORITY_NOT_DISTRIBUTED', raw.decode()).replace(str(ROOT), '.')
    # Documentary authorization literals are private conversation, not algorithms.
    text = re.sub(r'\bauthorization\s*=\s*([\'\"])[^\'\"\n]*\1',
                  'authorization="PRIVATE_AUTHORIZATION_TEXT_REDACTED"', text)
    ast.parse(text)
    return text


def require_receipt(p, status='complete', count=128, gt=False):
    d = read(p)
    assert d['status'] == status and d['count'] == count and d['GT_read'] is gt, p
    return d


def load_block(p, temporal=False):
    cpu = require_receipt(p/'CPU_COMPLETION.json', gt=True)
    root = require_receipt(p/'ROOT_READBACK.json', status='pass')
    predname = 'PREDICTION_GLOBAL_BARRIER.json' if temporal else 'GLOBAL_PREDICTION_BARRIER.json'
    pred = require_receipt(p/predname)
    score = read(p/'POSTSEAL_SCORE_BARRIER.json')
    assert score['status'] == 'complete' and score['case_count'] == 128
    assert cpu['score_barrier_sha256'] == sha(p/'POSTSEAL_SCORE_BARRIER.json')
    for k in ('rows', 'summary', 'paired'):
        assert score[k+'_sha256'] == sha(p/('POSTSEAL_'+k.upper()+'.json'))
    assert score['gt_exposure_sha256'] == sha(p/'POSTSEAL_GT_EXPOSURE.json')
    if temporal:
        require_receipt(p/'CPU_OUTPUT_BARRIER.json')
        bindings = score['prediction_bindings']
        assert bindings['prediction_barrier_sha256'] == sha(p/predname)
        assert bindings['root_readback_sha256'] == sha(p/'ROOT_READBACK.json')
        assert bindings['cpu_output_barrier_sha256'] == sha(p/'CPU_OUTPUT_BARRIER.json')
        assert root['cpu_output_barrier_sha256'] == sha(p/'CPU_OUTPUT_BARRIER.json')
        assert pred['root_readback_sha256'] == sha(p/'ROOT_READBACK.json')
        assert pred['runtime_lock_sha256'] == sha(p/'TEMPORAL_LITE_RUNTIME_LOCK.json')
    else:
        require_receipt(p/'STUDENT_GLOBAL_BARRIER.json')
        assert root['barrier_sha256'] == sha(p/'STUDENT_GLOBAL_BARRIER.json')
        assert pred['student_barrier_sha256'] == sha(p/'STUDENT_GLOBAL_BARRIER.json')
        assert pred['root_readback_sha256'] == sha(p/'ROOT_READBACK.json')
        assert root['runtime_sha256'] == sha(p/'RUNTIME.json')
        bindings = score['bindings']
        for k, f in [('runtime', 'RUNTIME.json'), ('root_readback', 'ROOT_READBACK.json'),
                     ('prediction_barrier', predname), ('student_barrier', 'STUDENT_GLOBAL_BARRIER.json')]:
            assert bindings[k+'_sha256'] == sha(p/f)
    rowsdoc = read(p/'POSTSEAL_ROWS.json')
    assert rowsdoc['bindings'] == bindings
    rows = rowsdoc['rows']
    assert len(rows) == cpu['row_count'] == (768 if temporal else 512)
    assert len({(r['case_id'], r['arm']) for r in rows}) == len(rows)
    summary = read(p/'POSTSEAL_SUMMARY.json')['summary']
    for s in summary:
        vals = [r['metrics'][s['metric']] for r in rows
                if r['dataset'] == s['dataset'] and r['arm'] == s['arm'] and s['metric'] in METRICS]
        if s['metric'] not in METRICS:
            continue
        assert len(vals) == 64
        known = [v for v in vals if v is not None]
        assert len(known) == s['n_valid'] and 64-len(known) == s['n_unknown']
        assert (s['mean'] is None and not known) or abs(math.fsum(known)/len(known)-s['mean']) < 1e-12
    return rows, root


def sanitized_rows(rows, aliases):
    out = []
    for r in rows:
        metrics = {}
        for k in METRICS:
            v = r['metrics'][k]
            assert v is None or isinstance(v, (int, float)) and math.isfinite(v)
            metrics[k] = v
        support = {k: v for k, v in r.get('support', {}).items()
                   if isinstance(v, (int, float, bool)) and re.fullmatch(r'[A-Za-z_]+', k)}
        out.append(dict(case_id=aliases[r['case_id']], dataset=r['dataset'], arm=r['arm'],
                        status=r['status'], metrics=metrics, support_counts=support))
    return out


def summaries(rows):
    out = []
    for ds in ('hc2', 'vidstg'):
        for arm in sorted({r['arm'] for r in rows}):
            sub = [r for r in rows if r['dataset'] == ds and r['arm'] == arm]
            assert len(sub) == 64
            for metric in METRICS:
                vals = [r['metrics'][metric] for r in sub if r['metrics'][metric] is not None]
                out.append(dict(dataset=ds, arm=arm, metric=metric, n_valid=len(vals), n_unknown=64-len(vals),
                                unknown_case_ids=[r['case_id'] for r in sub if r['metrics'][metric] is None],
                                mean=math.fsum(vals)/len(vals) if vals else None,
                                median=statistics.median(vals) if vals else None,
                                min=min(vals) if vals else None, max=max(vals) if vals else None))
            vals = [r['metrics']['m_vIoU'] for r in sub if r['metrics']['m_vIoU'] is not None]
            for th in (.3, .5):
                out.append(dict(dataset=ds, arm=arm, metric=f'R@vIoU>={th:.1f}',
                                n_valid=len(vals), n_unknown=64-len(vals), mean=sum(v >= th for v in vals)/len(vals)))
    return out


def contrast(a, aa, b, bb, label):
    out = []
    for ds in ('hc2', 'vidstg'):
        x = {r['case_id']: r for r in a if r['dataset'] == ds and r['arm'] == aa}
        y = {r['case_id']: r for r in b if r['dataset'] == ds and r['arm'] == bb}
        assert x.keys() == y.keys() and len(x) == 64
        for m in METRICS:
            delta = {cid: (x[cid]['metrics'][m]-y[cid]['metrics'][m])*100
                     for cid in sorted(x) if x[cid]['metrics'][m] is not None and y[cid]['metrics'][m] is not None}
            out.append(dict(dataset=ds, comparison=label, metric=m, n_pairs=len(delta),
                            mean_delta_pp=math.fsum(delta.values())/len(delta) if delta else None,
                            positive_case_ids=[c for c,v in delta.items() if v > 1e-10],
                            negative_case_ids=[c for c,v in delta.items() if v < -1e-10],
                            neutral_case_ids=[c for c,v in delta.items() if abs(v) <= 1e-10],
                            unknown_case_ids=sorted(set(x)-set(delta)), per_source_delta_pp=delta,
                            neutral_tolerance_pp=1e-10))
    return out


def main():
    assert not (OUT/'BATCH_RECEIPT.json').exists(), 'immutable batch already exported'
    data = {}; roots = {}
    for tid in TRIALS + ('TEMPORAL',):
        p = BASE/'temporal/revision003' if tid == 'TEMPORAL' else BASE/'trials'/tid
        data[tid], roots[tid] = load_block(p, tid == 'TEMPORAL')
    roster = data['L000']
    aliases = {}
    for ds, prefix in [('hc2', 'HC'), ('vidstg', 'VID')]:
        ids = sorted({r['case_id'] for r in roster if r['dataset'] == ds})
        assert len(ids) == 64
        aliases.update({c: f'{prefix}{i:03}' for i,c in enumerate(ids, 1)})
    public = {}; costs = {}
    for tid, rows in data.items():
        assert {r['case_id'] for r in rows} == set(aliases)
        rows = sanitized_rows(rows, aliases); public[tid] = rows
        write(OUT/'trials'/tid/'ROWS.json', dict(status='complete', count=128, row_count=len(rows),
              metrics_unit='fraction', all_sources_development=True, rows=rows))
        write(OUT/'trials'/tid/'SUMMARY.json', dict(status='complete', summary=summaries(rows)))
        controls = [('TS_temporal_lite', 'TS_offset_trust')] if tid == 'TEMPORAL' else [('TS_M2','F'),('TS_M2','T'),('S_M2','F')]
        paired = sum((contrast(rows,a,rows,b,a+' minus '+b) for a,b in controls), [])
        write(OUT/'trials'/tid/'PAIRED.json', dict(status='complete', paired=paired, bootstrap_executed=False))
        if tid != 'TEMPORAL':
            d = read(BASE/'trials'/tid/'DESIGN_LOCK.json')
            write(OUT/'trials'/tid/'CONFIGURATION.json', {k:d[k] for k in
                  ['status','trial_id','recipe','response_scope','lr_multiplier','configs','response_settings']})
            cost = read(BASE/'trials'/tid/'MATCHED_EFFICIENCY_READBACK.json')
            assert cost['status'] == 'complete' and cost['count'] == 128
            cr = []
            for r in cost['records']:
                fc = r['fit_cost']
                cr.append(dict(case_id=aliases[r['case_id']], no_fit=fc is None,
                     decoder_calls=sum(fc['decode_counts'].values()) if fc else 0,
                     fit_cuda_synchronized_seconds=fc['fit_wall_cuda_synchronized_seconds'] if fc else None))
            costs[tid] = dict(records=cr, decoder_calls=sum(r['decoder_calls'] for r in cr),
                fitted_query_count=sum(not r['no_fit'] for r in cr),
                median_fit_seconds=statistics.median(r['fit_cuda_synchronized_seconds'] for r in cr if not r['no_fit']))
        root = roots[tid]
        write(OUT/'trials'/tid/'EXECUTION_READBACK.json', dict(status='pass', count=128,
              counters=root.get('counters',{}), GT_read_during_fit_or_readback=False,
              GT_read_only_after_prediction_seal=True, saved_mathematics_independently_read=True,
              full_backward_independently_recomputed=False,
              temporal_conditional_readout=tid=='TEMPORAL',
              baseline_exact_replay_count=root.get('baseline_exact_replay_count'),
              temporal_lite_independent_replay_count=root.get('temporal_lite_independent_replay_count'),
              explicit_original_no_fit_fallback_count=root.get('explicit_original_no_fit_fallback_count')))
    write(OUT/'MATCHED_COSTS.json', dict(status='complete', trials=costs,
          scope='cached spatial fit/decoder costs only; excludes teacher, video preparation and end-to-end deployment'))
    pairs = []
    for tid in ('L001','L002'):
        pairs += contrast(public[tid],'TS_M2',public['L000'],'TS_M2',tid+' minus L000')
    for tid in ('SCOPE_ES','LR_080','LR_090','LR_110'):
        pairs += contrast(public[tid],'TS_M2',public['SCOPE_FG'],'TS_M2',tid+' minus SCOPE_FG')
    write(OUT/'CROSS_TRIAL_PAIRED.json',dict(status='complete',paired=pairs,bootstrap_executed=False))
    lock = read(BASE/'STRUCTURAL_SELECTION_LOCK.json')
    write(OUT/'STRUCTURE_SELECTION.json',{k:lock[k] for k in ['status','count','selected_spatial_trial',
          'selected_spatial_recipe','selected_temporal','structural_loss_budget_pp','GT_used_for_online_reference_or_step',
          'CURRENT_promoted','bootstrap_executed']})
    protocol = (ROOT/'protocols/stvg_lite_event_support_v1.md').read_text()
    protocol = protocol.replace(str(ROOT)+'/', './')
    protocol = re.sub(r'^授权：.*$', '研究范围：有限联合精简、事件支持与局部学习率开发研究；历史队列保持停止，CURRENT不变。', protocol, flags=re.M)
    assert '.codex/attachments/' not in protocol
    (OUT/'PROTOCOL.md').write_text(protocol)
    runtime = read(BASE/'RECOVERY_RUNTIME004.json')
    pins = {k:v for k,v in runtime['pins'].items() if k.endswith('.py')}
    manifest = []
    for rel, expected in sorted(pins.items()):
        raw = (ROOT/rel).read_bytes(); assert hashlib.sha256(raw).hexdigest() == expected
        text = public_source_text(raw)
        ast.parse(text)
        target = OUT/'source_snapshot'/rel; target.parent.mkdir(parents=True,exist_ok=True); target.write_text(text)
        manifest.append(dict(path=rel,original_sha256=expected,public_sha256=sha(target),private_path_normalized=raw != text.encode()))
    write(OUT/'SOURCE_BINDINGS.json',dict(status='pass',source_snapshot=manifest,measurement_sources=SOURCES,
          private_runtime_data_not_distributed=True))
    write(OUT/'ENGINEERING_RECOVERY.json',dict(originals_preserved=True,science_thresholds_unchanged=True,history=[
          dict(revision='002',issue='flat/no_information mean_gradient=None rejected by CPU reader',resolution='strict flat branch, unchanged 8e-5 tolerance; no GPU rerun'),
          dict(revision='003',issue='CPU context absent and valid per-library thread dictionary rejected',resolution='bind original lineage 2/20 BLAS contexts; 111 exact baseline and 111 independent Lite replays, 17 original fallbacks'),
          dict(revision='004',issue='selector expected count instead of Temporal case_count/row_count',resolution='strict 128-case/768-row adapter; no fit or GT scoring repeated')]))
    means = {}
    for tid,rows in public.items():
        arm = 'TS_temporal_lite' if tid=='TEMPORAL' else 'TS_M2'
        means[tid] = {ds:100*math.fsum(r['metrics']['m_vIoU'] for r in rows if r['dataset']==ds and r['arm']==arm)/64 for ds in ('hc2','vidstg')}
    plot(means, costs)
    lines = ['# Lite STVG development measurements — incremental batch 001', '',
      'This immutable batch publishes completed stages while the study continues. It is not the final recipe, independent confirmation, or Table 1.', '',
      '## Cohort and protocol', '',
      'TA-STVG cross-domain Source checkpoints; 128 exposed development sources, HC-STVG-v2 64 and VidSTG 64, one query per parent. Frozen R1, existing weight-free eight-candidate references and original Source weights are unchanged. Query/LN/Adam reset each query; writeback configuration fields do not activate online LN carry.', '',
      'All prediction stages were sealed and independently read before GT scoring. GT is used for offline development selection and final metrics, never for references, online acceptance, or candidate generation. No bootstrap, CURRENT promotion or official full-set execution.', '',
      '## Completed measurements', '', '| Block | HC m_vIoU (%) | VidSTG m_vIoU (%) |', '|---|---:|---:|']
    lines += [f'| {tid} | {v["hc2"]:.9f} | {v["vidstg"]:.9f} |' for tid,v in means.items()]
    lines += ['', 'L000/L001/L002 compare Full, Lite-backtrack and Lite-fullstep with the old Temporal clock. TEMPORAL is a conditional CPU readout using fixed original A000 spatial geometry, not a freshly re-clocked end-to-end experiment. SCOPE_FG/SCOPE_ES use selected Lite Temporal and fixed reference geometry; they isolate decoder-response support. LR blocks differ only in the locked spatial learning-rate multiplier.', '',
      'Event support changes HC by +0.006504 pp and VidSTG by -0.040380 pp relative to full-grid. The prelocked equal-domain non-degradation rule therefore retains full-grid. The 0.1 pp per-domain structural budget selects Lite-fullstep and Temporal-Lite; this is a development choice, not evidence of both-domain improvement.', '',
      '## Costs and complete results', '',
      f'Full cached spatial decoder calls: {costs["L000"]["decoder_calls"]}; Lite-fullstep: {costs["L002"]["decoder_calls"]}. These measured costs exclude teacher/video preparation and are not end-to-end deployment speedups.', '',
      'The linked result directory contains every anonymous per-source scalar, paired positive/negative/neutral/unknown case, actual trial configuration, matched fit costs, source hashes, complete pinned Python source snapshot, and engineering recovery history. Scores are fractions in JSON; contrasts are percentage points. Repeated baselines are logical readouts, not independent training runs.', '',
      '## Pending at this batch boundary', '',
      'LR_120 and final recipe selection, actual re-clocked development pipelines, new 64-source confirmation, and the later 8/16/32-frame visual-evidence P0 are outside this batch. The pending status is fixed at export time, even if workers finish subsequently. Subsequent batches will append their results; no negative result is omitted.', '',
      'No videos, annotations, model weights, raw tensors, real query/caption prompts, raw teacher responses or personal conversations are published. The frozen expert may have target-domain prior; no claim of overlap-free external training is made.', '',
      '[All anonymous results and source snapshot](../results/stvg_lite_event_support/2026-10-16/batch001/)', '',
      '![Completed-stage means](../results/stvg_lite_event_support/2026-10-16/batch001/COMPLETED_STAGE_MEANS.png)', '']
    DOC.write_text('\n'.join(lines))
    receipt = dict(status='complete',scope='incremental completed-stage publication only',batch='001',
          source_count=128,per_domain=64,spatial_blocks=list(TRIALS),temporal_blocks=['TEMPORAL'],
          logical_score_rows=sum(len(x) for x in public.values()),source_snapshot_count=len(manifest),
          all_positive_negative_neutral_unknown_preserved=True,new_inference=False,new_GT_read=False,
          whole_study_FINAL_complete=False,independent_confirmation_complete=False,
          bootstrap_executed=False,CURRENT_promoted=False)
    write(OUT/'BATCH_RECEIPT.json',receipt)
    print(json.dumps(receipt))


def plot(means, costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    labels = list(means)
    for ax,ds,title in zip(axs,('hc2','vidstg'),('HC-STVG-v2 development','VidSTG development')):
        ax.scatter(range(len(labels)),[means[k][ds] for k in labels],s=42)
        ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set_ylabel('m_vIoU (%)')
        ax.set_title(title);ax.grid(alpha=.25)
    fig.suptitle('Completed conditional stages; not final re-clocked confirmation',fontsize=12)
    fig.savefig(OUT/'COMPLETED_STAGE_MEANS.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
