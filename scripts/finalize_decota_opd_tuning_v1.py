"""Independent post-selection readback. No model, media or new GT scoring.

Wait mode is bounded and follows only the actual finite tuning seal. Baseline
GPU continuation may run concurrently; this process uses saved anonymous rows.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import collections
import csv
import gzip
import json
import time
import traceback
import numpy as np
from scripts.decota_opd_tuning_common_v1 import BASE, PUB, read, write, status, sha, verify, archive, DEFAULT


def independent(rows, field):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row['source_id']].append(float(row[field]))
    source = np.array([sum(v) / len(v) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(20261006)
    bootstrap = np.array([source[rng.integers(0, len(source), size=len(source))].mean()
                          for _ in range(10000)])
    result = dict(mean=float(source.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                  query_macro=float(np.mean([row[field] for row in rows])),
                  order_values=[float(np.mean([r[field] for r in rows if r['order'] == order]))
                                for order in sorted({r['order'] for r in rows})])
    if field.startswith('delta_'):
        result.update(gross_gain_pp=float(np.clip(source, 0, None).mean() * 100),
                      gross_loss_pp=float(-np.clip(source, None, 0).mean() * 100),
                      harm_gt5pp_sources=int((source < -.05).sum()),
                      harm_gt20pp_sources=int((source < -.20).sum()),
                      harm_gt5pp_cells=sum(r[field] < -.05 for r in rows),
                      harm_gt20pp_cells=sum(r[field] < -.20 for r in rows))
    return result


def rank(summary):
    metric = summary['statistics']['metrics']['delta_total_v']
    return metric['mean'], -metric['harm_gt20pp_sources'], -summary['GPU_fit_seconds']


def verify_root_pin():
    expected = read(BASE / 'ROOT_AUDIT_RUNTIME.json')['script_sha256']
    for revision in sorted((BASE / 'revisions').glob('root_report_*.json')):
        expected = read(revision)['new_root_script_sha256']
    assert sha(Path(__file__)) == expected


def run():
    verify()
    verify_root_pin()
    barrier = read(BASE / 'SELECTION_BARRIER.json')
    assert barrier['status'] == 'sealed' and set(barrier['datasets']) == {'vidstg', 'hc2'}
    assert sha(ROOT / 'methods/decota_spatial_opd_v1/configs.json') == barrier['configuration_file_sha256']
    assert read(BASE / 'TUNING_COMPLETION.json')['status'] == 'completed_selection_pending_root_publication'
    definition = read(BASE / 'DESIGN_LOCK.json')
    records = []; tables = {}; verified_fields = 0; raw_payloads = 0; failures = []
    for ds in ['vidstg', 'hc2']:
        summaries = []; screen = {}
        for trial in sorted((BASE / 'trials' / ds).iterdir()):
            if trial.name.startswith('qual_'):
                continue
            cfg = read(trial / 'CONFIG.json')
            if (trial / 'NUMERICAL_INVALID.json').exists():
                failures.append(dict(dataset=ds, trial=trial.name, config=cfg['config'],
                                     status='numerically_invalid_unscored',
                                     preserved_prefix=len(list(trial.glob('order*/*.pt'))),
                                     receipt_sha256=sha(trial / 'NUMERICAL_INVALID.json')))
                continue
            gpu = read(trial / 'PREDICTION_BARRIER.json'); cpu = read(trial / 'CPU_COMPLETION.json')
            assert gpu['status'] == 'sealed' and not gpu['GT_read']
            assert gpu['config_sha256'] == sha(trial / 'CONFIG.json')
            assert gpu['time'] <= read(trial / 'GT_EXPOSURE.json')['time'] <= cpu['time']
            for path, digest in gpu['files'].items():
                assert sha(BASE / path) == digest
            raw_payloads += len(gpu['files'])
            for path, digest in cpu['files'].items():
                assert sha(ROOT / path) == digest
            summary = read(PUB / ds / trial.name / 'SUMMARY.json')
            with gzip.open(PUB / ds / trial.name / 'ROWS.jsonl.gz', 'rt') as f:
                rows = [json.loads(line) for line in f]
            assert len(rows) == gpu['cells'] == cpu['rows']
            assert summary['independent_math_state_checks'] == len(rows)
            assert summary['config'] == cfg['config']
            expected_sources = 16 if cfg['phase'] == 'screen' else 32
            assert len({r['source_id'] for r in rows}) == expected_sources
            for field, original in summary['statistics']['metrics'].items():
                calculated = independent(rows, field)
                for key, value in calculated.items():
                    assert np.allclose(value, original[key], atol=1e-12, rtol=0), (ds, trial.name, field, key)
                verified_fields += 1
            assert abs(sum(r['compute']['fit_GPU_seconds'] for r in rows) - summary['GPU_fit_seconds']) < 1e-9
            total = summary['statistics']['metrics']['delta_total_v']
            current = summary['statistics']['metrics']['delta_current_v']
            inherited = summary['statistics']['metrics']['delta_inherited_v']
            assert abs(total['mean'] - current['mean'] - inherited['mean']) < 1e-12
            records.append(dict(dataset=ds, trial=trial.name, phase=cfg['phase'],
                factor=cfg.get('factor', cfg.get('coordinate')), **cfg['config'],
                sources=expected_sources, arrivals=len(rows), delta_v_pp=100 * total['mean'],
                ci_low_pp=100 * total['ci95'][0], ci_high_pp=100 * total['ci95'][1],
                current_pp=100 * current['mean'], inherited_pp=100 * inherited['mean'],
                harm20_sources=total['harm_gt20pp_sources'], harm20_cells=total['harm_gt20pp_cells'],
                fit_GPU_seconds=summary['GPU_fit_seconds'],
                capture_seconds=sum(r['compute']['capture_seconds'] for r in rows),
                CPU_math_seconds=sum(r['compute']['CPU_math_seconds'] for r in rows)))
            summaries.append(summary)
            if cfg['phase'] == 'screen':
                screen[trial.name] = summary
        assert len(screen) + sum(x['dataset'] == ds and x['trial'].startswith('screen_') for x in failures) == 16
        reference = next(s for s in screen.values() if s['config'] == DEFAULT)
        sensitivity = {}
        for factor in definition['space']:
            values = [rank(reference)[0]]
            for name, s in screen.items():
                if read(BASE / 'trials' / ds / name / 'CONFIG.json')['factor'] == factor:
                    values.append(rank(s)[0])
            sensitivity[factor] = max(values) - min(values)
        saved = read(BASE / 'selections' / f'{ds}_sensitivity.json')
        assert all(abs(value - saved['effect_ranges'][factor]) < 1e-12 for factor, value in sensitivity.items())
        sensitive = sorted(definition['space'], key=lambda x: (-sensitivity[x], x))[:2]
        assert sensitive == saved['selected_coordinates']
        refined = [s for s in summaries if s['phase'] == 'refine']
        selected = read(BASE / 'selections' / f'{ds}.json')
        best = max(refined, key=rank)
        assert best['trial'] == selected['trial'] and best['config'] == selected['config']
        assert sha(PUB / ds / best['trial'] / 'SUMMARY.json') == selected['summary_sha256']
        assert selected['old_confirmation_used'] is False
        tables[ds] = dict(selection=selected, sensitivity_pp={k: v * 100 for k, v in sensitivity.items()},
                          completed_unique_trials=len(summaries), invalid_trials=sum(x['dataset'] == ds for x in failures))
    output = PUB / 'ROOT_AUDIT.json'
    audit = dict(status='independent_saved_rows_selection_and_seal_readback_pass',
                 trial_payload_byte_hashes=raw_payloads, metric_aggregates_checked=verified_fields,
                 tables=tables, numerical_failures=failures, new_GT_reads=False,
                 new_model_execution=False, selection_is_exposed_development=True,
                 heldout_efficacy_claim=False, time=time.time())
    if output.exists():
        original = read(output)
        assert {k: v for k, v in original.items() if k != 'time'} == {k: v for k, v in audit.items() if k != 'time'}
    else:
        write(output, audit)
    with (PUB / 'TRIALS.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    if (PUB / 'NUMERICAL_FAILURES.json').exists():
        assert read(PUB / 'NUMERICAL_FAILURES.json') == failures
    else:
        write(PUB / 'NUMERICAL_FAILURES.json', failures)
    draw(records, tables)
    lines = ['# DeCoTA Spatial OPD：有限参数搜索', '',
             'VidSTG 与 HC2 各选择一套统一参数。下表来自历史曝光开发来源，',
             '用于选参；CI 描述这些开发样本，不是经过调参后的独立效能证据。', '',
             '| Target | lr | sigma | tau | steps | LN writeback | ΔvIoU (pp) | current / inherited (pp) | >20pp harm sources / cells |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    notes = []
    for ds, table in tables.items():
        chosen = table['selection']; cfg = chosen['config']; met = chosen['statistics']['metrics']
        m = met['delta_total_v']
        lines.append(f"| {ds} | {cfg['lr']} | {cfg['sigma']} | {cfg['tau']} | {cfg['steps']} | {cfg['writeback']} | {m['mean']*100:.3f} [{m['ci95'][0]*100:.3f}, {m['ci95'][1]*100:.3f}] | {met['delta_current_v']['mean']*100:.3f} / {met['delta_inherited_v']['mean']*100:.3f} | {m['harm_gt20pp_sources']} / {m['harm_gt20pp_cells']} |")
        notes.extend(['', f"{ds} 的敏感参数：{', '.join(chosen['sensitive_parameters'])}。16来源单因素筛查后，", f"在32来源双序上进行12次坐标提案，实际{chosen['unique_refine_configs']}个独立配置（重复提案复用）。", f"本数据集共{table['completed_unique_trials']}个完整评分配置、{table['invalid_trials']}个保留但未评分的数值无效配置。"])
    lines.extend(notes)
    for failure in failures:
        lines.extend(['', f"数值无效配置保留：{failure['dataset']} {failure['trial']}，{failure['config']}，失败前保存{failure['preserved_prefix']}条。该配置未评分、不能获选，没有跳过失败query或用fallback补分。"])
    lines.extend(['', 'Native WHEN、单DINO、Uniform4、原admission/Top1、1792参数、M32 antithetic与真实likelihood结构均固定。',
                  'GPU预测封存后才进行开发集评分。未使用原128确认来源挑参数，没有新增主方法全量、corruption或其他消融。',
                  '不同筛查/细化阶段的来源数不同，不能把16来源默认值与32来源获选值直接解释成调参的配对收益。',
                  '', f"独立复核覆盖{verified_fields}个指标聚合与{raw_payloads}个预测文件哈希；当前/继承/总收益分开，负尾与数值失败保留。", 
                  '原baseline从完整Adam/LN与已保存输出链续接。HC源训练媒体缺失仍是完整EATA的依赖，不用目标验证数据替代。',
                  'GT没有进入在线loss、admission、输出轮次或reset。调参后的独立效能检验仍待另行授权。',
                  '', '![单因素敏感性](SENSITIVITY.png)', '', '![选定配置的收益分解](SELECTED_DEVELOPMENT.png)', ''])
    (PUB / 'REVIEW.md').write_text('\n'.join(lines))
    status(BASE / 'ROOT_AUDIT_COMPLETION.json', dict(status='audited_pending_actual_visual_review_remote_publication',
            result_files={str(p.relative_to(ROOT)): sha(p) for p in [output, PUB / 'TRIALS.csv', PUB / 'NUMERICAL_FAILURES.json',
            PUB / 'REVIEW.md', PUB / 'SENSITIVITY.png', PUB / 'SELECTED_DEVELOPMENT.png']}, time=time.time()))
    archive('两集参数实际锁定后的独立CPU复核已完成：全部已封存trial的source统计、10000 bootstrap、收益分解、数值失败与选定配置均回读一致；报告图已生成，仍待根实际目检与最终GitHub结果核验；baseline按保存状态接续，其他方法实验未恢复')


def draw(records, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
    factors = ['lr', 'sigma', 'tau', 'steps', 'writeback']
    for ax, factor in zip(axes, factors):
        for ds, color in [('vidstg', '#215f9a'), ('hc2', '#bd5b25')]:
            rows = sorted([r for r in records if r['dataset'] == ds and r['phase'] == 'screen' and
                           r['factor'] in [factor, 'reference']], key=lambda r: r[factor])
            x = [r[factor] for r in rows]; y = [r['delta_v_pp'] for r in rows]
            ax.plot(x, y, marker='o', color=color, label=ds)
            if factor == 'lr': ax.set_xscale('log')
        ax.axhline(0, color='#777', lw=.8); ax.set_title(factor); ax.set_ylabel('Development delta vIoU (pp)'); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle('Single-factor sensitivity: 16 exposed sources, two orders'); fig.tight_layout()
    fig.savefig(PUB / 'SENSITIVITY.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (ds, table) in zip(axes, tables.items()):
        metrics = table['selection']['statistics']['metrics']
        fields = ['delta_total_v', 'delta_current_v', 'delta_inherited_v']
        means = np.array([metrics[f]['mean'] * 100 for f in fields])
        bounds = np.array([metrics[f]['ci95'] for f in fields]) * 100
        # Bootstrap percentile intervals need not straddle a point estimate.
        ax.scatter(range(3), means, s=45, color='#215f9a', zorder=3)
        for i, (lo, hi) in enumerate(bounds): ax.plot([i, i], [lo, hi], color='#215f9a', lw=2)
        ax.axhline(0, color='#777', lw=.8); ax.set_xticks(range(3), ['total', 'current', 'inherited'])
        ax.set_ylabel('Development delta vIoU (pp)'); ax.set_title(ds); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Selected configuration: 32 exposed sources, two orders; selection-biased intervals')
    fig.tight_layout(); fig.savefig(PUB / 'SELECTED_DEVELOPMENT.png', dpi=160); plt.close(fig)


def wait_then_run(timeout_seconds=14400):
    verify_root_pin()
    deadline = time.monotonic() + timeout_seconds
    while not (BASE / 'TUNING_COMPLETION.json').exists():
        current = read(BASE / 'STATUS.json')
        if current['status'] == 'failed_preserved':
            raise RuntimeError('Tuning controller failed; original failure retained')
        assert time.monotonic() < deadline, 'Bounded post-selection audit wait expired'
        status(BASE / 'ROOT_AUDIT_STAGE.json', dict(status='waiting_for_actual_both_dataset_selection',
                    pid=os.getpid(), GT_read=False, time=time.time()))
        time.sleep(30)
    status(BASE / 'ROOT_AUDIT_STAGE.json', dict(status='postselection_saved_rows_audit', pid=os.getpid(), time=time.time()))
    run()
    status(BASE / 'ROOT_AUDIT_STAGE.json', dict(status='pending_actual_visual_and_publication', pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    try:
        if '--wait' in sys.argv: wait_then_run()
        else:
            run()
            status(BASE / 'ROOT_AUDIT_STAGE.json', dict(status='pending_actual_visual_and_publication', pid=os.getpid(), time=time.time()))
    except BaseException:
        failure = BASE / 'root_audit_failures' / str(time.time_ns()); failure.mkdir(parents=True, exist_ok=True)
        (failure / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'ROOT_AUDIT_STAGE.json', dict(status='failed_preserved', failure=str(failure), time=time.time()))
        raise
