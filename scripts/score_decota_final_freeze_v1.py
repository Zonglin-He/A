"""F46 offline scoring and independent arithmetic audit; never selects a method.

Source clusters, not repeated queries or frames, are the bootstrap units.
Known unavailable input and unscorable metrics are counted, never imputed as zero.
This module is not imported by the label-free GPU prediction worker.
"""
import argparse
import collections
import functools
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.special import expit

from scripts.decota_matrix_common_v1 import read, write, status, load, sha
from scripts.run_decota_final_freeze_v1 import OUT, ARMS, SUPPLEMENT, verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.analyze_structured_calibration_v1 import independent_evidence
from scripts.analyze_structured_temporal_v1 import independent_loss, change
from scripts.audit_dense_support_v1 import independent as independent_framewise
from vg_tta.dense_support_tuning_v1 import state_hash
from vg_tta.structured_temporal_shrinkage_v1 import interpolate_state

METRICS = ['vIoU_corrected', 'sIoU', 'tIoU', 'temporal_recall', 'temporal_precision', 'span']
PAIRS = [('Final_DeCoTA', 'Frozen'), ('Final_DeCoTA', 'A4'),
         ('Temporal_final', 'Frozen'), ('Final_DeCoTA', 'Temporal_final'),
         ('Final_DeCoTA', 'Spatial_interpolation'), ('Final_DeCoTA', 'Direct_projection'),
         ('A4', 'Frozen'), ('A4', 'Interpolation_native_time'),
         ('Final_DeCoTA', 'Full_update_eta1'), ('Final_DeCoTA', 'F39_framewise'),
         ('Final_DeCoTA', 'Logit_damping_eta025'), ('Final_DeCoTA', 'Old_DeCoTA')]


@functools.lru_cache(maxsize=4)
def bootstrap_indices(n, samples=10000, seed=20260915):
    return np.random.default_rng(seed).integers(n, size=(samples, n), dtype=np.int32)


def aggregate(values, sources, *, samples=10000, seed=20260915):
    """Values are fractions (not pp), or seconds in explicitly marked cost tables."""
    assert len(values) == len(sources)
    groups = collections.defaultdict(list)
    q = []
    for v, s in zip(values, sources):
        if v is not None:
            assert math.isfinite(v)
            groups[s].append(float(v)); q.append(float(v))
    result = dict(nominal_queries=len(values), nominal_sources=len(set(sources)),
                  queries=len(q), sources=len(groups), missing_queries=len(values)-len(q),
                  missing_sources=len(set(sources))-len(groups))
    if not groups:
        return dict(**result, mean=None, ci95=None, query_mean=None, source_values={})
    source_values = {s: float(np.mean(groups[s])) for s in sorted(groups)}
    a = np.array(list(source_values.values())); q = np.asarray(q)
    boot = a[bootstrap_indices(len(a), samples, seed)].mean(1)
    ntrim = int(.1*len(a)); ordered = np.sort(a)
    result.update(mean=float(a.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                  query_mean=float(q.mean()), source_values=source_values,
                  median=float(np.median(a)), trimmed10=float(ordered[ntrim:len(a)-ntrim].mean()),
                  minimum=float(a.min()), maximum=float(a.max()),
                  source_wins=int((a > .001).sum()), source_neutral=int((abs(a) <= .001).sum()),
                  source_harms=int((a < -.001).sum()),
                  negative_gt5pp=int((a < -.05).sum()), negative_gt10pp=int((a < -.1).sum()),
                  query_negative_gt5pp=int((q < -.05).sum()), query_negative_gt10pp=int((q < -.1).sum()),
                  worst10_mean=float(ordered[:max(1, math.ceil(.1*len(a)))].mean()))
    return result


def summary(rows):
    groups = [r['group'] for r in rows]
    out = dict(queries=len(rows), sources=len(set(groups)), arms={}, contrasts={})
    for name in ARMS+SUPPLEMENT:
        out['arms'][name] = {m: aggregate([r['arms'].get(name, {}).get(m) for r in rows], groups)
                             for m in METRICS}
    for a, b in PAIRS:
        cells = {}
        for m in METRICS:
            values = []
            for r in rows:
                x, y = r['arms'].get(a, {}).get(m), r['arms'].get(b, {}).get(m)
                values.append(None if x is None or y is None else x-y)
            cells[m] = aggregate(values, groups)
        out['contrasts'][a+' - '+b] = cells
    return out


def audit_prediction(x, counts):
    """Checks every query, not just the successful or endpoint-changing cases."""
    assert x['source_restored'] and x['expert_frozen'] and not x['GT_online']
    assert x['student_output_only'] and not x['interpolation_output']
    assert x['eta'] == .25 and x['temporal_parameters'] == 66306 and x['spatial_parameters'] == 1792
    assert set(x['predictions']) == set(ARMS+SUPPLEMENT)
    for a in [x['full_forward_audit'], x['predictions']['F39_framewise']['audit']]:
        assert a['full_model'] and a['source_restored'] and not a['GT_online']
        assert a['box_max_error'] == a['logit_max_error'] == 0
        counts['full_model_predictions'] += 1
    sp, tr, point, f39 = x['spatial'], x['temporal'], x['anchored_temporal'], x['supplementary']['F39']
    assert sp['parameter_count'] == 1792 and sp['planned'] == 4 and sp['gamma'] == 0 and sp['kappa'] is None
    assert sp['restore_exact'] and sp['student_output_only'] and not sp['GT_online']
    assert sp['steps'] == 10
    if sp['failure'] is not None:
        # A4 already has an explicit source-state restoration rule. Retain
        # these cases in the denominator instead of rejecting its fallback.
        assert sp['failure'] in ['nonfinite', 'nonfinite_gradient']
        assert sp['best_step'] == 0 and sp['state_delta'] == 0
        counts['spatial_numerical_fallback_queries'] += 1
    assert torch.equal(sp['final']['boxes'], x['boxes'])
    for name in ['A4', 'Final_DeCoTA', 'Direct_projection', 'F39_framewise', 'Full_update_eta1', 'Logit_damping_eta025']:
        assert torch.equal(x['predictions'][name]['boxes'], x['boxes'])
        counts['fixed_spatial_fields'] += 1
    assert torch.equal(x['predictions']['Temporal_final']['boxes'], x['native_boxes'])
    assert torch.equal(x['predictions']['Frozen']['boxes'], x['native_boxes'])
    assert x['predictions']['A4']['indices'] == x['predictions']['Frozen']['indices'] == x['I_seed']
    assert x['predictions']['Final_DeCoTA']['indices'] == x['predictions']['Temporal_final']['indices'] == x['I_out']
    assert x['expert']['new_DINO'] <= 4
    counts['DINO_calls'] += x['expert']['new_DINO']
    for t, record in zip(x['teacher'], x['records']):
        f = np.asarray(record['frame_ids'], float)
        edges = np.r_[f[0], (f[:-1]+f[1:])/2, f[-1]+1]
        assert t['sigmoid_applications'] == 1 and t['semantic_threshold'] is None
        assert len(t['a']) == len(f) and bool(t['valid_mask'].all())
        assert not t['a'].requires_grad and not t['raw_logits'].requires_grad
        assert np.array_equal(t['cell_edges'].numpy(), edges)
        assert np.max(abs(t['omega'].numpy()-np.diff(edges)/np.diff(edges).sum())) < 1e-14
        assert np.max(abs(expit(t['raw_logits'].numpy().astype(float))-t['a'].numpy())) < 1e-7
        counts['teacher_positions'] += len(f)
    assert tr['parameters'] == 66306 and tr['parameter_TTA'] and tr['steps'] == tr['backwards'] == 5
    assert tr['teacher_frozen'] and tr['source_restored'] and tr['spatial_invariant'] and not tr['GT_online']
    ev = independent_evidence(x['native_logits'], x['teacher'], x['records'], tr['config'])
    errors = []
    for a, b in zip(ev, tr['evidence']['offsets']):
        err = max(np.max(abs(a['g']-b['score'].numpy())), np.max(abs(a['cost']-b['cost'].numpy())))
        assert err < 1e-8 and a['top'] == b['target'] and a['interval'] == b['interval']
        errors.append(float(err)); counts['independent_projections'] += 1
    chosen = 0
    for i, st in enumerate(tr['path']):
        vals = independent_loss(st['logits'], ev, tr['config'], 1., 1.)
        err = max(abs(a-b) for a, b in zip(vals, [st['loss'], st['parts']['nll'], st['parts']['hinge'], st['parts']['kl']]))
        assert err < 1e-8; errors.append(float(err)); counts['independent_hard_loss_states'] += 1
        if st['loss'] < tr['path'][chosen]['loss']-1e-12: chosen = i
        if i and not tr['path'][i-1]['accepted']:
            assert st['state_sha256'] == tr['path'][i-1]['state_sha256']
        if i < 5:
            assert not st['unused_parameters'] and math.isfinite(st['gradient_norm'])
            assert st['optimizer_restored_on_reject'] == (not st['accepted'])
    assert chosen == tr['best_step'] and state_hash(tr['state']) == tr['path'][chosen]['state_sha256']
    assert state_hash(tr['initial_state']) == state_hash(x['temporal_cache']['head_state'])
    expected = interpolate_state(tr['initial_state'], tr['state'], .25)
    assert all(torch.equal(expected[n], point['state'][n]) for n in expected)
    assert state_hash(expected) == point['state_sha256']
    assert sum(v.numel() for v in expected.values()) == 66306
    assert f39['parameters'] == 66306 and f39['backwards'] == f39['steps'] == 5 and f39['failure'] is None
    assert f39['lr'] == .01 and f39['beta'] == 1. and f39['teacher_frozen'] and f39['actual_best_state_verified']
    assert f39['spatial_invariant'] and not f39['GT_online']
    for st in f39['path']:
        vals = independent_framewise(st['logits'], x['native_logits'], x['teacher'], f39['beta'])[:3]
        err = max(abs(a-b) for a,b in zip(vals, [st['loss'], st['parts']['event'], st['parts']['keep']]))
        assert err < 1e-8; errors.append(float(err)); counts['independent_F39_loss_states'] += 1
    assert state_hash(f39['state']) == f39['final']['state_sha256']
    counts['spatial_backward'] += sp['backwards']; counts['main_temporal_backward'] += tr['backwards']
    counts['supplement_temporal_backward'] += f39['backwards']
    counts['spatial_parameter_changed_queries'] += int(x['spatial_parameter_changed'])
    counts['temporal_parameter_changed_queries'] += int(x['temporal_parameter_changed'])
    counts['temporally_changed_queries'] += int(x['I_out'] != x['I_seed'])
    counts['no_spatial_accepted_anchors'] += int(not sp['anchors'])
    counts['queries_audited'] += 1
    return max(errors, default=0.)


def case_table(rows, comparison):
    a, b = comparison
    scored = []
    for r in rows:
        x, y = r['arms'].get(a, {}).get('vIoU_corrected'), r['arms'].get(b, {}).get('vIoU_corrected')
        if x is not None and y is not None:
            scored.append(dict(key=r['key'], group=r['group'], caption=r['caption'],
                delta_v_pp=100*(x-y), v_after=100*x, v_before=100*y,
                before_interval=r['intervals'][b], after_interval=r['intervals'][a],
                GT_interval=r['GT_interval'], event_fraction=r['event_fraction'],
                accepted_anchors=r['accepted_anchors'], temporal_change=r['temporal_change']))
    ordered = sorted(scored, key=lambda r:(r['delta_v_pp'], r['key']))
    return dict(comparison=a+' - '+b, query_level=True, good=ordered[-20:][::-1], failure=ordered[:20])


def cell(x, name='mean'):
    return 'NA' if x.get(name) is None else f"{100*x[name]:.3f}"


def report_text(cohort, s, counts, unavailable, costs):
    title = 'VidSTG → HC-STVG1' if cohort == 'hcstvg1_test' else 'HC-STVG2 → VidSTG'
    lines = [f'# F46 {title}：冻结方法的完整历史池回顾性评价', '',
        '**不是 held-out confirmation。** 所有本地可用来源已在旧全量研究中使用；F45全部为开发证据。',
        '本轮未选择新参数，未把负例/no-op/未发生更新的来源排除。单位为百分数。', '',
        f"名义{s['queries']}查询 / {s['sources']}来源；已知输入不可用{len(unavailable)}条。", '',
        '| 主方法 | 源宏 vIoU | 源宏 sIoU | 源宏 tIoU | query宏 vIoU | query宏 sIoU | query宏 tIoU | 可评分q/源(v) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ARMS:
        m=s['arms'][name]; v=m['vIoU_corrected']
        values=[cell(m[k], unit) for unit in ['mean','query_mean'] for k in METRICS[:3]]
        lines.append('| '+name+' | '+' | '.join(values)+f" | {v['queries']}/{v['sources']} |")
    lines += ['', 'sIoU固定在同一GT有效帧集合；它不随预测时间选择而换分母。不同空间方法的框可以不同。',
        'Spatial_interpolation与Final使用同4时刻接受观察、同最终时间；Direct_projection与Final使用同A4框。',
        'Old_DeCoTA复用原封存输出和原精度路径，不能将其与匹配FP32新基线的差异全归于算法。', '',
        '## 预锁消融与强对照', '', '| 方案 | vIoU | sIoU | tIoU |', '|---|---:|---:|---:|']
    for name in SUPPLEMENT:
        lines.append('| '+name+' | '+' | '.join(cell(s['arms'][name][k]) for k in METRICS[:3])+' |')
    lines += ['', '## 配对增量及负尾', '',
        '| 对比 | 源宏 Δv pp [95% CI] | 源级 >5/>10pp 负尾 | query级 >5/>10pp 负尾 |', '|---|---:|---:|---:|']
    for name, mm in s['contrasts'].items():
        m=mm['vIoU_corrected']; ci=m['ci95']
        ci_text='NA' if ci is None else f'[{100*ci[0]:.3f}, {100*ci[1]:.3f}]'
        lines.append(f"| {name} | {cell(m)} {ci_text} | {m.get('negative_gt5pp',0)}/{m.get('negative_gt10pp',0)} | {m.get('query_negative_gt5pp',0)}/{m.get('query_negative_gt10pp',0)} |")
    lines += ['', '10000次来源cluster配对bootstrap，seed20260915。一个源多query先源内均值，再等权。',
        '这些区间描述当前历史池的不确定性，不消除长期方案选择和数据曝光造成的偏差。', '',
        '## 成本与状态', '', f"实际DINO调用{counts['DINO_calls']}；实际空间反向{counts['spatial_backward']}、主时间反向{counts['main_temporal_backward']}。",
        f"主时间参数改变{counts['temporal_parameter_changed_queries']}条，最终合并区间改变{counts['temporally_changed_queries']}条。",
        f"完整主方法平均 {costs['pipeline_seconds']['query_mean']:.3f} 秒/query；含F39补充重放的worker平均 {costs['worker_seconds']['query_mean']:.3f} 秒/query。",
        '成本为同一worker共享原始输入和控制状态的实际组件计时，不伪称11个方法各自独立端到端测速。',
        '详细组件、峰值显存、损失/状态/真实模型重放检查见COST.json及AUDIT.json。', '',
        '## 结论边界', '', '算法设计已固定；这些结果不自动触发新教师、新搜索或空间/时间-only回退。',
        '案例只用于本次误差说明，不据此更改eta、参数、观察选择或主结果集合。', '']
    return '\n'.join(lines)


def score_cohort(cohort, partial=False):
    torch.set_num_threads(2)
    p=verify(); dest=OUT/'analysis'/cohort
    if not partial and (dest/'COMPLETION.json').exists():
        seal=read(dest/'COMPLETION.json')
        for path,h in seal['files'].items(): assert sha(path)==h
        print('F46 scoring already complete',cohort);return
    run=OUT/'runs'/cohort; barrier=run/'BARRIER.json'
    if not partial: assert barrier.exists() and read(barrier)['all_available_complete']
    assert sha(p['labels'])==p['labels_sha256']; labels=read(p['labels'])
    rows=[]; receipts=[]; unavailable=[]; counts=collections.Counter(); errors=[]; cost_rows=[]
    started=time.time(); locksha=sha(OUT/'LOCK.json')
    for item in p['rows'][cohort]:
        path=run/f'{item["ordinal"]:06d}.pt'; rp=path.with_suffix('.json')
        if partial and not rp.exists(): continue
        receipt=read(rp); assert receipt['key']==item['key'] and receipt['lock_sha256']==locksha
        assert sha(path)==receipt['sha256']; receipts.append(receipt)
        x=load(path); q=item['input']; assert x['input']==q and x['key']==item['key']
        assert x['lock_sha256']==locksha and not x['GT_online']
        if x.get('status')=='known_input_unavailable':
            assert item['input_unavailable'] is not None and x['reason']==item['input_unavailable']
            unavailable.append(dict(key=item['key'],source=q['source'],reason=x['reason']))
            rows.append(dict(key=item['key'],group=q['source'],status=x['status'],arms={}))
            continue
        assert item['input_unavailable'] is None and x['historically_exposed']
        assert x['frame_ids']==q['frame_ids']; errors.append(audit_prediction(x,counts))
        gt=labels[item['key']]; ids=x['frame_ids']; metrics={}; intervals={}
        for name,pred in x['predictions'].items():
            a,b=pred['indices']; assert 0<=a<b<len(ids)
            assert tuple(pred['boxes'].shape)==(len(ids),4) and bool(torch.isfinite(pred['boxes']).all())
            intervals[name]=[ids[a],ids[b]+1]
            if 'physical_interval' in pred: assert pred['physical_interval']==intervals[name]
            m,_=checked_score(pred['boxes'],gt,ids,pred['indices']);metrics[name]=m
            counts['independently_rescored_predictions']+=1
        valid=np.asarray(gt['valid'],bool)
        r=dict(key=item['key'],group=q['source'],caption=q['caption'],query_type=item['query_type'],
            video_path=q['video_path'],video_sha256=q['video_sha256'],status='scored',arms=metrics,
            intervals=intervals,GT_interval=gt['interval'],valid_GT_positions=int(valid.sum()),
            observed_positions=len(ids),event_fraction=(gt['interval'][1]-gt['interval'][0])/(ids[-1]+1-ids[0]),
            accepted_anchors=len(x['spatial']['anchors']),temporal_change=change(intervals['Frozen'],intervals['Final_DeCoTA']),
            temporal_parameter_changed=x['temporal_parameter_changed'],spatial_parameter_changed=x['spatial_parameter_changed'],
            temporal_state_delta=x['anchored_temporal']['state_delta'],spatial_state_delta=x['spatial']['state_delta'],
            temporal_best_step=x['temporal']['best_step'],spatial_best_step=x['spatial']['best_step'],
            spatial_numerical_failure=x['spatial']['failure'],
            raw_final_indices=x['predictions']['Final_DeCoTA']['raw_offset_indices'],
            previously_F45_development=item['key'] in p['f45_development_keys'],all_pool_historically_exposed=True)
        rows.append(r)
        cost_rows.append(dict(group=q['source'],**x['cost'],worker_seconds=x['seconds'],
            F39_fit_seconds=x['supplementary']['F39_fit_seconds'],
            F39_full_replay_seconds=x['supplementary']['F39_full_replay_seconds'],
            interpolation_seconds=x['supplementary']['interpolation_seconds'],decode_seconds=x['decode_seconds'],
            peak_memory_bytes=x['peak_memory_bytes']))
        if len(rows)%100==0: print('F46 offline audit',cohort,len(rows),flush=True)
    if not partial:
        assert len(rows)==len(p['rows'][cohort]) and receipts==read(barrier)['receipts']
    assert rows and cost_rows
    s=summary(rows); sources=[r['group'] for r in cost_rows]
    costs={}
    for k in cost_rows[0]:
        if k=='group': continue
        values=aggregate([r[k] for r in cost_rows],sources)
        costs[k]={n:v for n,v in values.items() if n in
            ['nominal_queries','nominal_sources','queries','sources','missing_queries','missing_sources',
             'mean','ci95','query_mean','median','trimmed10','minimum','maximum','source_values']}
        costs[k]['unit']='bytes' if k=='peak_memory_bytes' else 'seconds'
    cases={a+' - '+b:case_table([r for r in rows if r['status']=='scored'],(a,b))
           for a,b in [('Final_DeCoTA','A4'),('A4','Frozen'),('Final_DeCoTA','Direct_projection')]}
    diagnostics=dict(temporal_change_counts=dict(collections.Counter(r.get('temporal_change') for r in rows if r['status']=='scored')),
        temporal_parameter_changed_but_interval_same=sum(r.get('temporal_parameter_changed',False) and r.get('temporal_change')=='no_op' for r in rows),
        offline_only=True,no_model_selection=True)
    audit=dict(status='pass',cohort=cohort,counts=dict(counts),max_independent_math_error=max(errors),
        known_unavailable=unavailable,full_pool=not partial,partial_smoke_only=partial,
        independent_test=False,GT_online=False,configuration_changes=0,
        F45_all_development=True,elapsed_seconds=time.time()-started,lock_sha256=locksha)
    if partial:
        smoke=OUT/'smoke_audit'/cohort/str(len(rows))
        write(smoke/'AUDIT.json',audit);write(smoke/'SUMMARY.json',s)
        print('F46 PARTIAL AUDIT PASS',cohort,len(rows),max(errors),flush=True);return
    files={}
    for name,obj in [('ALL_QUERY_RESULTS.json',dict(rows=rows)),('SUMMARY.json',s),('CASE_ANALYSIS.json',cases),
                     ('AUDIT.json',audit),('COST.json',costs),('DIAGNOSTICS.json',diagnostics)]:
        path=dest/name;write(path,obj);files[str(path)]=sha(path)
    md=dest/'RESULTS.md';assert not md.exists();md.write_text(report_text(cohort,s,counts,unavailable,costs))
    files[str(md)]=sha(md)
    write(dest/'COMPLETION.json',dict(finished=time.time(),files=files,all_available_complete=True,
        queries=len(rows),sources=len({r['group'] for r in rows}),unavailable=len(unavailable),
        status='completed_valid_retrospective',independent_confirmation=False,configuration_changes=0))
    print('F46 COHORT COMPLETE',cohort,len(rows),flush=True)


def finalize():
    p=verify(True); cohorts={};seals={}
    for cohort in p['rows']:
        base=OUT/'analysis'/cohort;seal=read(base/'COMPLETION.json')
        for path,h in seal['files'].items():assert sha(path)==h
        assert seal['all_available_complete'];cohorts[cohort]=read(base/'SUMMARY.json')
        seals[cohort]=dict(path=str(base/'COMPLETION.json'),sha256=sha(base/'COMPLETION.json'))
    contrasts={}
    for a,b in PAIRS:
        name=a+' - '+b;contrasts[name]={}
        for metric in METRICS:
            means=[s['contrasts'][name][metric]['mean'] for s in cohorts.values()]
            contrasts[name][metric]=None if any(x is None for x in means) else float(np.mean(means))
    allrows=[r for cohort in cohorts for r in read(OUT/'analysis'/cohort/'ALL_QUERY_RESULTS.json')['rows']]
    write(OUT/'ALL_SOURCE_RESULTS.json',dict(cohorts=cohorts,equal_direction_contrasts=contrasts,
        per_query_results=[str(OUT/'analysis'/c/'ALL_QUERY_RESULTS.json') for c in cohorts],
        queries=len(allrows),independent_test=False))
    lines=['# F46 Final DeCoTA：一次冻结后的完整历史池评价','',
        '**完成状态：完整可用池已运行和审计；不是独立确认。** F45和旧全量来源均已历史暴露。',
        '冻结设计为精确A4＋F44 hard structured temporal TTA＋全局参数收缩η=.25；没有追加搜索。','',
        '| 主方法 | HC源宏v/s/t | Vid源宏v/s/t | HC query宏v/s/t | Vid query宏v/s/t |',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        vals=[' / '.join(cell(cohorts[c]['arms'][arm][m],unit) for m in METRICS[:3])
              for unit in ['mean','query_mean'] for c in cohorts]
        lines.append('| '+arm+' | '+' | '.join(vals)+' |')
    lines += ['', '指标单位百分数；来源宏先同源query均值再来源等权。sIoU在固定GT帧计分。',
        'Source/Query counts、每方向CI、负尾、成本与全部good/failure记录见各方向RESULTS.md与JSON。', '',
        '## 时间分支关键增量', '']
    name='Final_DeCoTA - A4'
    for c,s in cohorts.items():
        m=s['contrasts'][name]['vIoU_corrected'];ci=m['ci95']
        lines.append(f"- {c}：Δv {cell(m)} pp，来源配对95% CI [{100*ci[0]:.3f}, {100*ci[1]:.3f}]；>5pp负尾来源{m['negative_gt5pp']}，query {m['query_negative_gt5pp']}。")
    lines += ['',f"两方向等权Δv {100*contrasts[name]['vIoU_corrected']:.3f} pp。不按本表重新选择eta或删除时间分支。",'',
        '## 可声明与不可声明', '',
        '本轮检验固定参数方法在完整可用历史池上的表现，报告包括所有有效输入、no-op与负结果。',
        '不声称未接触test、不声称普遍SOTA、不因有参数变化或loss下降声称任务成功。',
        '相同接受专家观察的绝对插值与同A4空间的直接时间投影均保留，强对照没有隐去。',
        '当前设计登记与独立泛化证明是两回事；若之后需要新域/新backbone，须另行授权冻结迁移测试。', '']
    report=OUT/'FULL_SYSTEM_RESULTS.md';assert not report.exists();report.write_text('\n'.join(lines))
    write(OUT/'CLAIMS.json',dict(method_fixed=True,eta=.25,A4_unchanged=True,new_search=False,
        F45_all_development=True,full_pool_retrospective=True,independent_confirmation=False,
        automatic_rollback_or_promotion=False,equal_direction_contrasts=contrasts))
    artifacts={str(f):sha(f) for folder in ['runs','analysis'] for f in (OUT/folder).rglob('*')
               if f.is_file() and f.suffix in ['.pt','.json','.md'] and f.name not in ['PROGRESS.json','FAILURE.json']}
    for name in ['ALL_SOURCE_RESULTS.json','FULL_SYSTEM_RESULTS.md','CLAIMS.json','LOCK.json','FREEZE_RECEIPT.json']:
        artifacts[str(OUT/name)]=sha(OUT/name)
    write(OUT/'COMPLETION.json',dict(finished=time.time(),cohorts=seals,artifacts=artifacts,
        code_pins=p['code_pins'],status='completed_valid_full_pool_retrospective',configuration_changes=0,
        actual_nominal_queries=len(allrows),available_queries=sum(r['status']=='scored' for r in allrows),
        source_counts={c:s['sources'] for c,s in cohorts.items()},independent_test=False))
    status(OUT/'STATUS.json',dict(stage='completed',finished=True,updated=time.time(),
        report=str(report),full_pool_retrospective=True,independent_test=False,new_search=False,
        finite_not_recurring=True,archive_update_required_before_next_user_completion_report=True))
    print('F46 FULL POOL FINISHED; no new experiments or retuning started',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['hcstvg1_test','vidstg_test'])
    ap.add_argument('--partial',action='store_true');ap.add_argument('--finalize',action='store_true')
    args=ap.parse_args()
    if args.finalize:finalize()
    else:
        assert args.cohort;score_cohort(args.cohort,args.partial)
