"""Root scalar-only analysis after the independently verified full timing report."""
from runtime_v4_common import *
import csv
import numpy as np


def run():
    started = time.perf_counter()
    plan = validate_runtime()
    audit = read(D / 'reporting_receipt.json')
    assert audit['status'] == 'verified_all1536_Source_output_input_reference_SHA_and_repeat_paired_report'
    assert read(D / 'CPU_report_v5_exit.json')['code'] == 0
    assert read(D / 'GPU_exit.json')['code'] == read(D / 'teacher_exit.json')['code'] == 0
    report = read(P / 'paired_timing_memory.json')
    assert sha(P / 'paired_timing_memory.json') == audit['report_sha256']
    measured = read(D / 'measurements.json')
    assert sha(D / 'measurements.json') == audit['measurement_sha256']
    items = measured['items']
    assert len(items) == 1536 and all(not e['warmup'] and not e['GT_read'] for e in items)
    parent = read(D / 'worker.json'); teacher = measured['loads']['ReferDINO']['worker']
    scalar_calls = []
    for e in items:
        peaks = e['teacher_phase_peaks']
        assert e['peak_pipeline_allocated_bytes'] == max([e['parent_peak_allocated_bytes']] + [p['allocated'] for p in peaks])
        assert e['peak_pipeline_reserved_bytes'] >= e['peak_pipeline_allocated_bytes'] > 0
        assert e['NVML']['samples'] > 0 and e['NVML']['sampling_interval_seconds'] == .01
        assert e['NVML']['target_PIDs'] == sorted([parent['pid']] + ([teacher['pid']] if e['arm'] == 'Current' else []))
        if e['arm'] == 'Current':
            assert len(peaks) == 1 and e['stage_seconds']['auxiliary_DP_seconds'] == 0
        else:
            assert not peaks
        scalar_calls.append({k: v for k, v in e.items() if k not in ['prediction_path', 'positions8']})
    output = dict(datasets={}, failure_lineage=read(P / 'failure_and_actual_cost_lineage_v1.json'),
                  report_path_failure=dict(actual_exit=read(D / 'CPU_report_exit.json'),
                                          failed_report_elapsed_seconds=None,
                                          missing_clock_is_not_zero=True,
                                          preserved_source_measurements_prediction_SHA=True,
                                          corrected_report_actual_exit=read(D / 'CPU_report_v5_exit.json'),
                                          correction='Only hardware path D/hardware.json to BASE/hardware.json; all original sources frozen.'),
                  actual_counts=dict(formal_Source=1536, formal_ReferDINO=768, formal_EdgeTAM=768,
                                     excluded_warmup_Source=4, excluded_warmup_ReferDINO=2, excluded_warmup_EdgeTAM=2,
                                     failed_Source=2, failed_ReferDINO=2, diagnostic_ReferDINO=1,
                                     total_actual_Source=1542, total_actual_ReferDINO=773, total_actual_EdgeTAM=770))
    lines = [
        '# Matched complete-pipeline timing and peak memory', '',
        'Both pipelines include actual Source inference, decode, parse, transfers/offload and their own complete inference to CPU boxes/interval. All 1536 measured calls and all outputs were independently checked exactly; no GT scoring or model rerun was needed for this analysis.', '',
        'Fixed128 distinct parents/domain, three actual repeats per query/arm. Average the three latency repeats within each query before the 128-parent paired bootstrap (10000, seed20260920). The panel has prior development exposure. Confidence intervals are pointwise, not familywise.', '',
        '| Domain | Current mean / p50 / p90 seconds | Source-box EdgeTAM mean / p50 / p90 seconds | Edge minus Current seconds (95% CI) | Edge / Current ratio (95% CI) |',
        '|---|---|---|---|---|',
    ]
    accuracy_file = R / 'results/source_prompt_edgetam_v1/three_single_expert_means_v1.json'
    prior_accuracy = read(accuracy_file)
    for ds in ['hc2', 'vidstg']:
        paired = report['statistics'][ds]['paired_repeat_averages']
        assert len(paired) == 128
        rng = np.random.default_rng(20260920)
        idx = rng.integers(0, 128, (10000, 128))
        dataset = dict(stage_means={}, per_repeat_means={}, memory={}, paired_signed=[])
        source_scores = R / f'results/source_prompt_edgetam_v1/{ds}/all_signed.tsv'
        with source_scores.open(newline='') as h:
            rows = list(csv.DictReader(h, delimiter='\t'))
        assert len(rows) == 128 and len({x['query'] for x in rows}) == 128
        for arm in ['Current', 'EdgeTAM']:
            calls = [e for e in items if e['dataset'] == ds and e['arm'] == arm]
            assert len(calls) == 384
            for i, q in enumerate(paired, 1):
                sub = [e for e in calls if e['query'] == i]
                assert len(sub) == 3 and set(e['repeat'] for e in sub) == {1, 2, 3}
                assert q['query'] == i and q[arm]['seconds'] == float(np.mean([e['query_wall_seconds'] for e in sub]))
            dataset['stage_means'][arm] = {key: float(np.mean([e['stage_seconds'][key] for e in calls])) for key in calls[0]['stage_seconds']}
            dataset['stage_means'][arm].update(RGB_seconds=float(np.mean([e['RGB_seconds'] for e in calls])), parse_seconds=float(np.mean([e['parse_seconds'] for e in calls])))
            dataset['per_repeat_means'][arm] = {str(rep): float(np.mean([e['query_wall_seconds'] for e in calls if e['repeat'] == rep])) for rep in [1, 2, 3]}
            dataset['memory'][arm] = {}
            for field in ['allocated_peak_bytes', 'reserved_peak_bytes', 'NVML_peak_bytes']:
                a = np.array([q[arm][field] for q in paired], dtype=np.float64)
                dataset['memory'][arm][field] = dict(mean_query_max_over3_GiB=float(a.mean() / 2**30),
                                                   median_query_max_over3_GiB=float(np.median(a) / 2**30),
                                                   p90_query_max_over3_GiB=float(np.quantile(a, .9) / 2**30),
                                                   maximum_GiB=float(a.max() / 2**30))
        a = np.array([q['Current']['seconds'] for q in paired]); b = np.array([q['EdgeTAM']['seconds'] for q in paired])
        stats = report['statistics'][ds]
        assert stats['EdgeTAM_minus_Current_seconds'] == float((b-a).mean())
        assert np.array_equal(np.quantile((b-a)[idx].mean(1), [.025, .975]), stats['difference_95CI'])
        assert np.array_equal(np.quantile(b[idx].mean(1)/a[idx].mean(1), [.025, .975]), stats['ratio_of_means_95CI'])
        for i, (current, edge) in enumerate(zip(a, b), 1):
            dataset['paired_signed'].append(dict(query=f'Q{i:03}', Current_seconds=float(current), EdgeTAM_seconds=float(edge),
                                                EdgeTAM_minus_Current_seconds=float(edge-current),
                                                Current_minus_EdgeTAM_v_pp=100*(float(rows[i-1]['CurrentOurs_v'])-float(rows[i-1]['EdgeTAM_v']))))
        for name, field, oldname in [('Current', 'CurrentOurs', 'ReferDINO_current'), ('EdgeTAM', 'EdgeTAM', 'Source_box_EdgeTAM')]:
            for metric in ['t', 'v']:
                value=100*float(np.mean([float(row[field+'_'+metric]) for row in rows]))
                assert abs(value-prior_accuracy[ds][oldname][metric]) < 1e-12
        dataset['reused_prior_accuracy_percent'] = {name: prior_accuracy[ds][name] for name in ['ReferDINO_current', 'Source_box_EdgeTAM']}
        dataset['prior_accuracy_input_sha256'] = sha(source_scores)
        dataset['latency'] = {k:v for k,v in stats.items() if k != 'paired_repeat_averages'}
        dataset['Current_lower_latency_fraction'] = float((a < b).mean())
        dataset['Current_relative_latency_reduction'] = float(1-a.mean()/b.mean())
        output['datasets'][ds] = dataset
        u,v=stats['arms']['Current'],stats['arms']['EdgeTAM'];ci=stats['difference_95CI'];ri=stats['ratio_of_means_95CI']
        lines.append(f"| {ds} | {u['mean_seconds']:.6f} / {u['median_query_mean_seconds']:.6f} / {u['p90_query_mean_seconds']:.6f} | {v['mean_seconds']:.6f} / {v['median_query_mean_seconds']:.6f} / {v['p90_query_mean_seconds']:.6f} | {stats['EdgeTAM_minus_Current_seconds']:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {stats['EdgeTAM_over_Current_ratio_of_means']:.6f} [{ri[0]:.6f}, {ri[1]:.6f}] |")
    lines += ['', 'Current has lower mean latency for every one of the128 repeat-averaged queries in each domain. This is a comparison of the complete deployment choices under their original component backends, not a claim of isolated propagation superiority, equal compute/knowledge, full-dataset throughput or uniformly matched arithmetic kernels.', '',
              '| Domain / pipeline | Max PyTorch allocated GiB | Max PyTorch reserved GiB | Max sampled active-pipeline NVML GiB |',
              '|---|---|---|---|']
    for ds, x in output['datasets'].items():
        for arm, m in x['memory'].items():
            lines.append(f"| {ds} / {arm} | {m['allocated_peak_bytes']['maximum_GiB']:.6f} | {m['reserved_peak_bytes']['maximum_GiB']:.6f} | {m['NVML_peak_bytes']['maximum_GiB']:.6f} |")
    lines += ['', 'The memory tradeoff is not an unconditional Current advantage. Current has a larger active-process NVML peak in both domains; Source dominates the maximum PyTorch allocated and reserved peaks for both Vid pipelines, making those Vid maxima equal. Mean/median/p90 memory over the per-query maximum of three repeats are reported separately.', '',
              'PyTorch allocated/reserved are tensor/allocator counters. For Current, the exact pipeline peak is max(parent whole-call peak, parent live-at-synchronous-teacher-request plus teacher-child peak). Source is offloaded before teacher work and the parent waits; peaks from non-overlapping phases are not added. NVML samples only active pipeline PIDs every10ms and includes their CUDA contexts; unrelated processes and the inactive teacher PID for Edge are excluded. Sampling can miss sub10ms transients and is not an exact device-wide peak.', '',
              'RTX5090 / i5-14600KF, shared host, CPU14-15 with original Source/pool Torch4. Source and ReferDINO preserve cuDNN TF32=True/matmul=False; Edge preserves its original cuDNN=False/matmul=False and official BF16-memory/lossless FP32 V-input compatibility. Both arms use the same original Source backend. These are original-component backend results, not a uniform-precision operator experiment.', '',
              'All models remain CPU-resident between sequential GPU stages; normal transfers and offload are timed. Decode, parsing, Source, the respective complete expert/tracker path and output readout are included. There is no cross-query RGB/feature/teacher/prediction cache reuse. Experimental attempt/reference validation/storage/auditing occurs outside the timer. Startup and four real warmup calls are excluded and exported separately. Official native tracking progress output remains part of its normal implementation.', '',
              'Stage timings in all_measured_call_scalars_v1.json and root_full_analysis_v1.json are nested and differently scoped: expert request contains transfers/serialization/forward; finish contains pooling/prototype/MAP/finalDP, and the two arms have distinct Source-wrapper boundaries. Do not add these stages or interpret them as isolated causal contributions. Model-load scopes are per actual PID/start; Edge wrapper load includes its nested strict-load duration, not two loads.', '',
              'Repeat-level means are also retained. HC Current7.754915/7.017038/7.037440s and Edge10.845183/9.965312/9.972120s; Vid Current6.910008/6.813858/8.416342s and Edge12.039691/11.820441/14.298748s. Both Vid arms are slower in repeat3. Alternating arm order and paired query averaging limit fixed-order confounding but do not eliminate shared-host temporal variability. Do not attribute this drift to a specific service or claim an isolated hardware benchmark.', '',
              'Accuracy is reused only from the old complete128-parent per-domain independent scoring. All1536 actual timed predictions exactly match those references: HC current31.791298 vs Edge26.227517v; Vid15.339259 vs15.257178v. The prior Vid accuracy difference CI crosses zero, so latency advantage does not establish a Vid accuracy advantage or equivalence. No new GT scoring, target tuning or accuracy-based sample choice.', '',
              'Failure lineage is fully retained: two pre-formal warmup Source/ReferDINO failures, one additional actual original-backend expert diagnostic, and a zero-model namespace-launch failure. The first RAM-auth failure did not retain every Source intermediate; missing clocks are null, not zero. The independent CPU report then completed its checking/statistics loops but failed at hardware-path assembly. A separate report-v5 fixes only that path, retains the original code/logs and all1536 prediction hashes, and successfully rechecks every record without new inference. Successful GPU and teacher actual waits are0; original report wait1; repaired report actual wait0. No old controller was resumed.', '',
              'This completes the numerical/root analysis of timing, pending verified publication. The separately authorized cached temporal-correspondence study remains gated on timing publication. All older CG/gamma/full-corruption/generic pauses and cancelled CG-HC control remain unchanged.']
    atomic(P / 'all_measured_call_scalars_v1.json', scalar_calls)
    atomic(P / 'startup_warmup_v1.json', {k:v for k,v in measured['loads'].items() if k not in ['hc2','vidstg']} | {ds:dict(Source_and_parser_startup_seconds=measured['loads'][ds]['Source_and_parser_startup_seconds'], warmup=[{k:v for k,v in e.items() if k not in ['prediction_path','positions8']} for e in measured['loads'][ds]['warmup']]) for ds in ['hc2','vidstg']})
    atomic(P / 'root_full_analysis_v1.json', output)
    (P / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    atomic(P / 'actual_exit_cost_lineage_v5.json',dict(GPU_exit=read(D/'GPU_exit.json'),teacher_exit=read(D/'teacher_exit.json'),failed_report_exit=read(D/'CPU_report_exit.json'),repaired_report_exit=read(D/'CPU_report_v5_exit.json'),failure_lineage=output['failure_lineage'],counts=output['actual_counts'],failed_report_elapsed_seconds=None,failed_report_clock_missing_not_zero=True))
    rec=dict(status='root_all1536_scalar_memory_stage_cost_prior_accuracy_analysis_complete', code_sha256=sha(__file__), worker=identity(),
             report_sha256=sha(P/'paired_timing_memory.json'), measurement_sha256=sha(D/'measurements.json'),
             all_call_scalar_sha256=sha(P/'all_measured_call_scalars_v1.json'), root_analysis_sha256=sha(P/'root_full_analysis_v1.json'),
             ANALYSIS_sha256=sha(P/'ANALYSIS.md'), elapsed_wall_seconds=time.perf_counter()-started,
             all1536_checked=True,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_score=0)
    atomic(D/'root_export_receipt_v1.json',rec);atomic(P/'root_export_receipt_v1.json',rec)
    print(json.dumps({ds:dict(latency=x['latency'],memory=x['memory'])for ds,x in output['datasets'].items()}))


if __name__ == '__main__':
    run()
