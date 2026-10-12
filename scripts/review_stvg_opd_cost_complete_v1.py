"""Original full-fit root plus independent fresh-expert and timing readback."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import shutil
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha, load
from scripts.run_stvg_opd_cost_after_N_priority001 import SCIENCE as BASE, verify
from scripts.run_stvg_opd_end_to_end_cost_supplement_v1 import PUB, OLD, expert_science
from scripts.audit_stvg_opd_cost_complete_public_v1 import REL, KEYS, run as audit
OUT = ROOT/REL


def run():
    verify()
    assert read(BASE/'after_N_priority001/CPU_COMPLETION.json')['query_fits'] == 32
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.review_stvg_opd_supplement_v1 import cost
    cost()
    import torch
    torch.set_num_threads(2)
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    count = 0
    observed = 0
    for ds in ['hc2', 'vidstg']:
        complete = read(BASE/ds/'COMPLETION.json')
        expected = read(PUB/ds/'QUERY_TIMINGS.json')
        for i, binding in enumerate(read(BASE/'DESIGN_LOCK.json')['stages'][ds]['bindings']):
            f = BASE/ds/'fits'/f'{i:02}.pt'
            assert sha(f) == complete['files'][str(f.relative_to(ROOT))]
            z = load(f)
            old = load(ROOT/binding['path'])
            ip = OLD/old['input']['path']
            assert sha(ip) == z['original_input_sha256'] == old['input']['sha256']
            inp = normalized(load(ip))
            assert expert_science(z['expert']) == expert_science(inp['expert'])
            assert inp['source_model_state_sha256'] == expected['source_model_state_sha256']
            row = expected['rows'][i]
            assert row['anonymous_profile_id'] == i
            assert all(row[k] == z['timing'][k] for k in KEYS[:-1])
            assert row['total_seconds'] == z['total_seconds'] and row['dispatcher_seconds'] == z['dispatcher_seconds']
            assert row['admitted_references'] == len(z['fit']['positions'])
            assert row['backward_steps'] == z['fit']['gradient_calls']
            count += 1
            observed += len(z['expert']['observations'])
    assert count == 32 and not torch.cuda.is_initialized()
    write(BASE/'ROOT_FRESH_EXPERT_AND_TIMING_READBACK.json', dict(status='pass', complete_matching_fresh_expert_packs=count,
        observed_detector_receipts=observed, complete_saved_timing_payloads=count,
        source_checkpoint_native_pixel_frame_interval_and_expert_exact=True,
        original_science_and_runtime_unchanged=True, CPU_only=True, GT_read=False,
        new_GPU_calls=0, new_model_calls=0, time=time.time()))
    OUT.mkdir(parents=True, exist_ok=False)
    for ds in ['hc2', 'vidstg']:
        p = OUT/ds/'QUERY_TIMINGS.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PUB/ds/'QUERY_TIMINGS.json', p)
        assert sha(p) == sha(PUB/ds/'QUERY_TIMINGS.json')
    shutil.copy2(PUB/'END_TO_END_SUMMARY.json', OUT/'END_TO_END_SUMMARY.json')
    counts = {}
    summary = read(OUT/'END_TO_END_SUMMARY.json')
    for ds in ['hc2', 'vidstg']:
        rr = read(OUT/ds/'QUERY_TIMINGS.json')['rows']
        counts[ds] = dict(paths=len(rr), zero_DINO_paths=sum(r['actual_new_DINO_calls'] == 0 for r in rr),
            zero_admitted_paths=sum(r['admitted_references'] == 0 for r in rr), zero_backward_paths=sum(r['backward_steps'] == 0 for r in rr),
            actual_DINO_calls=sum(r['actual_new_DINO_calls'] for r in rr), admitted_references=sum(r['admitted_references'] for r in rr),
            actual_backward_steps=sum(r['backward_steps'] for r in rr))
    write(OUT/'ALL_PATH_COUNTS.json', dict(status='all32 actual paths retained', datasets=counts))
    checked = audit(ROOT)
    write(OUT/'PUBLIC_SCALAR_AUDIT.json', checked)
    write(BASE/'ROOT_INDEPENDENT_TIMING_STATISTICS.json', dict(checked,
        source_complete_fit_root_sha256=sha(BASE/'ROOT_COMPLETE_TIMED_FIT_READBACK.json'),
        source_expert_timing_root_sha256=sha(BASE/'ROOT_FRESH_EXPERT_AND_TIMING_READBACK.json'), time=time.time()))
    write(OUT/'DESIGN_PUBLIC.json', dict(status='complete_original_fixed32_audited_full_paths',
        paths=32, paths_per_target=16, fixed_cohort_seed=2026101104, original_science_source_and_runtime_unchanged=True,
        source_weights_resident=True, cold_application_frame_cache=True, OS_page_cache_cold=False,
        saved_query_parse=True, inactive_expert_storage_overhead_in_native_prefix=True,
        identity_check_overhead_in_native_prefix=True, peak_memory_is_this_process_not_whole_GPU=True,
        original_cancelled_S_CUDA_context_resident=True, all32_slow_no_information_paths_retained=True,
        baseline_remaining_formal_cancelled=True, cancelled_baseline_not_resumed=True,
        GT_read=False, production_promoted=False, manuscript_written=False, whole_followup_request_complete=False))
    lines = ['# Original audited Full cost: all32 actual paths', '',
        'Original locked historical16 queries per target, cohort seed2026101104. All32 measured Full trajectories and all fresh detector/input/native/source outputs exactly match their saved original scientific fits. Original parameters, frames, optimizer/readout and five-stage timer are unchanged. The original cancelled-baseline barrier dependency was replaced by genuine C/N FINAL; no baseline barrier was fabricated and no paused baseline was resumed.', '',
        '| Target | Paths | Total median (s) | Total p95 (s) | Actual DINO calls | Zero DINO | Zero admitted | Zero backward |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ds, s in summary['datasets'].items():
        c = counts[ds]
        lines.append(f"| {ds} | 16 | {s['timings']['total_seconds']['median']:.4f} | {s['timings']['total_seconds']['p95']:.4f} | {c['actual_DINO_calls']} | {c['zero_DINO_paths']} | {c['zero_admitted_paths']} | {c['zero_backward_paths']} |")
    lines += ['', '| Target | Stage | Median (s) | p95 (s) |', '|---|---|---:|---:|']
    for ds, s in summary['datasets'].items():
        for key in KEYS[:-1]:
            z = s['timings'][key]
            lines.append(f"| {ds} | {key} | {z['median']:.5f} | {z['p95']:.5f} |")
    lines += ['', 'The five stages are summed separately for every path. The median or p95 of a sum is not the sum of stage medians or p95s. All32 raw anonymous timings, actual detector/admission/backward counts, startup and this-process peak allocated/reserved memory are exported; no slow or no-information path is dropped.', '',
        'Weights are resident, the application frame cache is empty for every query, OS pages are not forcibly evicted, and query parsing is saved. All replay calls are CUDA synchronized; the qualified engineering recorder is included. The exact O012 frozen capture, inactive expert storage/restoration and S-identity check are included in the native-prefix timer. CPU hash/math/serialization follows the timed interval. The cancelled S optimizer/model CUDA context remains resident in its original stopped processes. No measured overhead is subtracted to claim production latency.', '',
        'These historical audited implementation measurements do not estimate fresh-population production throughput or whole-GPU memory use. They use a different fixed cohort and instrumentation from C\'s128 Full/Frozen paired deployment paths; their absolute seconds cannot be pooled as one speed benchmark. This study reads no GT and changes no method, gate, teacher or manuscript. Actual PDF pixel view, all anonymous public scalars, remote bytes/SHA/Git blob/mode/tree and archive precede genuine FINAL.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    pdf = ROOT/'output/pdf/stvg_opd_supplement/cold_decode_end_to_end_cost.pdf'
    write(BASE/'FIGURE_EXPORT.json', dict(status='pending_actual_PDF_pixel_view', PDF_files={str(pdf.relative_to(ROOT)): dict(sha256=sha(pdf), bytes=pdf.stat().st_size)},
        actual_timed_paths=32, CPU_only=True, GT_read=False, time=time.time()))
    print('COST32_COMPLETE_ROOT_AND_INDEPENDENT_STATISTICS', checked['scalar_comparisons'])


if __name__ == '__main__':
    run()
