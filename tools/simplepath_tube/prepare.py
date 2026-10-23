"""Freeze a bounded, no-label real adapter qualification after old full publication."""
import json
from pathlib import Path
from stvg.runtime import ROOT, sha
from stvg.tube_runner import write


def main():
    root = ROOT
    pub = root / 'results/simplepath_tubedetr'
    path = pub / 'qualification_plan.json'
    if path.exists():
        raise ValueError('Already frozen qualification plan; never overwrite')
    final = root / 'outputs/maintenance/llava_st_final_publication.json'
    assert json.loads(final.read_text())['final_full_results_published'] is True
    previous = root / 'results/tubedetr_boxfixed_development/plan.json'
    old = json.loads(previous.read_text())
    upstream = json.loads((root / 'runtime/upstream_main/a238705af9203c4cda407d7d1199ce18b2e6b72f/configs/default.json').read_text())
    files = [*sorted((root / 'tools/simplepath_tube').rglob('*.py')),
             *sorted((root / 'tools/reuse_cache').glob('*.py')),
             *[root / f for f in old['code_sha256'] if not f.startswith('tests/')]]
    datasets = json.loads(json.dumps(old['datasets']))
    for ds, d in datasets.items():
        d['configuration']['initialization_gamma'] = upstream['datasets'][ds]['initialization_gamma']
        d['configuration']['temporal_mode'] = 'native_last_layer_prior_one_prototype_update'
        for key in ['manifest', 'full_manifest']:
            assert sha(root / d[key]) == d[key + '_sha256']
    plan = dict(status='frozen_before_new_real_adapter_qualification',
                upstream_commit='a238705af9203c4cda407d7d1199ce18b2e6b72f',
                method='SimplePath + one all-video Native2D prototype pair',
                backbone='TubeDETR', datasets=datasets, dataset_order=['vidstg', 'hc2'],
                teacher_fingerprint=old['teacher_fingerprint'],
                prior='exact native last-layer FP32 full-grid endpoint posterior, start<end',
                source='original native MAP unchanged; no fake TA offsets or Global6',
                full_readout='original physical-time MBR after one native-prior update',
                local_representation='actual full-time fast input_proj grid before video/text encoder',
                spatial_representation='actual native decoder-input post-fusion visual token prefix',
                fixed=dict(kappa=64, beta=1, rounds=1, context=1.5,
                           observed_weight=.5, unobserved_weight=.5, link_weight=.25),
                retuning=False, network_updates=0, expert_count=1,
                qualification=dict(inputs='first original dev128 query in each domain',
                     maximum_native_passes=4, cold_passes=2, cached_prefix_passes=2,
                     new_teacher_calls=0, GT_reads=0, query_replacement=False,
                     requirements=['cold/cache Source, logits6, q0, spatial features and Full bitexact',
                                   'Source/native time/eight positions equal old complete sealed prediction',
                                   'actual pre-text grid/support finite and physically aligned',
                                   'cache-only exact teacher context', 'model state hash unchanged']),
                cache=dict(append_only=True, lossless=True, new_budget_bytes=64 << 30,
                           reserve_bytes=25 << 30, no_old_asset_deletion=True,
                           cross_query_prefix_reuse_requires_recorded_validation=True),
                future_evaluation='both fixed128 panels and both complete full populations after real qualification; no outcome-based tuning',
                scientific_scope='engineering adapter qualification only; no efficacy or unexposed-test claim',
                prior_LLaVA_final_publication_sha256=sha(final),
                previous_TubeDETR_plan_sha256=sha(previous),
                code_sha256={str(f.relative_to(root)): sha(f) for f in sorted(set(files))},
                official_code_sha256={str(f.relative_to(root)): sha(f)
                    for f in sorted((root / 'external/TubeDETR').rglob('*.py'))})
    write(path, plan)
    print('qualification plan', sha(path))


if __name__ == '__main__':
    main()
