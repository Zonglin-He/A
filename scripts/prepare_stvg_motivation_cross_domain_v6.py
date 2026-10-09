"""CPU preregistration only: checkpoints as bytes, sanitized cohorts, runtime pins."""
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_motivation_cross_domain_common_v6 import *


def run():
    import numpy as np
    assert not (BASE/'RUNTIME_LOCK.json').exists(), 'Preserve an existing registration'
    BASE.mkdir(parents=True, exist_ok=True)
    history = ROOT/'artifacts/stvg_native_support_fig1_v1/uniform64_v2'
    oldvid = read(history/'ROSTER.json'); oldsubjects = read(history/'SUBJECTS.json')
    paperhc = read(PAPER/'hc2/PLAN.json')
    p0 = read(P1/'DESIGN_LOCK.json')
    hcparents = p0['datasets']['hc2']['confirmation_query_ordinals']
    assert len(hcparents) == 128 and p0['confirmation_is_previously_exposed']
    metadata = {}
    for rel in ['artifacts/stvg_native_support_fig1_v1/uniform64_v2/ROSTER.json',
                'artifacts/stvg_native_support_fig1_v1/uniform64_v2/SUBJECTS.json',
                'artifacts/stvg_opd_paper_hc2_revision_v2/DESIGN_LOCK.json',
                'artifacts/decota_paper_experiments_v1/hc2/PLAN.json',
                'artifacts/decota_paper_experiments_v1/hc2/SUBJECT_BARRIER.json',
                'methods/CURRENT_METHOD.json', 'external/TubeDETR/README.md',
                'checkpoints/ParallelTubeDecoding-Qwen3-VL-4B/README.md']:
        metadata[rel] = sha(ROOT/rel)
    designs = {}
    for direction in DIRECTIONS:
        source, target = direction_datasets(direction)
        selected = [paperhc['rows'][i] for i in hcparents] if target == 'hc2' else oldvid['rows']
        rows = []; subjects = {}
        hb = read(PAPER/'hc2/SUBJECT_BARRIER.json') if target == 'hc2' else None
        for ordinal, old in enumerate(selected):
            row = copy.deepcopy(old)
            ids = row['frame_ids']
            pos = np.rint(np.linspace(0, len(ids)-1, min(len(ids), 64))).astype(int).tolist()
            chosen = [ids[i] for i in pos]
            assert len(chosen) >= 4 and chosen == sorted(set(chosen))
            assert chosen[0] == ids[0] and chosen[-1] == ids[-1]
            row['historical_ordinal'] = row['ordinal']; row['ordinal'] = ordinal
            row['original_frame_ids'] = ids; row['frame_ids'] = chosen; row['input']['frame_ids'] = chosen
            if target == 'hc2':
                name = f"{old['ordinal']:05}.json"
                path = PAPER/'hc2/subjects'/name
                assert sha(path) == hb['files'][name]
                value = read(path)
                assert not value['GT_read'] and value['caption_sha256'] == hashlib.sha256(row['input']['caption'].encode()).hexdigest()
                subjects[str(ordinal)] = value['parses']['subject']
                metadata[str(path.relative_to(ROOT))] = sha(path)
            else:
                subjects[str(ordinal)] = oldsubjects[str(old['ordinal'])]
            rows.append(row)
        assert len(rows) == len({r['source'] for r in rows}) == 128
        roster = dict(direction=direction, source_dataset=source, target_dataset=target,
            rows=rows, queries=128, parent_sources=128, historical_exposure=True,
            score_selected=False, conditions=['clean'], uniform_max_frames=64,
            common_RGB_for_all_backbones=True, native_readout_only=True, GT_read=False)
        write(BASE/direction/'ROSTER.json', roster); write(BASE/direction/'SUBJECTS.json', subjects)
        for name in ['ROSTER.json', 'SUBJECTS.json']:
            rel = str((BASE/direction/name).relative_to(ROOT)); metadata[rel] = sha(ROOT/rel)
        designs[direction] = dict(source_dataset=source, target_dataset=target,
            queries=128, parent_sources=128, frames_per_query=min(64, max(len(r['frame_ids']) for r in rows)),
            roster_sha256=sha(BASE/direction/'ROSTER.json'),
            qualification_ordinals=[0, 1], models=list(MODELS), cells=256,
            cohort_origin='original P0 HC2 confirmation list' if target == 'hc2' else 'original native-support VidSTG 128-parent list')
    checkpoints = {}
    for model in MODELS:
        checkpoints[model] = {}
        for source in ['vidstg', 'hc2']:
            path = verify_checkpoint(model, source)
            checkpoints[model][source] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), bytes=path.stat().st_size,
                STVG_training_dataset=source, target_STVG_dataset_training_claim=False)
    design = dict(status='locked_before_new_GPU_or_GT', directions=designs, checkpoints=checkpoints,
        cells=512, qualification_cells=8, main_thresholds=dict(tIoU=.5, GT_frame_mean_IoU=.5, strict_greater_than=True),
        paired_parent_bootstrap=10000, seed=SEED, historically_exposed=True,
        no_formal_OPD_payload_access=True, parameter_updates=0, new_training=False,
        PTD_excluded='official model card lists joint VidSTG and HC-STVG training',
        white_background=True, outer_or_panel_borders=False, overall_figure_title=False,
        human_scope='Figure1 PanelB must be cross domain', GT_read=False, time=time.time())
    write(BASE/'DESIGN_LOCK.json', design)
    files = ['protocols/stvg_motivation_cross_domain_v6.md',
        'scripts/stvg_motivation_cross_domain_common_v6.py',
        'scripts/prepare_stvg_motivation_cross_domain_v6.py',
        'scripts/run_stvg_motivation_cross_domain_v6.py',
        'scripts/continue_stvg_motivation_cross_domain_v6.py',
        'scripts/score_stvg_motivation_cross_domain_v6.py',
        'scripts/test_stvg_motivation_cross_domain_v6.py',
        'scripts/render_stvg_motivation_cross_domain_v6.py',
        'vg_tta/stvg_motivation_quadrants_v6.py']
    # Keep every existing native implementation pin, including preprocessing,
    # exact frame decoders and the source loader, without editing those files.
    original = read(history/'RUNTIME_LOCK.json')
    code = dict(original['pins'])
    for f in sorted((history/'revisions').glob('*.json')):
        code.update(read(f)['pins'])
    # CURRENT has legitimately advanced since the October 1 native diagnostic.
    # It is protected by its CURRENT byte hash in this new metadata lock; the
    # old runtime, registry pin and same-domain result remain unchanged.
    prior_registry_sha256 = code.pop('methods/CURRENT_METHOD.json', None)
    for rel in ['scripts/run_tastvg_evidence_vulnerability_v2.py',
                'vg_tta/tastvg_paper48_hc2_decode_v1.py',
                'vg_tta/tastvg_paper48_hc2_metrics_v1.py',
                'external/TA-STVG/datasets/evaluation/hcstvg_eval.py'] + files:
        code[rel] = sha(ROOT/rel)
    for rel, h in code.items():
        assert sha(ROOT/rel) == h, rel
    write(BASE/'RUNTIME_LOCK.json', dict(status='pinned_before_native_qualification', code=code,
        metadata=metadata, design_sha256=sha(BASE/'DESIGN_LOCK.json'),
        checkpoint_bytes=checkpoints, original_native_runtime_sha256=sha(history/'RUNTIME_LOCK.json'),
        previous_native_registry_sha256=prior_registry_sha256,
        current_registry_sha256=metadata['methods/CURRENT_METHOD.json'],
        preserves_CURRENT=True, preserves_original_OPD_science=True, GT_read=False, time=time.time()))
    status(BASE/'STATUS.json', dict(status='registered_waiting_for_current_P1_GPU_release',
        formal_predictions=0, qualification_cells_passed=0, GT_read=False, GPU_started=False,
        cells_total=512, source_training_only=True, whole_figure_complete=False, time=time.time()))
    print('Registered 512 cross-domain native predictions; 8 real qualification cells still pending.', flush=True)


if __name__ == '__main__':
    run()
