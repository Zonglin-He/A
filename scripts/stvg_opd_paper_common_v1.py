"""Locked paper-suite metadata. Existing experiments and main-method bytes are immutable."""
import hashlib, json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, save, load
from scripts.decota_opd_tuning_common_v1 import verify_paper, bridge

BASE = ROOT / 'artifacts/stvg_opd_paper_v1'
PAPER = ROOT / 'artifacts/decota_paper_experiments_v1'
OLD = ROOT / 'artifacts/decota_spatial_opd_v1'
PUB = ROOT / 'results/stvg_opd_paper/2026-10-08'
PYTHON = ROOT / '.conda/tubedetr/bin/python'
ATTACHMENT = Path('/home/wwww/.codex/attachments/029fff2e-6089-4da7-a443-8f9bb07b9bef/已粘贴的文本.txt')
ARMS = ['on_policy', 'frozen_rollout', 'shuffled_feedback']
DATASETS = ['hc2', 'vidstg']
FAMILIES = ['frame_drop', 'frame_freeze', 'motion_blur', 'occlusion', 'exposure']

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def budget():
    assert shutil.disk_usage(ROOT).free > 8 * 2**30, 'Free disk below 8 GiB; retain evidence and stop.'

def committed(initial, final, alpha):
    import torch
    return {n: torch.zeros_like(v.cpu()) if n == 'spatial.query_residual' else
            v.cpu() + (final[n].cpu() - v.cpu()) * alpha for n, v in initial.items()}

def prepare():
    if (BASE / 'DESIGN_LOCK.json').exists():
        return read(BASE / 'DESIGN_LOCK.json')
    old = read(OLD / 'DESIGN_LOCK.json')
    configs = read(ROOT / 'methods/decota_spatial_opd_v1/configs.json')
    datasets, stages = {}, {}
    for ds in DATASETS:
        p = read(PAPER / ds / 'PLAN.json')
        confirm = old['stages']['confirm_' + ds]
        dev = old['stages']['dev_' + ds]
        assert len(confirm['parents']) == 128
        confirm_sources = {p['rows'][q]['source'] for q in confirm['parents']}
        dev_sources = {p['rows'][q]['source'] for q in dev['parents']}
        assert len(confirm_sources) == 128 and len(dev_sources) == 32
        assert not confirm_sources & dev_sources
        groups = {}
        for r in p['rows']:
            groups.setdefault(r['source'], []).append(r)
        representatives = [min(v, key=lambda r: digest(['opd-query-v1', r['key']]))['ordinal']
                           for _, v in sorted(groups.items())]
        balanced_order = sorted(representatives, key=lambda q: digest(['opd-balanced-order-v1', ds, p['rows'][q]['source']]))
        assert len(balanced_order) == (732 if ds == 'vidstg' else 237)
        cfg = configs['datasets'][ds]['config']
        assert cfg == dict(lr=.03, sigma=.1, tau=.25, steps=10 if ds == 'vidstg' else 20,
                           writeback=1/8 if ds == 'vidstg' else 1/16, samples=32)
        datasets[ds] = dict(config=cfg, source=confirm['source'],
            confirmation_query_ordinals=confirm['parents'], confirmation_orders=confirm['orders'],
            development_query_ordinals=dev['parents'], development_parent_sources=len(dev_sources),
            full_queries=len(p['rows']), full_parent_sources=len(groups), full_orders=p['orders'],
            balanced_one_query_per_parent=representatives, balanced_order=balanced_order,
            PLAN_sha256=sha(PAPER / ds / 'PLAN.json'), historically_exposed=True,
            confirmation_excluded_from_OPD_parameter_selection=True, fresh_unseen_claim=False)
        stages['P0_' + ds] = dict(dataset=ds, source=confirm['source'], split='locked_confirmation',
            parents=confirm['parents'], orders=confirm['orders'], conditions=['clean'], arms=ARMS,
            adapted_arrivals=768, Frozen_logical_arrivals=256, observation_budget=4)
        stages['P1_' + ds] = dict(dataset=ds, source=confirm['source'], split='full_cross_domain',
            parents=list(range(len(p['rows']))), orders=p['orders'], conditions=['clean'],
            arms=['on_policy'], adapted_arrivals=len(p['rows'])*3, observation_budget=4)
    d = dict(version='stvg_opd_paper_v1', attachment_sha256=sha(ATTACHMENT),
        datasets=datasets, stages=stages, fixed_method=True, retuning=False,
        execution_order=['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'], single_GPU_serial=True,
        P0_gate=dict(primary='paired parent-macro OPD minus Frozen vIoU', bootstrap=10000,
            stable_both_directions='both lower endpoints of paired 95% bootstrap CI strictly exceed zero',
            otherwise='root completes source-level failure attribution before a decision on expensive P1',
            mechanism_claims_separate=True, configuration_or_roster_selection_on_confirmation=False),
        later_stages=dict(
            P2=dict(arms=['Frozen','direct_L1_GIoU','shuffled_feedback','frozen_rollout','on_policy'],
                parents_per_target=128, clean_cross_orders=2, same_domain_five_5percent_orders=1,
                same_panel_as_P0=True, direct_weights=dict(L1=5.,GIoU=2.), last_step=True),
            P3=dict(arms=['Frozen','query_only','LN_only','joint_alpha0','on_policy'],
                parents_per_target=128, clean_cross_orders=2, same_domain_five_5percent_orders=1,
                independent_source_initialization=True, same_panel_as_P0=True),
            P4=dict(arms=['Frozen','on_policy'], cohort_unit='parent source', vid_parents=732,
                HC_parent_movies=237, HC_selected_clips=237, total_queries=969, orders=1,
                conditions=['clean']+[f'{f}_{c}' for c in [2.5,5,10] for f in FAMILIES],
                physical_burst_coverage_percent=[2.5,5,10], arrivals_per_online_method=15504,
                reset_each_condition=True, GT_event_independent_placement=True),
            P5=dict(budgets=[1,2,4,8], main_method_budget_unchanged=4,
                parents_per_target=128, cross_orders=2, no_budget_selection=True,
                cost_components=['DINO forward','STVG frozen forward','decoder fit GPU','CPU audit'],
                mechanism_cases=['correction success','identity-confused failure','expert gain without task gain']),
            P6=dict(parents_per_target=32, selection='hash subset of locked P0 parents before scoring',
                same_inputs_required=True, diagnostics=['Native WHEN','fixed deployable temporal signal',
                    'GT temporal-head oracle, offline only','Spatial OPD'],
                oracle_is_deployable_baseline=False, new_temporal_method=False)),
        appendix_unified_config=dict(lr=.03,sigma=.1,tau=.25,steps=10,writeback=1/16,samples=32),
        EATA_missing_direction='user paused and absent; no intake/Fisher/qualification/inference authorization inferred',
        reuse_baselines='sealed and scored Source/TENT/SAR/DINO/reference, existing EATA HC supplementary only',
        stage_GT_barrier='all deployment arms and both directions in that stage sealed before labels',
        time=time.time())
    write(BASE / 'DESIGN_LOCK.json', d)
    write(BASE / 'USER_SCOPE.json', dict(attachment_sha256=sha(ATTACHMENT),
        authorization='按照这个做: fixed-method OPD paper stages, P0 first with conditional P1',
        supersedes_other_OPD_experiment_pause=True, EATA_preparation_still_paused=True,
        CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        config_sha256=sha(ROOT/'methods/decota_spatial_opd_v1/configs.json'),time=time.time()))
    write(BASE / 'TABLE2_COHORT_RESOLUTION.json', dict(status='resolved_by_new_user_attachment',
        original_hold_sha256=sha(PAPER/'TABLE2_COHORT_UNIT_HOLD.json'),
        attachment_sha256=sha(ATTACHMENT), HC_unit='parent movie', HC_parent_movies=237,
        HC_one_clip_query_per_parent=237, Vid_videos=732, total_queries=969,
        original_hold_bytes_preserved=True, applied_to_new_suite_only=True, time=time.time()))
    return d

def lock():
    prepare()
    if not (BASE / 'RUNTIME_LOCK.json').exists():
        files = ['scripts/stvg_opd_paper_common_v1.py','scripts/run_stvg_opd_paper_v1.py',
            'scripts/continue_stvg_opd_paper_v1.py','scripts/test_stvg_opd_paper_v1.py',
            'protocols/stvg_opd_paper_v1.md','vg_tta/decota_spatial_opd_tunable_v1.py',
            'vg_tta/decota_spatial_opd_tunable_audit_v1.py','methods/decota_spatial_opd_v1/predictor.py',
            'methods/decota_spatial_opd_v1/configs.json','methods/CURRENT_METHOD.json',
            'scripts/run_decota_spatial_opd_v1.py','scripts/run_decota_paper_main_v1.py',
            'artifacts/decota_spatial_opd_v1/DESIGN_LOCK.json',
            'artifacts/decota_paper_baseline_scoring_v1/FINAL_COMPLETION.json',
            'artifacts/decota_paper_baseline_scoring_v1/EVALUATION_BARRIER.json']
        write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in files},
            design_sha256=sha(BASE/'DESIGN_LOCK.json'), attachment_sha256=sha(ATTACHMENT),
            original_paper_runtime_sha256=sha(PAPER/'RUNTIME_LOCK.json'),
            frozen_method_and_baselines=True, GT_used_by_GPU=False, time=time.time()))
    return verify()

def verify():
    verify_paper()
    p = read(BASE/'RUNTIME_LOCK.json')
    assert sha(BASE/'DESIGN_LOCK.json') == p['design_sha256']
    for f,h in p['pins'].items():
        assert sha(ROOT/f) == h, f
    return p

if __name__ == '__main__':
    lock()
    print('LOCKED', sha(BASE/'RUNTIME_LOCK.json'))
