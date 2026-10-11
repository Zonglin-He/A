"""Separate, human-authorized component interventions; old suite is immutable."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, save, load

BASE = ROOT / 'artifacts/stvg_opd_design_ablation_v1'
PUB = ROOT / 'results/stvg_opd_design_ablation/2026-10-11'
OLD = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2'
OLDPUB = ROOT / 'results/stvg_opd_paper_hc2_revision/2026-10-08'
PAPER = ROOT / 'artifacts/decota_paper_experiments_v1'
PYTHON = ROOT / '.conda/tubedetr/bin/python'
ARMS = ['whole_video', 'carry_query', 'full_LN_carry']
QUERY = 'spatial.query_residual'
ALLOCATOR = 'expandable_segments:True'
AUTHORIZATION_ATTACHMENT_SHA256 = ['3d10fe3a1c383747244d1e4fba4feec76572d03b27439ab4fed0c0bde3606330', 'f5f09a8fda1f1191ab8036a93212f506d1f14f6f5a33e4344e10cdf72dec698a']


def clone_state(state):
    return {n: v.detach().cpu().clone() for n, v in state.items()}


def initial_state(source, previous, arm):
    import torch
    if arm not in ARMS:
        raise ValueError(arm)
    if previous is None:
        out = clone_state(source)
        assert torch.count_nonzero(out[QUERY]) == 0
        return out
    if set(source) != set(previous):
        raise ValueError('Different full state schema')
    if any(source[n].shape != previous[n].shape for n in source):
        raise ValueError('Different state tensor shapes')
    out = clone_state(previous)
    if arm != 'carry_query':
        out[QUERY] = torch.zeros_like(out[QUERY])
    return out


def committed(initial, final, alpha, arm):
    import torch
    if arm not in ARMS or isinstance(alpha, bool) or not 0 <= alpha <= 1:
        raise ValueError('Invalid lifecycle')
    if set(initial) != set(final):
        raise ValueError('Different full state schema')
    if (arm == 'full_LN_carry') != (alpha == 1):
        raise ValueError('Alpha must implement the declared single intervention')
    result = {}
    for n, value in initial.items():
        if value.shape != final[n].shape:
            raise ValueError('Different state tensor shapes')
        if n == QUERY:
            result[n] = final[n].detach().cpu().clone() if arm == 'carry_query' else torch.zeros_like(value.cpu())
        elif alpha == 1:
            result[n] = final[n].detach().cpu().clone()
        else:
            result[n] = value.cpu() + (final[n].cpu() - value.cpu()) * alpha
    return result


def observation_extent(ids, native, arm):
    if arm not in ARMS or not ids or len(native) != 2:
        raise ValueError('Invalid observation scope')
    if any(isinstance(x, bool) or int(x) != x for x in native):
        raise ValueError('Invalid native extent')
    if not 0 <= native[0] <= native[1] < len(ids):
        raise ValueError('Invalid native extent')
    return [0, len(ids) - 1] if arm == 'whole_video' else list(native)


def stages():
    return read(BASE / 'DESIGN_LOCK.json')['stages']


def audit(fit, expert):
    from vg_tta.stvg_opd_p4_joint_precision001 import audit as qualified
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit as original
    return qualified(fit, expert, original)


def scientific(fit):
    from vg_tta.stvg_opd_p4_joint_precision001 import FIELD, ACTION
    return {k: v for k, v in fit.items() if k not in [FIELD, ACTION, 'numerical_chart']}


def verify():
    runtime = read(BASE / 'RUNTIME_LOCK.json')
    assert sha(BASE / 'DESIGN_LOCK.json') == runtime['design_sha256']
    assert sha(BASE / 'CPU_CONTRACTS.json') == runtime['CPU_contracts_sha256']
    for name, digest in {**runtime['pins'], **runtime['protected']}.items():
        assert sha(ROOT / name) == digest, name
    assert read(OLD / 'FINAL_COMPLETION.json')['paper_suite_complete'] is True
    return runtime


def activate_old():
    from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate
    activate()
    from scripts.decota_opd_tuning_common_v1 import bridge
    bridge()


def old_record(stage, condition, order, at):
    barrier = read(OLD / 'stages' / stage['matched_full_stage'] / 'PREDICTION_BARRIER.json')
    key = f'{condition}/{order}/on_policy/{at:05}'
    binding = barrier['logical_records'][key]
    path = OLD / binding['path']
    assert sha(path) == binding['sha256'] == barrier['files'][binding['path']]
    receipt = read(path.with_suffix('.json'))
    assert receipt['sha256'] == binding['sha256'] and not receipt['GT_read']
    assert receipt['bytes'] == path.stat().st_size
    return path, binding


def cpu_contracts():
    import torch
    from methods.decota_final_simplified_v1.observations import uniform_positions
    source = {QUERY: torch.zeros(256), 'LN': torch.zeros(1536)}
    final = {QUERY: torch.full((256,), .75), 'LN': torch.linspace(-.02, .03, 1536)}
    positive = []; rejected = []
    def good(name, check):
        assert check; positive.append(name)
    for arm in ARMS:
        alpha = 1. if arm == 'full_LN_carry' else 1/16
        first = initial_state(source, None, arm)
        carry = committed(first, final, alpha, arm)
        second = initial_state(source, carry, arm)
        good(arm + '_full1792', sum(x.numel() for x in second.values()) == 1792)
        good(arm + '_query_lifetime', torch.equal(second[QUERY], final[QUERY] if arm == 'carry_query' else source[QUERY]))
        good(arm + '_LN_lifetime', torch.equal(second['LN'], final['LN'] if alpha == 1 else final['LN'] * alpha))
        good(arm + '_no_source_mutation', torch.count_nonzero(source['LN']).item() == 0)
    ids = [0, 3, 7, 11, 16, 22, 29, 35, 42, 48]
    good('whole_video_physical_uniform4', uniform_positions(ids, observation_extent(ids, [3, 6], 'whole_video'), 4) == uniform_positions(ids, [0, 9], 4))
    good('state_arms_original_scope', all(observation_extent(ids, [3, 6], a) == [3, 6] for a in ARMS[1:]))
    good('whole_video_original_grid', len(uniform_positions(ids, [0, 9], 4)) == 4)
    good('singleton_original_sampling_rule', uniform_positions([5], [0, 0], 4) == [0])
    bad = [('unknown_arm', lambda: initial_state(source, None, 'other')),
           ('missing_state', lambda: initial_state(source, {'LN': source['LN']}, 'carry_query')),
           ('wrong_shape', lambda: initial_state(source, {QUERY: torch.zeros(1), 'LN': source['LN']}, 'carry_query'))]
    for a in [-.1, 1.1, True]:
        bad.append(('invalid_alpha_' + str(a), lambda a=a: committed(source, final, a, 'carry_query')))
    for arm in ARMS:
        bad.append(('wrong_declared_alpha_' + arm, lambda arm=arm: committed(source, final, .5 if arm == 'full_LN_carry' else 1., arm)))
    for extent in [[-1, 2], [0, 10], [4, 2], [True, 2], [0]]:
        bad.append(('invalid_extent_' + str(extent), lambda extent=extent: observation_extent(ids, extent, 'whole_video')))
    for name, fn in bad:
        try: fn()
        except (ValueError, AssertionError): rejected.append(name)
        else: raise AssertionError('Invalid contract was accepted: ' + name)
    return dict(status='pass', scope='synthetic sampling/lifetime/single-factor contracts; not GPU qualification',
                valid_contracts=len(positive), invalid_contracts_rejected=len(rejected), valid=positive,
                rejected=rejected, GT_read=False, actual_GPU_fits=0, time=time.time())


def prepare():
    if (BASE / 'RUNTIME_LOCK.json').exists():
        return verify()
    assert read(OLD / 'FINAL_COMPLETION.json')['paper_suite_complete'] is True
    assert read(OLD / 'PAPER_SUITE_FINAL_PUBLICATION_RECEIPT.json')['status'] == 'pass'
    original = read(OLD / 'LATER_DESIGN_LOCK.json')['stages']
    design_stages = {}
    for name, definition in original.items():
        if not name.startswith('P3_'): continue
        stage = copy.deepcopy(definition)
        stage['matched_full_stage'] = name
        stage['arms'] = ARMS
        stage['variant_configs'] = {a: dict(definition['variant_configs']['on_policy'],
            writeback=1. if a == 'full_LN_carry' else definition['variant_configs']['on_policy']['writeback']) for a in ARMS}
        stage['reuse_identical_presealed_streams'] = False
        stage['expected_new_fits'] = len(stage['conditions']) * sum(map(len, stage['orders'].values())) * len(ARMS)
        design_stages[name.replace('P3_', 'A_')] = stage
    assert sum(s['expected_new_fits'] for s in design_stages.values()) == 5376
    design = dict(version='stvg_opd_design_ablation_v1', human_request_date='2026-10-11',
        authorization_attachments_sha256=AUTHORIZATION_ATTACHMENT_SHA256, stages=design_stages,
        three_single_factor_interventions=ARMS, new_formal_fits=5376,
        qualification_arrivals=72, actual_qualification_fits=144, historically_exposed_parents=True,
        coverage_bins=[0., .25, .5, .75, 1.], coverage_last_bin_inclusive=True,
        coverage_GT_support='[minimum annotated frame, maximum annotated frame+1)',
        paired_parent_bootstrap=10000, bootstrap_seed=20261011,
        GT_only_after_all_new_predictions_global_seal=True, source_reset_per_arm_condition_order=True,
        Adam_reset_each_query=True, final_round_output=True, no_new_parameter_selection=True,
        second_backbone_scope='pending_human_optional_preference', production_promotion=False, time=time.time())
    write(BASE / 'DESIGN_LOCK.json', design)
    contracts = cpu_contracts(); write(BASE / 'CPU_CONTRACTS.json', contracts)
    pins = ['protocols/stvg_opd_design_ablation_v1.md',
            'scripts/stvg_opd_design_ablation_common_v1.py',
            'scripts/run_stvg_opd_design_ablation_v1.py',
            'scripts/continue_stvg_opd_design_ablation_v1.py',
            'scripts/score_stvg_opd_design_ablation_v1.py',
            'scripts/finalize_stvg_opd_design_ablation_v1.py',
            'vg_tta/stvg_opd_p4_joint_precision001.py',
            'vg_tta/stvg_opd_paper_component_audit_v1.py',
            'vg_tta/decota_spatial_opd_tunable_v1.py',
            'scripts/stvg_opd_paper_inputs_v1.py',
            'scripts/stvg_opd_legacy_input_bridge003.py',
            'scripts/run_decota_paper_main_v1.py',
            'methods/decota_final_simplified_v1/observations.py',
            'scripts/score_stvg_opd_p1_v1.py',
            'vg_tta/tastvg_oracle_event5_v1.py']
    numeric = read(OLD / 'P4_joint_engineering_revision001/REVISION_RUNTIME.json')
    allpins = {**numeric['pins'], **{p: sha(ROOT / p) for p in pins}}
    protect = ['methods/CURRENT_METHOD.json', 'methods/decota_spatial_opd_v1/configs.json',
        str((OLD / 'FINAL_COMPLETION.json').relative_to(ROOT)),
        str((OLD / 'PAPER_SUITE_ROOT_CLOSING_RECEIPT.json').relative_to(ROOT)),
        str((OLD / 'LATER_DESIGN_LOCK.json').relative_to(ROOT)),
        str((OLD / 'P4_joint_engineering_revision001/REVISION_RUNTIME.json').relative_to(ROOT))]
    for s in design_stages.values():
        protect.append(str((OLD / 'stages' / s['matched_full_stage'] / 'PREDICTION_BARRIER.json').relative_to(ROOT)))
        for phase in ['P2', 'P3']:
            name = s['matched_full_stage'].replace('P3_', phase + '_')
            protect.append(str((OLDPUB / name / 'ROWS.jsonl.gz').relative_to(ROOT)))
    write(BASE / 'RUNTIME_LOCK.json', dict(status='locked_before_new_GPU_and_GT',
        pins=allpins, protected={p: sha(ROOT / p) for p in protect}, design_sha256=sha(BASE / 'DESIGN_LOCK.json'),
        CPU_contracts_sha256=sha(BASE / 'CPU_CONTRACTS.json'),
        reused_qualified_numeric_runtime_sha256=sha(OLD / 'P4_joint_engineering_revision001/REVISION_RUNTIME.json'),
        scientific_changes_only=ARMS, GT_read=False, time=time.time()))
    status(BASE / 'STATUS.json', dict(status='prepared_pending_actual_GPU_qualification',
        new_formal_fits_done=0, new_formal_fits_total=5376, GT_read=False, time=time.time()))
    return verify()


if __name__ == '__main__':
    prepare(); print('DESIGN_ABLATION_LOCKED', sha(BASE / 'RUNTIME_LOCK.json'), flush=True)
