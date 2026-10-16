"""Later-stage metadata and exact presealed whole-stream reuse; no target labels."""
from scripts.stvg_opd_paper_common_v1 import *


def phases():
    stages = {**read(BASE / 'LATER_DESIGN_LOCK.json')['stages'],
              **read(BASE / 'APPENDIX_STAGE_LOCK.json')['stages']}
    return {p: [n for n in stages if n.startswith(p + '_')]
            for p in ['P2', 'P3', 'P4', 'P5']}


def stage_definition(name):
    stages = read(BASE / 'LATER_DESIGN_LOCK.json')['stages']
    if name in stages:
        return stages[name]
    stage = read(BASE / 'APPENDIX_STAGE_LOCK.json')['stages'][name]
    assert stage['selected_before_P0_GT'] and not stage['replace_main_method']
    return {**stage, 'variant_configs': {'on_policy': stage['config']},
            'split': 'fixed_unified_appendix', 'reuse_identical_presealed_streams': False}


def verify_later():
    verify()
    old = read(BASE / 'COMPONENT_RUNTIME_LOCK.json')
    new = read(BASE / 'COMPONENT_RUNTIME_LOCK_revision001.json')
    assert sha(BASE / 'COMPONENT_RUNTIME_LOCK.json') == new['original_runtime_sha256']
    assert sha(BASE / 'LATER_DESIGN_LOCK.json') == old['design_sha256']
    assert sha(BASE / 'APPENDIX_STAGE_LOCK.json') == new['appendix_design_sha256']
    for f, h in {**old['pins'], **new['pins']}.items():
        assert sha(ROOT / f) == h, f
    return new


def record_key(condition, order, arm, at):
    return f'{condition}/{order}/{arm}/{at:05}'


def record_path(barrier, condition, order, arm, at):
    record = barrier['logical_records'][record_key(condition, order, arm, at)]
    assert record['path'] in barrier['files']
    assert record['sha256'] == barrier['files'][record['path']]
    return BASE / record['path'], record


def alias_stage(stage_name, stage, arm):
    """Only exact complete online streams are eligible, never clipped histories."""
    if not stage.get('reuse_identical_presealed_streams', False):
        return None
    if stage['observation_budget'] == 4 and stage['source'] == (
            'vidstg' if stage['dataset'] == 'hc2' else 'hcstvg2'):
        if stage['conditions'] == ['clean'] and arm in ARMS:
            return 'P0_' + stage['dataset']
    if stage_name.startswith('P3_') and arm == 'on_policy':
        return stage_name.replace('P3_', 'P2_', 1)
    return None


def identical_stream(source, target, arm, source_cfg):
    assert source['dataset'] == target['dataset']
    assert source['source'] == target['source']
    assert source['observation_budget'] == target['observation_budget']
    assert source['conditions'] == target['conditions']
    assert source['orders'] == target['orders'], 'Subset histories are not online stream reuse'
    assert arm in source['arms'] and source_cfg == target['variant_configs'][arm]


def build_alias(stage_name, stage, arm):
    origin = alias_stage(stage_name, stage, arm)
    if origin is None:
        return None
    design = read(BASE / 'DESIGN_LOCK.json')
    source = (design['stages'][origin] if origin.startswith('P0_') else
              read(BASE / 'LATER_DESIGN_LOCK.json')['stages'][origin])
    cfg = (design['datasets'][stage['dataset']]['config'] if origin.startswith('P0_')
           else source['variant_configs'][arm])
    identical_stream(source, stage, arm, cfg)
    path = BASE / 'stages' / origin / 'PREDICTION_BARRIER.json'
    barrier = read(path)
    assert barrier['status'] == 'sealed' and not barrier['GT_read']
    result = dict(origin_stage=origin, origin_barrier_sha256=sha(path), records={})
    for condition in stage['conditions']:
        for order, seq in stage['orders'].items():
            previous = None
            for at, query in enumerate(seq):
                if origin.startswith('P0_'):
                    f = BASE / 'stages' / origin / condition / order / arm / f'{at:05}.pt'
                    assert str(f.relative_to(BASE)) in barrier['files']
                else:
                    f, _ = record_path(barrier, condition, order, arm, at)
                receipt = read(f.with_suffix('.json'))
                h = sha(f)
                assert h == receipt['sha256'] == barrier['files'][str(f.relative_to(BASE))]
                assert receipt['bytes'] == f.stat().st_size and not receipt['GT_read']
                assert receipt['time'] <= barrier['time']
                z = load(f)
                assert z['dataset'] == stage['dataset'] and z['source'] == stage['source']
                assert z['query_ordinal'] == query and z['arrival'] == at and z['order'] == order
                assert z['arm'] == arm and z['condition'] == condition and z['config'] == cfg
                assert z['previous_payload_sha256'] == previous and not z['GT_read']
                assert z['input']['sha256'] == sha(BASE / z['input']['path'])
                assert z['fit']['selected_step'] == cfg['steps']
                previous = h
                result['records'][record_key(condition, order, arm, at)] = dict(
                    path=str(f.relative_to(BASE)), sha256=h, query_ordinal=query,
                    origin_stage=origin, reused_complete_identical_stream=True,
                    prior_payload_sha256=z['previous_payload_sha256'],
                    input=z['input'])
    assert len(result['records']) == sum(map(len, stage['orders'].values())) * len(stage['conditions'])
    return result


def input_equivalent(left, right):
    import torch
    for field in ['pixel_sha256', 'source_model_state_sha256', 'interval', 'frame_ids']:
        assert left[field] == right[field], field
    assert torch.equal(left['native_boxes'], right['native_boxes'])
    # The expert tensors must be identical too; a native-only match is insufficient.
    assert digest(left['expert']) == digest(right['expert']), 'Different admitted specialist evidence'


def authorize_after_root(stage, previous):
    receipt = read(BASE / f'{previous}_ROOT_CLOSING_RECEIPT.json')
    assert receipt['status'] == 'complete' and not receipt['paper_suite_complete']
    git_receipt = read(BASE / f'{previous}_FINAL_GITHUB_RECEIPT.json')
    assert git_receipt['status'] == 'pass'
    if stage == 'P2' and not (BASE / 'P1_STAGE_AUTHORIZATION.json').exists():
        write(BASE / 'P1_STAGE_AUTHORIZATION.json', dict(
            status='completed_before_later_stages', P1_root_receipt_sha256=sha(
                BASE / 'P1_ROOT_CLOSING_RECEIPT.json'),
            P1_public_receipt_sha256=sha(BASE / 'P1_FINAL_GITHUB_RECEIPT.json'),
            no_configuration_selection=True, time=time.time()))
    path = BASE / f'{stage}_STAGE_AUTHORIZATION.json'
    if path.exists():
        old = read(path)
        assert old['status'] == 'actual_previous_stage_root_closed' and old['previous'] == previous
        assert old['previous_root_sha256'] == sha(BASE / f'{previous}_ROOT_CLOSING_RECEIPT.json')
        assert old['fixed_runtime_sha256'] == sha(BASE / 'COMPONENT_RUNTIME_LOCK_revision001.json')
        return
    write(path, dict(
        status='actual_previous_stage_root_closed', previous=previous,
        previous_root_sha256=sha(BASE / f'{previous}_ROOT_CLOSING_RECEIPT.json'),
        fixed_runtime_sha256=sha(BASE / 'COMPONENT_RUNTIME_LOCK_revision001.json'),
        time=time.time()))
