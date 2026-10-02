"""Lock the small matched audit without reading labels or outcome scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_ur_decomposition_common_v1 import *


def run():
    import numpy as np
    from methods.decota_final_simplified_v1.tensors import state_hash
    assert not BASE.exists(), 'new namespace only'
    assert read(OLD / 'FINAL_COMPLETION.json')['status'] == 'completed'
    plan = read(OLD / 'hc2/PLAN.json')
    context = load(CONTEXT)
    assert not context['GT_read']
    np.testing.assert_allclose(context['vectors'].numpy() @ context['vectors'].numpy().T,
                               context['cosine'], rtol=0, atol=1e-14)
    parents = context['parents']
    inputs = {}
    def bind(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for path in [OLD / 'FINAL_COMPLETION.json', OLD / 'hc2/PLAN.json', CONTEXT,
                 CONTEXT.with_suffix('.json'), POOL / 'hc2/CAPTURE_BARRIER.json',
                 ROOT / 'methods/CURRENT_METHOD.json']:
        bind(path)
    sources = {s: i for i, s in enumerate(sorted({plan['rows'][p]['source'] for p in parents}))}
    rows = []
    for condition in plan['conditions']:
        for order, sequence in plan['splits']['search']['orders'].items():
            assert sorted(sequence) == parents
            for arrival in range(0, 32, 4):
                path = OLD / 'hc2/R/online' / condition / order / f'{arrival:05}.pt'
                old = load(path)
                assert sha(path) == read(path.with_suffix('.json'))['sha256']
                assert old['expert_scheduled'] and old['parent'] == sequence[arrival]
                assert state_hash(old['pre_state']) == old['pre_sha']
                bind(path); bind(path.with_suffix('.json'))
                parent = sequence[arrival]
                ur = POOL / 'hc2/experts/spatial' / condition / f'{parent:05}.json'
                uniform = read(ur); uf = POOL / 'hc2/experts' / uniform['cache']
                routed = old['routed_expert']; rf = ROOT / routed['cache']
                assert sha(uf) == uniform['cache_sha256'] and sha(rf) == routed['cache_sha256']
                assert uniform['pixel_sha256'] == routed['pixel_sha256'] == old['pixel_sha256']
                bind(ur); bind(uf); bind(rf)
                future = [i for i in range(arrival+1, 32) if i % 4 != 0]
                similarities = {i: float(context['cosine'][parents.index(parent),
                                  parents.index(sequence[i])]) for i in future}
                roles = dict(self=arrival, next=future[0],
                             near=min(future, key=lambda i: (-similarities[i], i)),
                             far=min(future, key=lambda i: (similarities[i], i)))
                targets = [dict(arrival=i, parent=sequence[i], source_id=sources[plan['rows'][sequence[i]]['source']],
                               cosine=1. if i == arrival else similarities[i],
                               roles=[role for role, j in roles.items() if j == i],
                               broader=i in future) for i in [arrival, *future]]
                rows.append(dict(condition=condition, order=order, donor_arrival=arrival,
                    donor_parent=parent, source_id=sources[plan['rows'][parent]['source']],
                    donor_payload=str(path.relative_to(ROOT)), common_pre_sha=old['pre_sha'],
                    uniform_receipt=str(ur.relative_to(ROOT)), uniform_cache=str(uf.relative_to(ROOT)),
                    routed_cache=str(rf.relative_to(ROOT)), routed_positions=routed['positions'],
                    pixel_sha256=old['pixel_sha256'], targets=targets, roles=roles))
    assert len(rows) == 96 and sum(len(r['targets']) for r in rows) == 1392
    assert sum(sum(t['broader'] for t in r['targets']) for r in rows) == 1296
    cohort = dict(rows=rows, sources=32, donor_sources=len({r['source_id'] for r in rows}),
                  orders=plan['splits']['search']['orders'], conditions=plan['conditions'],
                  target_context='frozen text-only RoBERTa; eligible future nonexpert',
                  same_state_base='saved R pre-arrival', GT_read=False, time=time.time())
    write(BASE / 'COHORT.json', cohort); bind(BASE / 'COHORT.json')
    config = dict(dataset='HC-STVG-v2', development_sources=32, queries=32,
                  historically_exposed=True, params={**plan['params'], 'steps': 1},
                  historical_steps=8, common_state='saved R pre-arrival',
                  gradient_specific='elementwise float32(g_R-g_U)',
                  probes='source-fixed offsets, added to common current state',
                  primary_interval='common pre-state native interval of each target',
                  logical_role_cells=384, broader_target_cells=1296,
                  unique_donor_target_cells=1392, donors=96,
                  checkpoint_state_sha256=read(POOL / 'hc2/CAPTURE_BARRIER.json')['checkpoint_state_sha256'],
                  bootstrap_resamples=10000, seed=20261001, new_experts=0,
                  new_backbone_forwards=0, no_full_online_method=True)
    write(BASE / 'CONFIG.json', config); bind(BASE / 'CONFIG.json')
    names = ['scripts/tastvg_ur_decomposition_common_v1.py',
             'scripts/prepare_tastvg_ur_decomposition_v1.py',
             'scripts/run_tastvg_ur_decomposition_v1.py',
             'vg_tta/tastvg_ur_write_decomposition_v1.py',
             'protocols/tastvg_ur_write_decomposition_v1.md',
             'vg_tta/tastvg_spatial_online_opd_s1_v1.py',
             'vg_tta/tastvg_selected_rollout_v1.py',
             'vg_tta/tastvg_spatial_rank_s11_v1.py',
             'vg_tta/tastvg_spatial_critic_s06_v1.py',
             'vg_tta/tastvg_reference_selection_v1.py',
             'vg_tta/tastvg_native_spatial_rollout_s05_v1.py',
             'vg_tta/tastvg_causal_round2_v1.py',
             'scripts/run_tastvg_evidence_vulnerability_v2.py',
             'scripts/run_spatial_regression_alignment_v1.py',
             'methods/decota_final_simplified_v1/backbone.py',
             'methods/decota_final_simplified_v1/objectives.py']
    write(BASE / 'RUNTIME_LOCK.json', dict(pins={name: sha(ROOT / name) for name in names},
          inputs=inputs, new_experts=0, new_backbone_forwards=0, GT_read=False, time=time.time()))
    status(BASE / 'STATUS.json', dict(status='ready_for_no_GT_smoke', done=0, total=96, time=time.time()))
    archive('96个同状态donor与1392个self/future目标已CPU锁定，零新预测零GT；即将无GT资格复现')
    print('LOCKED', len(rows), 'donors', 1392, 'unique targets', cohort['donor_sources'], 'donor sources')


if __name__ == '__main__':
    run()
