"""Expanded fixed historical non-development cohort; original science retained."""
import copy
import hashlib
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_order_supplement_common_v1 import (
    read, write, status, sha, save, load, audit, scientific, activate_old,
    clone_state, initial_state, committed, observation_extent, cpu_contracts,
    OLD, PAPER, PYTHON, ARMS, QUERY, ALLOCATOR)
BASE = ROOT / 'artifacts/stvg_opd_claim_strengthening_v1/expanded_state'
PUB = ROOT / 'results/stvg_opd_claim_strengthening/2026-10-11/expanded_state'


def stages():
    return read(BASE / 'DESIGN_LOCK.json')['stages']


def old_record(stage, condition, order, at):
    assert condition == 'clean'
    q = stage['orders'][order][at]
    path = BASE / 'source_input_bindings' / stage['dataset'] / f'{q:05}.pt'
    receipt = read(path.with_suffix('.json'))
    assert sha(path) == receipt['sha256'] and not receipt['GT_read']
    return path, receipt


def verify():
    r = read(BASE / 'RUNTIME_LOCK.json')
    assert sha(BASE / 'DESIGN_LOCK.json') == r['design_sha256']
    assert sha(BASE / 'CPU_CONTRACTS.json') == r['CPU_contracts_sha256']
    for f, h in {**r['pins'], **r['protected']}.items():
        assert sha(ROOT / f) == h, f
    return r


def prepare():
    if (BASE / 'RUNTIME_LOCK.json').exists():
        return verify()
    assert read(OLD / 'FINAL_COMPLETION.json')['paper_suite_complete'] is True
    d = read(OLD / 'DESIGN_LOCK.json')
    definitions = {}; cohort = {}; bindings = {}
    for ds, expected in [('hc2', 205), ('vidstg', 700)]:
        old = d['datasets'][ds]; planpath = PAPER / ds / 'PLAN.json'
        assert sha(planpath) == old['PLAN_sha256']
        plan = read(planpath)
        dev = {plan['rows'][q]['source'] for q in old['development_query_ordinals']}
        assert len(dev) == 32
        seq = [q for q in old['balanced_order'] if plan['rows'][q]['source'] not in dev]
        assert len(seq) == len(set(seq)) == expected
        assert len({plan['rows'][q]['source'] for q in seq}) == expected
        cfg = old['config']
        orders = {'order1': seq, 'order2': sorted(seq, key=lambda q: hashlib.sha256(
            f'stvg-expanded-order|{ds}|2026101105|{q}'.encode()).hexdigest())}
        assert orders['order1'] != orders['order2']
        stage = dict(dataset=ds, source=old['source'], split='cross_clean',
            parents=seq, conditions=['clean'], orders=orders, arms=ARMS,
            variant_configs={a:dict(cfg, writeback=0. if a=='alpha0' else
                1. if a=='alpha1' else cfg['writeback']) for a in ARMS},
            expected_new_fits=expected*6, parent_count=expected,
            original_P1_stage='P1_'+ds, formerly_evaluated_not_fresh=True)
        definitions['E_'+ds+'_cross_clean'] = stage
        barrierpath = OLD / 'stages' / ('P1_'+ds) / 'PREDICTION_BARRIER.json'
        barrier = read(barrierpath)
        assert barrier['status']=='sealed' and not barrier['GT_read']
        direction = 'vidstg_to_hc2' if ds=='hc2' else 'hcstvg2_to_vidstg'
        for q in seq:
            rel = f'inputs/{direction}/clean/{q:05}.pt'; path = OLD / rel
            assert sha(path)==barrier['inputs'][rel]
            rc=read(path.with_suffix('.json')); inp=load(path)
            assert not rc['GT_read'] and not inp['GT_read']
            assert inp.get('query_ordinal',inp.get('parent'))==q
            target=BASE/'source_input_bindings'/ds/f'{q:05}.pt'
            z=dict(query_ordinal=q, config=cfg, interval=inp['interval'],
                   input=dict(path=rel,sha256=sha(path)), GT_read=False,
                   reused_evidence_only_not_old_fit_or_state=True)
            save(target,z)
            write(target.with_suffix('.json'),dict(sha256=sha(target),bytes=target.stat().st_size,
                  original_input_sha256=sha(path),GT_read=False,time=time.time()))
            bindings[str(target.relative_to(ROOT))]=sha(target)
        cohort[ds]=dict(parent_count=expected,excluded_development_parent_count=32,
            original_balanced_roster_count=len(old['balanced_order']),
            excluded_development_parent_sources=sorted(dev),query_ordinals=seq,
            PLAN_sha256=sha(planpath),original_P1_barrier_sha256=sha(barrierpath))
    design=dict(status='locked_before_new_GPU_and_GT',stages=definitions,cohorts=cohort,
        new_formal_fits=5430,qualification_arrivals=24,actual_qualification_fits=48,
        all_parent_sources=905,paired_parent_bootstrap=10000,bootstrap_seed=20261011,
        fixed_second_order_seed=2026101105,source_reset_per_arm_order=True,
        query_and_Adam_reset_per_query=True,GT_only_after_global_seal=True,
        parameters_selected=False,historical_exposure=True,GT_read=False,time=time.time())
    write(BASE/'DESIGN_LOCK.json',design)
    write(BASE/'CPU_CONTRACTS.json',cpu_contracts())
    files=['protocols/stvg_opd_claim_strengthening_v1.md',
        'scripts/stvg_opd_expanded_state_common_v1.py','scripts/run_stvg_opd_expanded_state_v1.py',
        'scripts/score_stvg_opd_expanded_state_v1.py','scripts/finalize_stvg_opd_expanded_state_v1.py',
        'scripts/continue_stvg_opd_claim_strengthening_v1.py',
        'scripts/stvg_opd_order_supplement_common_v1.py',
        'scripts/run_stvg_opd_order_supplement_v1.py',
        'scripts/score_stvg_opd_order_supplement_v1.py',
        'scripts/finalize_stvg_opd_design_ablation_v1.py']
    files+=['scripts/review_stvg_opd_supplement_v1.py']
    pins={**read(OLD/'P4_joint_engineering_revision001/REVISION_RUNTIME.json')['pins'],
          **{p:sha(ROOT/p) for p in files},**bindings}
    protect=['methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json',
             str((OLD/'FINAL_COMPLETION.json').relative_to(ROOT)),
             str((OLD/'DESIGN_LOCK.json').relative_to(ROOT)),
             'artifacts/stvg_opd_supplement_v1/orders/RUNTIME_LOCK.json',
             'artifacts/stvg_opd_supplement_v1/baselines/recovery/qualification_optimizer_isolation005/REVISION_RUNTIME.json']
    write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_new_GPU_and_GT',
        design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),
        pins=pins,protected={p:sha(ROOT/p) for p in protect},GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_serial_dependencies_and_actual_GPU_qualification',
        total=5430,GT_read=False,experiment_complete=False,time=time.time()))
    return verify()


if __name__=='__main__':
    prepare(); print('EXPANDED_STATE_LOCKED',sha(BASE/'RUNTIME_LOCK.json'),flush=True)
