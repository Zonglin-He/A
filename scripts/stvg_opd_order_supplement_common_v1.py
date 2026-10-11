"""Separate, fixed-scope stream-order evidence; existing results are immutable."""
import copy
import hashlib
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, save, load
from scripts.stvg_opd_design_ablation_common_v1 import audit, scientific, activate_old, clone_state
BASE = ROOT / 'artifacts/stvg_opd_supplement_v1/orders'
PUB = ROOT / 'results/stvg_opd_supplement/2026-10-11/orders'
OLD = ROOT / 'artifacts/stvg_opd_paper_hc2_revision_v2'
PAPER = ROOT / 'artifacts/decota_paper_experiments_v1'
PYTHON = ROOT / '.conda/tubedetr/bin/python'
ARMS = ['full', 'alpha0', 'alpha1']
QUERY = 'spatial.query_residual'
ALLOCATOR = 'expandable_segments:True'
SEEDS = [2026101101, 2026101102, 2026101103]

def initial_state(source, previous, arm):
    import torch
    if arm not in ARMS: raise ValueError('Unknown state arm')
    if previous is not None and (set(source) != set(previous) or any(source[n].shape != previous[n].shape for n in source)):
        raise ValueError('Invalid full-state schema')
    out = clone_state(source if previous is None else previous)
    out[QUERY] = torch.zeros_like(out[QUERY])
    return out

def committed(initial, final, alpha, arm):
    import math
    import torch
    if arm not in ARMS or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError('Invalid arm/alpha')
    if not 0 <= alpha <= 1 or (arm == 'alpha0' and alpha != 0) or (arm == 'alpha1' and alpha != 1) or (arm == 'full' and not 0 < alpha < 1):
        raise ValueError('Declared coefficient mismatch')
    if set(initial) != set(final) or any(initial[n].shape != final[n].shape for n in initial):
        raise ValueError('Invalid state schema')
    return {n: torch.zeros_like(v.cpu()) if n == QUERY else (final[n].cpu().clone() if alpha == 1 else v.cpu() + (final[n].cpu()-v.cpu())*alpha) for n,v in initial.items()}

def observation_extent(ids, native, arm):
    if arm not in ARMS or not ids or len(native) != 2 or any(isinstance(x, bool) or int(x) != x for x in native) or not 0 <= native[0] <= native[1] < len(ids):
        raise ValueError('Invalid native observation')
    return list(native)

def stages(): return read(BASE / 'DESIGN_LOCK.json')['stages']

def old_record(stage, condition, order, at):
    q = stage['orders'][order][at]
    original_at = stage['input_order'].index(q)
    b = read(OLD / 'stages' / stage['matched_full_stage'] / 'PREDICTION_BARRIER.json')
    z = b['logical_records'][f'{condition}/order1/on_policy/{original_at:05}']
    p = OLD / z['path']; rc = read(p.with_suffix('.json'))
    assert sha(p) == z['sha256'] == rc['sha256'] and rc['bytes'] == p.stat().st_size and not rc['GT_read']
    return p, z

def verify():
    r = read(BASE / 'RUNTIME_LOCK.json')
    assert sha(BASE / 'DESIGN_LOCK.json') == r['design_sha256']
    assert sha(BASE / 'CPU_CONTRACTS.json') == r['CPU_contracts_sha256']
    for p,h in {**r['pins'], **r['protected']}.items(): assert sha(ROOT/p) == h, p
    return r

def cpu_contracts():
    import torch
    src = {QUERY: torch.zeros(256), 'LN': torch.linspace(-1,1,1536)}
    fin = {QUERY: torch.ones(256), 'LN': src['LN']+.25}
    good = []; bad = []
    for arm,a in [('full',1/16),('alpha0',0.),('alpha1',1.)]:
        c = committed(src,fin,a,arm); s = initial_state(src,c,arm)
        assert torch.count_nonzero(s[QUERY]) == 0 and sum(v.numel() for v in s.values()) == 1792
        assert torch.equal(s['LN'],src['LN']+(fin['LN']-src['LN'])*a)
        good += [arm+'_query_reset',arm+'_1792',arm+'_LN_formula']
    for arm,a in [('alpha0',.1),('alpha1',.9),('full',0),('full',1),('full',True),('full',float('nan')),('full',-1),('unknown',.1)]:
        try: committed(src,fin,a,arm)
        except ValueError: bad.append(str((arm,a)))
        else: raise AssertionError('Invalid intervention accepted')
    try: initial_state(src,{QUERY:torch.zeros(1),'LN':src['LN']},'full')
    except ValueError: bad.append('wrong_shape')
    else: raise AssertionError('Invalid shape accepted')
    return dict(status='pass',scope='synthetic state/declared-intervention contracts; not GPU qualification',valid_contracts=len(good),invalid_contracts_rejected=len(bad),valid=good,rejected=bad,GT_read=False,time=time.time())

def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists(): return verify()
    assert read(OLD/'FINAL_COMPLETION.json')['paper_suite_complete'] is True
    assert read(ROOT/'artifacts/stvg_opd_design_ablation_v1/FINAL_COMPLETION.json')['status'] == 'complete'
    original=read(OLD/'LATER_DESIGN_LOCK.json')['stages']; definitions={}
    for name in ['P3_hc2_cross_clean','P3_vidstg_cross_clean']:
        old=original[name]; s=copy.deepcopy(old); s['matched_full_stage']=name; s['input_order']=old['orders']['order1']
        s['orders']={f'order{i+3}':sorted(s['input_order'],key=lambda q:hashlib.sha256(f'stvg-order|{s["dataset"]}|{seed}|{q}'.encode()).hexdigest()) for i,seed in enumerate(SEEDS)}
        lists=list(s['orders'].values())+list(old['orders'].values())
        assert len({tuple(x) for x in lists})==5 and all(len(x)==128 and set(x)==set(s['input_order']) for x in lists)
        s['arms']=ARMS; cfg=old['variant_configs']['on_policy']
        s['variant_configs']={a:dict(cfg,writeback=0. if a=='alpha0' else 1. if a=='alpha1' else cfg['writeback']) for a in ARMS}
        s['expected_new_fits']=1152; definitions[name.replace('P3_','O_')]=s
    d=dict(version='stvg_opd_order_supplement_v1',stages=definitions,order_seeds=SEEDS,new_formal_fits=2304,qualification_arrivals=36,actual_qualification_fits=72,
        bootstrap_seed=20261011,paired_parent_bootstrap=10000,progress_bin_size=16,historically_exposed_parents=True,
        source_reset_per_arm_order=True,query_and_Adam_reset_per_query=True,GT_only_after_global_seal=True,no_parameter_selection=True,
        input_reuse_only_not_old_state_trajectory=True,production_promotion=False,time=time.time())
    write(BASE/'DESIGN_LOCK.json',d); write(BASE/'CPU_CONTRACTS.json',cpu_contracts())
    files=['protocols/stvg_opd_supplement_v1.md','scripts/stvg_opd_order_supplement_common_v1.py','scripts/run_stvg_opd_order_supplement_v1.py','scripts/continue_stvg_opd_order_supplement_v1.py','scripts/score_stvg_opd_order_supplement_v1.py','scripts/finalize_stvg_opd_order_supplement_v1.py','scripts/stvg_opd_design_ablation_common_v1.py','scripts/stvg_opd_legacy_input_bridge003.py','scripts/run_decota_paper_main_v1.py','scripts/stvg_opd_paper_inputs_v1.py','scripts/score_stvg_opd_p1_v1.py','vg_tta/tastvg_oracle_event5_v1.py']
    pins={**read(OLD/'P4_joint_engineering_revision001/REVISION_RUNTIME.json')['pins'],**{p:sha(ROOT/p) for p in files}}
    protect=['methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json',str((OLD/'FINAL_COMPLETION.json').relative_to(ROOT)),str((OLD/'LATER_DESIGN_LOCK.json').relative_to(ROOT)),
        'artifacts/stvg_opd_design_ablation_v1/FINAL_COMPLETION.json']+[str((OLD/'stages'/s['matched_full_stage']/'PREDICTION_BARRIER.json').relative_to(ROOT)) for s in definitions.values()]
    write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_new_GPU_and_GT',design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),pins=pins,protected={p:sha(ROOT/p) for p in protect},GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_actual_qualification',total=2304,GT_read=False,time=time.time()))
    return verify()

if __name__=='__main__': prepare(); print('ORDER_SUPPLEMENT_LOCKED',sha(BASE/'RUNTIME_LOCK.json'),flush=True)
