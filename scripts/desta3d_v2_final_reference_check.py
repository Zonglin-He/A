"""Read-only shared-reference confirmation at the source-selected fixed B1 state.

Reuse the accepted cached-v3 execution, without mutating its source or old runs.
Source-development selection is disclosed; this is not a new training arm.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scripts import desta3d_v2_reference_audit_cached_v3 as engine
from scripts.score_desta3d_v2_aux_recovery import state_sha, save_once

BASE = ROOT / 'artifacts/desta3d_v2/aux_backflow_v1/final_B1_reference_v1'
FIT = BASE.parent / 'fit_recovery_v2'
RUN_ID = 'B1_final001'
ADAPTER_SHA = '4a2ef2cf87e1753fad7c0c5c1f582f499c14b0b2945a839680b296abb0d9c7eb'


def register():
    assert not BASE.exists(), 'write-once registration'
    report = engine.read(FIT / 'independent_readback_v1/REPORT.json')
    audit = engine.read(FIT / 'ROOT_SCORE_CROSSCHECK.json')
    assert audit['status'] == 'passed'
    assert audit['report_sha'] == engine.sha(FIT / 'independent_readback_v1/REPORT.json')
    means = {a: audit['arms'][a]['parent_macro']['vIoU'] for a in ['B0', 'B1', 'B2', 'common', 'Frozen']}
    assert means['B1'] >= means['Frozen'] and means['B1'] == max(means[a] for a in ['B0', 'B1', 'B2'])
    final = FIT / 'FINAL_B.pt'
    assert engine.sha(final) == 'db9fb8900332bc73325c9627c1001dc3fa61bd9c9842aa2ec076baa0ce1cf35f'
    state = torch.load(final, map_location='cpu', weights_only=False)['B1']
    assert state_sha(state) == ADAPTER_SHA
    from vg_tta.desta3d_v2 import Desta3DAdapterV2
    adapter = Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False)
    adapter.load_state_dict(state, strict=True)
    assert engine.adapter_sha256(adapter) == ADAPTER_SHA
    assert all(torch.isfinite(t).all() for t in state.values())
    pilot = ROOT / 'artifacts/desta3d_v2/shared_reference_v1/pilot003'
    assert engine.read(pilot / 'COMPLETE.json')['interface_pass']
    rows = sorted([r for r in engine.read(engine.PARENT / 'source_fit/INPUTS.json') if r['split'] == 'validation'], key=lambda r: r['key'])
    assert len(rows) == 198 and len({r['source'] for r in rows}) == 31
    BASE.mkdir(parents=True)
    ck = BASE / 'B1_FIXED_FINAL.pt'
    torch.save({'adapter': state}, ck)
    selection = {
        'selected_arm': 'B1', 'optimizer_steps': 155, 'training_queries': 618, 'training_parents': 95,
        'criterion': 'highest fixed-B endpoint source-validation parent-macro vIoU; literal >= Frozen resource gate',
        'selection_data': '198 source-development queries / 31 Vid parents; already exposed and not independent confirmation',
        'parent_macro_vIoU': means, 'common_A_has_higher_mean': means['common'] > means['B1'],
        'candidate_scope': 'trained B integration candidate only, not best among every state or production promotion',
        'limitations': 'B1-Frozen CI crosses zero; native-v-good retention 65/68 versus B0/B2 67/68; no demonstrated stable positive TTA gain',
        'target_GT_read': False, 'final_checkpoint': str(final), 'final_sha': engine.sha(final),
        'adapter_sha256': ADAPTER_SHA, 'time': time.time(),
    }
    save_once(BASE / 'SOURCE_CANDIDATE_SELECTION.json', selection)
    dest = BASE / RUN_ID
    config = {
        'run_id': RUN_ID, 'scope': 'full', 'seed': 20260927, 'queries': 198, 'parents': 31,
        'checkpoint': {'checkpoint': str(ck), 'sha256': engine.sha(ck), 'adapter_sha256': ADAPTER_SHA},
        'arms': ['independent_reference', 'shared_reference_time'],
        'execution': 'unchanged cached-v3 accepted official staged prefix; fresh spatial KV; no event KV reuse',
        'event_pair': 'deterministic shared event output, identical pixels and fixed adapter',
        'residual_control': 'same event reference/time, spatial residual vs zero, actual coordinate logits',
        'source_training_updates': 0, 'optimizer_steps': 0, 'gradient_probe': 'not run; scope full',
        'source_validation_previously_exposed': True, 'source_val_GT_read_by_runner': False, 'target_GT_read': False,
        'state_selection_by_this_run': False, 'phase_seconds': 1800, 'cumulative_cap_seconds': None,
        'free_disk_bytes': 8 * 2**30,
        'scoring': '396 predictions + 198 diagnostics sealed before scoring; compare all 198 with original B1 and Frozen physical inputs; retain format failures',
    }
    save_once(dest / 'CONFIG.json', config)
    save_once(dest / 'INPUTS.json', rows)
    paths = [Path(__file__), Path(engine.__file__), final, ck, BASE / 'SOURCE_CANDIDATE_SELECTION.json',
             FIT / 'ROOT_SCORE_CROSSCHECK.json', FIT / 'independent_readback_v1/REPORT.json',
             FIT / 'ALL_PREDICTIONS_SEAL.json', FIT / 'ROOT_FINAL_TRAIN_AUDIT.json',
             pilot / 'COMPLETE.json', pilot / 'PREDICTIONS_SEAL.json',
             dest / 'CONFIG.json', dest / 'INPUTS.json', engine.PARENT / 'source_fit/LOCK.json',
             ROOT / 'vg_tta/desta3d_v2_shared_reference_cached.py', ROOT / 'vg_tta/desta3d_v2_ptd.py',
             ROOT / 'vg_tta/desta3d_v2.py', ROOT / 'methods/CURRENT_METHOD.json']
    save_once(dest / 'LOCK.json', {'pins': {str(p): engine.sha(p) for p in paths}})
    save_once(BASE / 'REGISTRATION.json', {'status': 'registered_not_started', 'run_id': RUN_ID,
        'read_only': True, 'optimizer_steps': 0, 'accepted_interface': str(pilot), 'time': time.time()})
    print('REGISTERED', dest, flush=True)


def run():
    engine.BASE = BASE
    engine.run(RUN_ID)


def launch():
    # Preserve launch overhead separately without double counting engine seconds.
    receipt = engine.PARENT / 'receipts' / f'shared_reference_{RUN_ID}.json'
    assert not receipt.exists() and not (BASE / RUN_ID / 'STARTED.json').exists()
    began = time.monotonic()
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), 'run'], cwd=ROOT)
    wall = time.monotonic() - began
    recorded = engine.read(receipt)['seconds'] if receipt.exists() else 0.
    save_once(engine.PARENT / 'receipts' / f'shared_reference_wrapper_{RUN_ID}.json', {
        'seconds': max(0., wall - recorded), 'child_wall_seconds': wall, 'worker_seconds': recorded,
        'status': 'completed' if result.returncode == 0 else 'failed', 'returncode': result.returncode,
        'scope': 'incremental subprocess imports/checks/finalization overhead, excludes worker receipt',
        'cumulative_cap_seconds': None})
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['register', 'run', 'launch'])
    args = p.parse_args()
    {'register': register, 'run': run, 'launch': launch}[args.action]()
