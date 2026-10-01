"""Metadata-only follow-on design, anchored to completed sealed search selections."""
import sys, hashlib, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_extended_common_v3 import BASE, read, write, sha, status, DATASETS


def run():
    assert not (BASE/'DESIGN_LOCK.json').exists()
    prior = ROOT/'artifacts/tastvg_coordinate_tuning_v2'
    assert read(prior/'FINAL_COMPLETION.json')['status'] == 'completed_verified_and_published'
    plans = {}
    for dataset, panel in [('vidstg', 'P1'), ('hc2', 'P5')]:
        prev = read(prior/dataset/'PLAN.json')
        old = read(ROOT/'artifacts/tastvg_optuna_v1'/dataset/'PLAN.json')
        paper = read(ROOT/'artifacts/tastvg_paper48_v1'/panel/'PLAN.json')
        excluded = {r['source'] for r in prev['rows']+old['rows']}
        pool = [r for r in paper['rows'] if r['source'] not in excluded]
        pool.sort(key=lambda r: (hashlib.sha256(('Extended-confirm-v3|'+dataset+'|'+r['source']).encode()).hexdigest(), r['source']))
        assert len(pool) >= 16, 'Insufficient distinct confirmation sources'
        rows = prev['rows'][:32]+[{**r, 'original_parent':r['ordinal'], 'ordinal':32+j} for j,r in enumerate(pool[:16])]
        assert len({r['source'] for r in rows}) == 48
        orders = {f'order{k}':sorted(range(32,48), key=lambda i:hashlib.sha256((f'Extended-order-v3|{dataset}|{k}|'+rows[i]['source']).encode()).hexdigest()) for k in [1,2]}
        params = {**read(prior/dataset/'SELECTION.json')['params'], 'steps':1,
                  'student_temperature':1., 'direction_count':4}
        plan = dict(dataset=dataset, paper48_panel=panel, rows=rows,
            conditions=prev['conditions'], splits=dict(search=prev['splits']['search'],
            confirm=dict(sources=16, orders=orders, total=192)), availability=25,
            historical_exposure=True, anchor_params=params,
            predecessor_selection_sha256=sha(prior/dataset/'SELECTION.json'),
            confirmation_excludes_old_optuna_and_v2=True)
        path = BASE/dataset/'PLAN.json'
        write(path, plan)
        plans[str(path.relative_to(BASE))] = sha(path)
    write(BASE/'DESIGN_LOCK.json', dict(protocol_sha256=sha(ROOT/'protocols/tastvg_extended_sensitivity_v3.md'),
        registration_sha256=sha(BASE/'PLAN.json'), plans=plans, GT_read=False, time=time.time()))
    status(BASE/'STATUS.json', dict(status='design_locked_pending_runtime_qualification',
         model_predictions=0, GT_read=False, time=time.time()))


if __name__ == '__main__': run()
