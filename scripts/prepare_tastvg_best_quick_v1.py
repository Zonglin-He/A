"""Lock the exact former P1/P5 streams with sealed v3 parameters, metadata only."""
import copy, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tastvg_best_quick_common_v1 import *

def run():
    assert not (BASE/'DESIGN_LOCK.json').exists(), 'Do not overwrite a locked design'
    full = ROOT/'artifacts/tastvg_best_full_v1'
    assert read(full/'STATUS.json')['status'] == 'paused_by_user_for_quick_result'
    metadata = {}
    for dataset, panel in [('vidstg', 'P1'), ('hc2', 'P5')]:
        prior_path = ROOT/'artifacts/tastvg_paper48_v1'/panel/'PLAN.json'
        prior = read(prior_path)
        source = read(full/dataset/'PLAN.json')
        lookup = {r['key'] if dataset == 'vidstg' else r['input']['original_video_id']: r for r in source['rows']}
        rows = []
        for i, r in enumerate(prior['rows']):
            key = r['key'] if dataset == 'vidstg' else r['input']['original_video_id']
            match = lookup[key]
            assert r['source'] == match['source'] and r['frame_ids'] == match['frame_ids']
            for k, v in r['input'].items():
                if k not in ['index', 'video_path']:
                    assert v == match['input'][k], (dataset, i, k)
            assert Path(r['input']['video_path']).is_file()
            row = copy.deepcopy(r)
            row.update(ordinal=i, tuning_source_exposed=match['tuning_source_exposed'])
            if dataset == 'hc2':
                row['annotation_key'] = match['annotation_key']
            rows.append(row)
        orders = copy.deepcopy(prior['orders'])
        for seq in orders.values():
            assert sorted(seq) == list(range(len(rows)))
        assert len({r['source'] for r in rows}) == len(rows)
        assert prior['conditions'] == CONDS and prior['availability'] == 25
        selected = ROOT/source['selected_config_path']
        assert sha(selected) == source['selected_config_sha256']
        cfg = read(selected)['params']
        needed = sorted({i for seq in orders.values() for i in seq[::4]})
        out = BASE/dataset
        plan = dict(dataset=dataset, rows=rows, orders=orders, conditions=CONDS,
                    params=cfg, total=len(rows)*len(orders)*len(CONDS), sources=len(rows),
                    queries=len(rows), availability=25, full_official_cohort=False,
                    historically_exposed=True, GT_used_for_selection=False,
                    selected_config_path=str(selected.relative_to(ROOT)),
                    selected_config_sha256=sha(selected), tuning_sources=source['tuning_sources'],
                    matched_previous_panel=panel, matched_previous_plan_sha256=sha(prior_path),
                    same_video_repeated_within_stream=False)
        write(out/'PLAN.json', plan)
        write(out/'EXPERT_PLAN.json', dict(rows=rows, conditions=CONDS, expert_needed=needed,
              conditions_by_parent={str(i):CONDS for i in needed}, total=len(needed)*len(CONDS)))
        cohort = {k:v for k,v in plan.items() if k not in ['rows','orders','tuning_sources']}
        cohort.update(orders=len(orders), expert_pairs=len(needed)*len(CONDS),
                      tuning_overlap_sources=sum(r['tuning_source_exposed'] for r in rows),
                      outside_tuning_sources=sum(not r['tuning_source_exposed'] for r in rows))
        write(out/'COHORT.json', cohort)
        # Reuse the two genuine no-GT logging parity inputs from the unchanged full method.
        smoke = read(full/dataset/'SMOKE.json')
        assert smoke['status'] == 'pass' and smoke['actual_selected_params'] == cfg
        write(out/'SMOKE.json', smoke)
        write(out/'SMOKE_REUSE.json', dict(qualified_unchanged_method=True,
              source=str((full/dataset/'SMOKE.json').relative_to(ROOT)),
              source_sha256=sha(full/dataset/'SMOKE.json'),
              method_sha256=sha(ROOT/'vg_tta/tastvg_best_full_method_v1.py'),
              no_new_smoke_inference=True, time=time.time()))
        for f in ['PLAN.json','EXPERT_PLAN.json','COHORT.json','SMOKE.json','SMOKE_REUSE.json']:
            metadata[f'{dataset}/{f}'] = sha(out/f)
    assert sum(read(BASE/d/'PLAN.json')['total'] for d in DATASETS) == 8808
    write(BASE/'DESIGN_LOCK.json', dict(status='locked_before_new_predictions',
          metadata=metadata, protocol_sha256=sha(ROOT/'protocols/tastvg_best_quick_v1.md'),
          parent_design_sha256=sha(full/'DESIGN_LOCK.json'),
          hc_annotation_metadata_sha256=read(full/'DESIGN_LOCK.json')['hc_annotation_metadata_sha256'],
          predictions=0, GT_geometry_used=False, time=time.time()))
    status(BASE/'STATUS.json', dict(status='design_locked_pending_runtime', time=time.time()))
    print({d:read(BASE/d/'COHORT.json') for d in DATASETS}, flush=True)

if __name__ == '__main__':
    run()
