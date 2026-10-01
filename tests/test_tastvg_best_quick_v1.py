"""Meaningful checks for changed cohort/stream handling; method itself is unchanged."""
import ast, json
from pathlib import Path

def test_matched_roster_and_expert_positions():
    for ds,panel,count,orders in [('vidstg','P1',670,2),('hc2','P5',128,1)]:
        old=json.loads((Path('artifacts/tastvg_paper48_v1')/panel/'PLAN.json').read_text())
        new=json.loads((Path('artifacts/tastvg_best_quick_v1')/ds/'PLAN.json').read_text())
        assert len(new['rows'])==count and len(new['orders'])==orders
        assert new['orders']==old['orders'] and new['conditions']==old['conditions']
        assert new['total']==count*orders*6
        assert len({r['source'] for r in new['rows']})==count
        for a,b in zip(old['rows'],new['rows']):
            assert a['key']==b['key'] and a['input']==b['input'] and a['frame_ids']==b['frame_ids']
        selected=json.loads(Path(new['selected_config_path']).read_text())
        assert new['params']==selected['params']

def test_online_math_and_update_order_are_unchanged():
    # Only the namespace import differs; the source capture/update/output loop is identical.
    full=ast.parse(Path('scripts/run_tastvg_best_full_online_v1.py').read_text())
    quick=ast.parse(Path('scripts/run_tastvg_best_quick_online_v1.py').read_text())
    a=next(n for n in full.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    b=next(n for n in quick.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    assert ast.dump(a)==ast.dump(b)
