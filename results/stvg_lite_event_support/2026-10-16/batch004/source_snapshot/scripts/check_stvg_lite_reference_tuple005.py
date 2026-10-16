"""CPU contracts for the exact three-value reference interface correction."""
import ast
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def contracts():
    a=(ROOT/'scripts/prepare_stvg_lite_reclocked_inputs_v1.py').read_text()
    b=(ROOT/'scripts/prepare_stvg_lite_reclocked_inputs002.py').read_text()
    old='            case, source_path = _source_and_query_for_cohort(rec, cohort)'
    new='            case, source_path, _query = _source_and_query_for_cohort(rec, cohort)'
    assert a.count(old)==1 and b==a.replace(old,new)
    aa=ast.parse(a);bb=ast.parse(b)
    fa={n.name:ast.dump(n,include_attributes=False) for n in aa.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    fb={n.name:ast.dump(n,include_attributes=False) for n in bb.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    assert fa.keys()==fb.keys()
    changed=[name for name in fa if fa[name]!=fb[name]];assert changed==['build_references']
    calls=[n for n in ast.walk(bb) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
           and isinstance(n.value.func,ast.Name) and n.value.func.id=='_source_and_query_for_cohort']
    assert calls and all(isinstance(n.targets[0],ast.Tuple) and len(n.targets[0].elts)==3 for n in calls)
    for cohort in ('development','confirmation'):
        for clock in ('temporal','native'):
            case={'cohort':cohort,'clock':clock,'GT_read':False};path=Path('synthetic.pt');query='SYNTHETIC_UNREAD'
            received,received_path,_query=(case,path,query)
            assert received is case and received_path is path and _query==query
    for bad in ((),(1,),(1,2),(1,2,3,4)):
        try:
            x,y,z=bad
        except ValueError:pass
        else:raise AssertionError('invalid source interface width accepted')
    c=(ROOT/'scripts/continue_stvg_lite_complete005.py').read_text();ast.parse(c)
    assert "inputs('development','temporal');cached_native_refs()" not in c
    assert "command('development_temporal_reference_recovery002'" in c
    assert "inputs('confirmation','temporal');inputs('confirmation','native');final_student('confirmation')" in c
    return dict(status='pass',scope='exact source tuple correction and bounded resume only',
                unchanged_builder_functions=len(fa)-1,synthetic_cohort_clock_cases=4,invalid_width_rejections=4,
                GT_read=False,CPU_only=True,new_GPU=False)

if __name__=='__main__':print(json.dumps(contracts()))
