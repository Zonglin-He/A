"""Root-only prelaunch lock for the already authorized finite continuation.

This file never runs experiments or schedules periodic work.
"""
from pathlib import Path
import ast
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_lite_event_common_v1 import BASE,read,write,sha,verify


def closure(paths):
    todo=list(paths);seen=set()
    while todo:
        p=todo.pop().resolve()
        if p in seen:continue
        assert p.is_file(),str(p)
        seen.add(p)
        if p.suffix!='.py':continue
        for n in ast.walk(ast.parse(p.read_text())):
            candidates=[]
            if isinstance(n,ast.Import):
                candidates=[a.name for a in n.names]
            elif isinstance(n,ast.ImportFrom) and n.module:
                candidates=[n.module]+[n.module+'.'+a.name for a in n.names]
            for module in candidates:
                if module.startswith('scripts.'):
                    q=ROOT/('/'.join(module.split('.'))+'.py')
                    if q.is_file():todo.append(q)
    return seen


def main():
    initial=verify()
    names=[
        'select_stvg_lite_event_v1.py','continue_stvg_lite_event_search_v1.py',
        'continue_stvg_lite_complete_v1.py','continue_stvg_lite_tail_v1.py',
        'lock_stvg_lite_finite_tail_v1.py','prepare_stvg_lite_reclocked_inputs_v1.py',
        'run_stvg_lite_confirmation_temporal_v1.py','run_stvg_lite_complete_student_v1.py',
        'review_stvg_lite_complete_v1.py','score_stvg_lite_complete_v1.py',
        'prepare_stvg_lite_confirmation_source_v1.py','run_stvg_lite_confirmation_source_v1.py',
        'stvg_temporal_lite_fit_v1.py','run_stvg_temporal_lite128_v1.py',
        'readback_stvg_temporal_lite128_v1.py','score_stvg_temporal_lite128_v1.py',
    ]
    sources=closure([ROOT/'scripts'/n for n in names])
    pins=dict(initial['pins'])
    for p in sources:pins[str(p.relative_to(ROOT))]=sha(p)
    for p in [BASE/'EXECUTION_RUNTIME.json',BASE/'DESIGN_LOCK.json',
              BASE/'temporal/TEMPORAL_LITE_RUNTIME_LOCK.json',
              BASE/'confirmation/ROSTER_LOCK.json',BASE/'confirmation/SOURCE_RUNTIME.json',
              BASE/'confirmation/ROOT_ROSTER_READBACK002.json',
              ROOT/'protocols/stvg_lite_event_support_v1.md']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    protected=initial['protected']
    for rel,h in {**pins,**protected}.items():assert sha(ROOT/rel)==h,rel
    template=BASE/'COMPLETE_DISPATCH_RUNTIME_TEMPLATE.json'
    write(template,dict(status='locked',scope='prelocked finite final pipeline code and input custody; no future measurement claimed',pins=pins,protected=protected,GT_read=False,CPU_only=True,confirmation_count=64,source_closure_count=len(sources),no_further_method_selection_on_confirmation=True))
    unit='stvg-lite-event128-001.service'
    props=dict(x.split('=',1)for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=ActiveState,MainPID'],text=True).splitlines()if'='in x)
    assert props['ActiveState']=='active'
    pid=int(props['MainPID']);assert pid==891914
    process=Path('/proc')/str(pid)
    start=int((process/'stat').read_text().split(') ',1)[1].split()[19])
    tailpins=dict(pins);tailpins[str(template.relative_to(ROOT))]=sha(template)
    write(BASE/'FINITE_TAIL_RUNTIME.json',dict(status='locked',scope='one source-identity-bound pidfd wait then one finite authorized continuation, no recurring scheduler',pins=tailpins,protected=protected,parent=dict(unit=unit,pid=pid,start_ticks=start,cmdline_sha256=sha(process/'cmdline')),GT_read=False,CPU_only=True))
    for rel,h in {**tailpins,**protected}.items():assert sha(ROOT/rel)==h,rel
    write(BASE/'ROOT_FINITE_TAIL_PRELAUNCH_READBACK.json',dict(status='pass',count=len(tailpins),source_closure_count=len(sources),GT_read=False,CPU_only=True,scope='all current SHA and protected bindings; future results remain pending',runtime_sha256=sha(BASE/'FINITE_TAIL_RUNTIME.json'),complete_template_sha256=sha(template)))
    print(json.dumps(dict(status='locked',pins=len(tailpins),source_closure=len(sources),parent_pid=pid,GT_read=False)))


if __name__=='__main__':main()
