"""Wait for the new six-arm coordinator, then run the unchanged global32 study."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_global32_context_v1 as old
UNIT='stvg-global32-context-002.service'
def verify():
    old.verify()
    for p,h in old.read(old.BASE/'EXECUTION_RUNTIME002.json')['pins'].items():assert old.sha(ROOT/p)==h,p
def supervise(pid):
    source=inspect.getsource(old.supervise)
    before="ROOT/'scripts/run_stvg_global32_student_v1.py'";assert source.count(before)==1
    source=source.replace(before,"ROOT/'scripts/run_stvg_global32_speed_student002.py'")
    ns=dict(old.__dict__);ns.update(verify=verify,UNIT=UNIT);exec(compile(source,str(Path(__file__)),'exec'),ns);ns['supervise'](pid)
def contracts():
    import ast
    ast.parse(inspect.getsource(old.supervise));assert "wait_core(pid);verify()" in inspect.getsource(old.supervise)
    print('GLOBAL32_SPEED_DEFERRED_DEPENDENCY_CPU_CONTRACTS_PASS_NO_MODEL_NO_GT')
if __name__=='__main__':contracts() if sys.argv[1]=='contracts' else supervise(int(sys.argv[2]))
