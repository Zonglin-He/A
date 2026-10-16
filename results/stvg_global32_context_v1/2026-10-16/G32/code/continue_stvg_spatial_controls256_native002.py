"""Adds an independent policy replay before the first large Student group."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_spatial_controls256_native_v1 as old
BASE=old.BASE
def verify():
    for p,h in old.read(BASE/'NATIVE_DISPATCH_RUNTIME002.json')['pins'].items():assert old.sha(ROOT/p)==h,p
    old.verify()
def supervise(pid):
    # Same finite stage schedule; only this entrypoint and extra readback change.
    ns=dict(old.__dict__);ns.update(__file__=str(Path(__file__)),UNIT='stvg-spatial-controls256-native-002.service',verify=verify);exec(compile(inspect.getsource(old.supervise),str(Path(__file__)),'exec'),ns);ns['supervise'](pid)
if __name__=='__main__':
    m=sys.argv[1]
    if m=='after_inputs':supervise(int(sys.argv[2]))
    else:
        verify()
        if m=='qualification':old.qualification()
        elif m=='teacher':old.teacher(sys.argv[2])
        else:
            old.teacher_root(sys.argv[2])
            if sys.argv[2]=='Full_E1':
                from scripts.audit_stvg_spatial_controls256_policies_v1 import main
                main()
