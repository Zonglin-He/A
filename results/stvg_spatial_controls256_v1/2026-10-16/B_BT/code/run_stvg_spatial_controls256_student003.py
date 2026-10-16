"""Bind the reduced execution scope to unchanged reference fitting and readback."""
from pathlib import Path
import inspect,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_spatial_controls256_student_v1 as old
from scripts.continue_stvg_spatial_controls256_native003 import verify,CORE
def prepare(arm):
    assert arm in CORE;verify()
    src=inspect.getsource(old.prepare)
    before="for p in [Path(__file__),"
    after="for p in [Path(__file__), BASE/'NATIVE_DISPATCH_RUNTIME003.json', BASE/'EXECUTION_SCOPE_LOCK003.json',"
    assert src.count(before)==1;src=src.replace(before,after)
    ns=dict(old.__dict__);ns.update(__file__=str(Path(__file__)),verify=verify)
    exec(compile(src,str(Path(__file__)),'exec'),ns);ns['prepare'](arm)
def check(arm):
    assert arm in CORE;verify();old.check(arm)
def execute(mode,arm):
    ns=dict(old.__dict__);ns.update(check=check,__file__=str(Path(__file__)))
    exec(compile(inspect.getsource(old.execute),str(Path(__file__)),'exec'),ns);ns['execute'](mode,arm)
if __name__=='__main__':
    mode,arm=sys.argv[1:3]
    prepare(arm) if mode=='prepare' else execute(mode,arm)
