"""Clean no-annotation Source constructor; no Teacher repeat or scientific change."""
from pathlib import Path
import inspect,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import stvg_temporal_semantic_grounder32_v1 as original
BASE=original.BASE
read,write,sha=original.read,original.write,original.sha
def verify():
    original.verify()
    for p,h in read(BASE/'EXECUTION_RUNTIME002.json')['pins'].items():assert sha(ROOT/p)==h,p
    assert time.time()<read(BASE/'ACTIVE_EXECUTION.json')['deadline_epoch']
def clean():
    import torch
    from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
    install_clean_loader();torch.manual_seed(20261015);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def run(stage):
    verify();clean();ns=dict(original.__dict__);ns['verify']=verify
    exec(compile(inspect.getsource(getattr(original,stage)),__file__,'exec'),ns);return ns[stage]()
if __name__=='__main__':run(sys.argv[1])
