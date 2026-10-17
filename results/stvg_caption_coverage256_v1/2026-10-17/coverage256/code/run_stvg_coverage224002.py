"""Launcher-only deadline binding; original coverage computation unchanged."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_coverage224_v1 as original
configure_original=original.configure
def configure(module):
    module=configure_original(module)
    for p,h in original.read(original.BASE/'RECOVERY_RUNTIME002.json')['pins'].items():assert original.sha(ROOT/p)==h,p
    module.common.deadline=lambda:original.read(original.BASE/'RECOVERY_RUNTIME002.json')['deadline_epoch']
    return module
original.configure=configure
if __name__=='__main__':original.run(sys.argv[1])
