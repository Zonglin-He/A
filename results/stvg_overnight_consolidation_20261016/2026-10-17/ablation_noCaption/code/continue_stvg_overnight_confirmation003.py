"""System-Python pidfd launcher; unchanged confirmation science and dispatch."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import continue_stvg_overnight_confirmation002 as original
from scripts import stvg_overnight_common_v1 as c
def main(pid):
    for p,h in c.read(c.BASE/'CONFIRMATION_LAUNCH_RUNTIME003.json')['pins'].items():assert c.sha(ROOT/p)==h,p
    original.UNIT='stvg-overnight-confirmation-003.service';original.main(pid)
if __name__=='__main__':main(int(sys.argv[1]))
