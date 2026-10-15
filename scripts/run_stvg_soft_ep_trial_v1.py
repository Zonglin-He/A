"""One predeclared single-factor trial, preserving the primary experiment."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_stvg_soft_ep_trial_worker_v1 as worker
label=sys.argv.pop(1);assert label in ['mu025','mu4','response1']
worker.BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1'/label
if __name__=='__main__':worker.main()
