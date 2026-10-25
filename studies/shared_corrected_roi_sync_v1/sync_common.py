"""Independent shared-ROI structural and own-K synchronization namespace."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'studies/shared_corrected_roi_v1'))
import shared_common as selected
from shared_common import *
S=Path(__file__).parent;D=R/'outputs/shared_corrected_roi_sync_v1';P=R/'results/shared_corrected_roi_sync_v1'
sys.path.insert(0,str(S))
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for f,h in p['code_sha256'].items():assert sha(R/f)==h,f
 for f,h in p['input_sha256'].items():assert sha(R/f)==h,f
 selected.validate();return p
def arms(task):return parent.arms(task)
def pairs(task):return parent.pairs(task)
def rows(task,ds):return parent.rows(task,ds)
