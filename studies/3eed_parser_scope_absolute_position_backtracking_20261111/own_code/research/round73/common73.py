from pathlib import Path
import sys,json,hashlib,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round72'))
from common72 import read,dump,sha,load,seal,pin,rows,cohort_map,prefix,prep,perm
H=ROOT/'research/round73'
ARMS=['source','old_R72','parse_fix_direct','position_direct_opt','position_TTA_backtrack1','position_mismatch_direct_opt','R59','CLIP_reference']
