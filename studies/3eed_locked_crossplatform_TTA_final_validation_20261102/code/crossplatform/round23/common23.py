import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round22'),str(ROOT/'crossplatform/round21'),str(ROOT/'crossplatform/round20')]
from common22 import read,dump,sha,seal,verify,arrsha,old,feature as prior_feature
BASE=ROOT/'crossplatform/round23';EXP=ROOT/'research/round23/experiment'
def feature(r):
    if r['stage']!='K':return prior_feature(r)
    import numpy as np
    with np.load(BASE/'sourceK64_attempt001/samples_PRIVATE'/(r['id']+'.npz')) as z:return {k:z[k].copy() for k in z.files}
def lock(name):
    d=read(EXP/name)
    for p,h in d.get('programs',{}).items():assert sha(ROOT/p)==h,p
    for p,h in d.get('inputs',{}).items():assert sha(ROOT/p)==h,p
    return d
