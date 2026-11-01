import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round23'),str(ROOT/'crossplatform/round22'),str(ROOT/'crossplatform/round21'),str(ROOT/'crossplatform/round20')]
from common23 import read,dump,sha,seal,verify,arrsha,feature as previous_feature
BASE=ROOT/'crossplatform/round24';EXP=ROOT/'research/round24/experiment'
def feature(r):
 import numpy as np
 with np.load(BASE/'capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz')) as z:return {k:z[k].copy() for k in z.files}
def rows(split='ALL336'):return read(EXP.parent/'input'/(split+'_PRIVATE.json'))
def load_npz(p):
 import numpy as np
 with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def lock(name):
 d=read(EXP/'CODE_LOCK.json')
 for p,h in d['programs'].items():assert sha(ROOT/p)==h,p
 return d
