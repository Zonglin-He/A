import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crossplatform/round26'))
import common26 as prior
from common26 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round27';EXP=ROOT/'research/round27/experiment'
def rows(split='ALL528'):return read(EXP.parent/'input'/(split+'_PRIVATE.json'))
def feature(r):return load_npz(BASE/'sourceQ64_attempt001/samples_PRIVATE'/(r['id']+'.npz')) if r['stage']=='Q' else prior.feature(r)
def paths(r):return {key:BASE/(name+'Q64_attempt001') for key,name in [('prep','prepared'),('text','text'),('dino','dino'),('evidence','evidence'),('bank','bank')]} if r['stage']=='Q' else prior.paths(r)
def lock(name):
 for p,h in read(EXP/'BASE_CODE_LOCK.json')['programs'].items():assert sha(ROOT/p)==h,p
