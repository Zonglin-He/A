import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round32'),str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round24')]
import common32 as prior
from common32 import read,dump,sha,seal,verify,arrsha,load_npz,rows
BASE=ROOT/'crossplatform/round33';EXP=ROOT/'research/round33/experiment'
def feature(r):
 if r['stage']=='U':return load_npz(BASE/'sourceU128_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
 return prior.feature(r)
