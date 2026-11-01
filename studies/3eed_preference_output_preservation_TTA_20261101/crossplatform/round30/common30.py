import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round29'),str(ROOT/'crossplatform/round24')]
import common29 as prior
from common29 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round30';EXP=ROOT/'research/round30/experiment'
def rows(split='A208'):
 if split=='A208':return prior.rows()[:208]
 if split=='DEV64':return prior.rows()[464:]
 if split=='S64':return read(EXP.parent/'input/S64_PRIVATE.json')
 raise ValueError(split)
def feature(r):return load_npz(BASE/'sourceS64_attempt001/samples_PRIVATE'/(r['id']+'.npz')) if r['stage']=='S' else prior.feature(r)
