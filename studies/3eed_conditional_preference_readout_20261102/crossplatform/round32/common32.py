import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round31'),str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round24')]
import common31 as prior
from common31 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round32';EXP=ROOT/'research/round32/experiment'
def rows(split='A208'):
 if split in ['T128','U128','FINAL384']:return read(ROOT/'research/NIGHT_20261101/input'/(split+'_PRIVATE.json'))
 return prior.rows(split)
def feature(r):
 if r['stage']=='T':return load_npz(BASE/'sourceT128_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
 return prior.feature(r)
