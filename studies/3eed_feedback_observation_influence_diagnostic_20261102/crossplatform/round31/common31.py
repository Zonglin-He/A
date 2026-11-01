import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round29'),str(ROOT/'crossplatform/round24')]
import common30 as prior
from common30 import read,dump,sha,seal,verify,arrsha,load_npz,feature
BASE=ROOT/'crossplatform/round31';EXP=ROOT/'research/round31/experiment'
def rows(split='A208'):
 if split=='R64':return read(ROOT/'research/round29/input/R64_PRIVATE.json')
 return prior.rows(split)
