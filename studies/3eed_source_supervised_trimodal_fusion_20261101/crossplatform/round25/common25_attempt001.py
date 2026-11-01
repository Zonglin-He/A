import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round24'),str(ROOT/'crossplatform/round23'),str(ROOT/'crossplatform/round22'),str(ROOT/'crossplatform/round21'),str(ROOT/'crossplatform/round20')]
from common24 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round25';EXP=ROOT/'research/round25/experiment'
def rows(split='ALL4594'):return read(EXP.parent/'input'/(split+'_PRIVATE.json'))
def feature(r):return load_npz(BASE/'source_features_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
def rgb_feature(r):return load_npz(BASE/'rgb_features_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
