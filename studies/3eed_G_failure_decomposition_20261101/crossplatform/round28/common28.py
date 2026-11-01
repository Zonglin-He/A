import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round27'))
import common27 as prior
from common27 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round28';EXP=ROOT/'research/round28/experiment'
def rows():return prior.rows()
feature=prior.feature;paths=prior.paths
