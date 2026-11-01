"""R35 only exposes already-used A208/DEV64; protected confirmation is separate."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round24')]
from common30 import read,dump,sha,seal,verify,load_npz,rows,feature
BASE=ROOT/'crossplatform/round35';EXP=ROOT/'research/round35/experiment'
