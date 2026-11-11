import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'research/round74'))
from common74 import *
H=ROOT/'research/round74/expanded'
def rows():return read(H/'input/REMAINDER2240_PRIVATE.json')
