import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round21'),str(ROOT/'crossplatform/round20')]
from common21 import read,dump,sha,seal,verify,feature,old,ROWS,arrsha
BASE=ROOT/'crossplatform/round22';EXP=ROOT/'research/round22/experiment';PRIOR=ROOT/'crossplatform/round21'
def lock(name):
    x=read(EXP/name)
    for p,h in x.get('programs',{}).items():assert sha(ROOT/p)==h,p
    for p,h in x.get('inputs',{}).items():assert sha(ROOT/p)==h,p
    return x
def prior(r,kind):
    folder={'text':'text208_attempt003','dino':'dino208_attempt003','reference':'reference208_attempt003','direct':'direct208_attempt003'}[kind]
    return read(PRIOR/folder/'cases_PRIVATE'/(r['id']+'.json'))
