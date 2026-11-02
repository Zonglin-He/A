import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'crossplatform/round20'))
from common_v2 import read,dump,sha,seal,verify,feature,arrsha
BASE=ROOT/'crossplatform/round21'
EXP=ROOT/'research/round21/experiment'
ROWS=ROOT/'research/round20/input/ALL208_PRIVATE.json'
def old(r,kind):
    name={'text':'textZ64_attempt001' if r['stage']=='Z' else 'text144_attempt001','pairs':'pairsZ64_attempt001' if r['stage']=='Z' else 'pairs144_attempt001','visual':'visualZ64_attempt001' if r['stage']=='Z' else 'visual144_attempt001','dino':'dinoZ64_attempt001' if r['stage']=='Z' else 'dino144_attempt001'}[kind]
    return read(ROOT/'crossplatform/round20'/name/'cases_PRIVATE'/(r['id']+'.json'))
def lock(name):
    d=read(EXP/name)
    for rel,h in d.get('programs',{}).items(): assert sha(ROOT/rel)==h,rel
    for rel,h in d.get('inputs',{}).items(): assert sha(ROOT/rel)==h,rel
    return d
