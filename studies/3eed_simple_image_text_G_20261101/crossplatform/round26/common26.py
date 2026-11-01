import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round25'),str(ROOT/'crossplatform/round24'),str(ROOT/'crossplatform/round23'),str(ROOT/'crossplatform/round22'),str(ROOT/'crossplatform/round21'),str(ROOT/'crossplatform/round20')]
from common25 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round26';EXP=ROOT/'research/round26/experiment'
def rows(split='ALL464'):return read(EXP.parent/'input'/(split+'_PRIVATE.json'))
def feature(r):
 p=BASE/'sourceP64_attempt001' if r['stage']=='P' else ROOT/'crossplatform/round25/source_features_attempt002'
 return load_npz(p/'samples_PRIVATE'/(r['id']+'.npz'))
def lock(name):
 d=read(EXP/'BASE_CODE_LOCK.json')
 for p,h in d['programs'].items():assert sha(ROOT/p)==h,p
 return d
def paths(r):
 if r['stage'] in ['N','P']:b=BASE;suffix='F128'
 elif r['stage']=='M':b=ROOT/'crossplatform/round24';suffix='M64'
 else:b=ROOT/'crossplatform/round23';suffix='272'
 return {key:b/(name+suffix+'_attempt001') for key,name in [('prep','prepared'),('text','text'),('dino','dino'),('evidence','evidence'),('bank','bank')]}
