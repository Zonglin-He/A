import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round28'),str(ROOT/'crossplatform/round22')]
import common28 as c28
from common28 import read,dump,sha,seal,verify,arrsha,load_npz
from common22 import old
BASE=ROOT/'crossplatform/round29';EXP=ROOT/'research/round29/experiment'
rows=c28.rows;paths=c28.paths
def feature(r):
 return load_npz(BASE/'sourceR64_attempt001/samples_PRIVATE'/(r['id']+'.npz')) if r['stage']=='R' else c28.feature(r)

def metrics(p):
 views=[]
 for v in p['subject']['views']:
  x=[]
  for order in ['normal','reversed']:
   z=v[order]['logits'];a,b=('A','B') if order=='normal' else ('B','A');blank=v['blank']['logits']
   x.append({'d':z[b]-z[a],'uA':z[a]-z['U'],'uB':z[b]-z['U'],'blank':(z[b]-z[a])-(blank[b]-blank[a])})
  views.append(x)
 return views

def policies(p):
 v=metrics(p);d=[x['d'] for w in v for x in w];ctx=v[1];avg=sum(x['d'] for x in ctx)/2
 sgn=lambda x:'B' if x>1e-7 else 'A' if x< -1e-7 else 'U'
 raw=sgn(ctx[0]['d']);mean=sgn(avg)
 def ulabel(x):return 'B' if x['d']>0 and x['uB']>0 else 'A' if x['d']<0 and x['uA']>0 else 'U'
 agree=sgn(ctx[0]['d'])==sgn(ctx[1]['d'])==mean
 orig='B' if all(min(x['d'],x['uB'],x['blank'])>=.5 for w in v for x in w) else 'A' if all(min(-x['d'],x['uA'],-x['blank'])>=.5 for w in v for x in w) else 'U'
 label=lambda ok:mean if ok else 'U'
 return {'raw_context_AB':raw,'raw_context_ABU':ulabel(ctx[0]),'order_mean_AB':mean,'order_mean_ABU':label(sum(x['u'+mean] for x in ctx)/2>0) if mean!='U' else 'U','order_agreement':label(agree),'order_agree_ABU':label(agree and all(ulabel(x)==mean for x in ctx)),'order_agree_ABU_margin05':label(agree and all(min(abs(x['d']),x['u'+mean])>=.5 for x in ctx)) if mean!='U' else 'U','crossview_sign':label(all(sgn(x)==mean for x in d)),'all_raw_margin05':label(all((x if mean=='B' else -x)>=.5 for x in d)),'all_U_margin05':label(all(x['u'+mean]>=.5 for w in v for x in w)) if mean!='U' else 'U','all_blank_margin05':label(all((x['blank'] if mean=='B' else -x['blank'])>=.5 for w in v for x in w)),'original_all_gates':orig}
