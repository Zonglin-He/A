"""Independently verify every public fit/trial/gradient/activity scalar."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[2];H=R/'research/round113';O=H/'publication/public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def csvread(p):
 with p.open() as f:return list(csv.DictReader(f))
def equal(a,b):
 assert set(a)==set(b)
 for k,v in b.items():assert a[k]==str(v) if isinstance(v,(str,bool)) else float(a[k])==v,(k,a[k],v)
def main():
 n=aliases=fits=trials=grads=0;mapping={}
 for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json'):
  name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f);tasks=csvread(O/'tasks'/(name+'.csv'));wanted_f=[];wanted_t=[];wanted_g=[]
  for i,(r,t) in enumerate(zip(rr,tasks)):
   alias=f'ROW{i+1:06d}';mapping[(name,r['id_PRIVATE'])]=alias;n+=1;aliases+=r['protected_alias_fallback'];assert t['protected_alias_fallback']==str(r['protected_alias_fallback']) and t['C_available']==str(r['C_available']) and float(t['seconds'])==r['seconds'];rec=rd(R/'crossplatform/round113'/name/'records_PRIVATE'/(r['id_PRIVATE']+'.json'))
   for fit,x in r['fits'].items():
    wanted_f.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,**{k:v for k,v in x.items() if k!='trials'}})
    for j,z in enumerate(x['trials']):wanted_t.append({'row':alias,'source':r['source'],'target':r['target'],'fit':fit,'trial':j+1,**z})
   for parent,x in rec.get('gradients',{}).items():wanted_g.append({'row':alias,'source':r['source'],'target':r['target'],'parent':parent,**{k:v for k,v in x.items() if k!='group_gradient_L2'},**{'grad_'+k:v for k,v in x['group_gradient_L2'].items()}})
  for folder,want in [('fit_endpoints',wanted_f),('trials',wanted_t),('gradients',wanted_g)]:
   got=csvread(O/folder/(name+'.csv'));assert len(got)==len(want)
   for a,b in zip(got,want):equal(a,b)
  fits+=len(wanted_f);trials+=len(wanted_t);grads+=len(wanted_g)
 want=[]
 for x in rd(H/'UPDATE_ACTIVITY_PRIVATE.json'):
  name=x['source']+'_to_'+x['target'];want.append({'row':mapping[(name,x['id_PRIVATE'])],**{k:v for k,v in x.items() if k!='id_PRIVATE'}})
 got=csvread(O/'UPDATE_ACTIVITY.csv');assert len(got)==len(want)
 for a,b in zip(got,want):equal(a,b)
 assert n==768 and aliases==44 and fits==5068 and len(want)==10136 and grads==2172
 result={'state':'PASS_ALL_PUBLIC_TIMING_FALLBACK_FIT_TRIAL_GRADIENT_ACTIVITY_SCALARS','n':n,'aliases':aliases,'fit_rows':fits,'trial_rows':trials,'gradient_rows':grads,'state_rows':len(want)};(H/'independent_public_audit_attempt001/METADATA_AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
