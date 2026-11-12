"""Real two-platform pre-score nonzero-update and endpoint contract check."""
from core87 import *
P=B87/'engineering_attempt002';assert read(P/'SEALED.json')['n']==2;accepted=0
for f in (P/'records_PRIVATE').glob('*.json'):
 r=read(f)
 for v,trajectory in r['variants'].items():
  for s in trajectory:
   if s['accepted']:assert s['parameter_L2']>0 and s['classification_after']<=s['classification_before']+1e-10;assert v=='candidate_only' or s['total_after']<=s['total_before']+1e-10;accepted+=1
 assert r['variants']['full'][0]['classification_gradient_norm']>0
assert accepted>0;O=H87/'engineering_audit_attempt002';O.mkdir();dump(O/'AUDIT_RESULT.json',{'state':'PASS_R87_REAL_DRONE_QUAD_NONZERO_PRIORITY_STEPS','n':2,'accepted_steps':accepted,'task_GT_read':False});print('PASS_R87_ENGINEERING',accepted)
