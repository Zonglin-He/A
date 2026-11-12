"""Priority is checked on actual endpoint losses, not just gradient dot products."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round86';P=R/'crossplatform/round86/prediction366_attempt001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=read(P/'SEALED.json');n=0;accepted=0;priority=0;class_up=0;sum_up=0;unknown=0;trials=0;inactive=0
for f in sorted((P/'records_PRIVATE').glob('*.json')):
 assert sha(f)==seal['files'][f.relative_to(P).as_posix()];r=read(f);n+=1
 for b,rec in r['bases'].items():
  assert rec['unknown_probability_error']==0;unknown+=1
  for v,ss in rec['variants'].items():
   for s in ss:
    assert len(s['trials'])==s['trial_count']<=9;trials+=len(s['trials'])
    if not rec['active']:assert s['classification_before'] is None and not s['accepted'];inactive+=1;continue
    for i,t in enumerate(s['trials']):
     assert t['scale']==1/(2**i);cb=s['classification_before']-1e-4*t['scale']*.1*max(0,s['classification_direction_dot']);tb=s['total_before']-1e-4*t['scale']*.1*max(0,s['total_direction_dot']);assert abs(cb-t['candidate_bound'])<1e-10 and abs(tb-t['total_bound'])<1e-10
     assert t['candidate_ok']==(t['classification']<=cb+1e-10) and t['total_ok']==(t['total']<=tb+1e-10)
     expect=t['candidate_ok'] if v=='candidate' else t['total_ok'] if v=='sum' else t['candidate_ok'] and t['total_ok'];assert expect==t['accepted'];assert not expect or i==len(s['trials'])-1
    if s['accepted']:
     accepted+=1;assert s['scale']==s['trials'][-1]['scale'];assert abs(s['total_after']-s['trials'][-1]['total'])<1e-10 and abs(s['classification_after']-s['trials'][-1]['classification'])<1e-10
     if v in ['joint','mismatch']:priority+=1;class_up+=s['classification_after']>s['classification_before']+1e-10;assert s['total_after']<=s['total_before']+1e-10
     if v=='candidate':assert s['classification_after']<=s['classification_before']+1e-10
     if v=='sum':sum_up+=s['classification_after']>s['classification_before']+1e-10
    else:assert abs(s['classification_after']-s['classification_before'])<1e-10 and abs(s['total_after']-s['total_before'])<1e-10
assert n==366 and class_up==0 and accepted==read(P/'COUNTS.json')['accepted_updates'];O=H/'independent_optimization_audit_attempt001';O.mkdir();result={'state':'PASS_R86_ACTUAL_ENDPOINT_CLASS_AND_TOTAL_ACCEPTANCE_ROLLBACK_AND_UNKNOWN_MASS','n':n,'accepted_steps':accepted,'accepted_priority_steps':priority,'accepted_priority_class_increases':class_up,'sum_control_candidate_increases':int(sum_up),'inactive_proposals':inactive,'unknown_mass_checks':unknown,'trial_objective_checks':trials,'task_GT_read':False};(O/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
