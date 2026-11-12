"""Independent numeric Armijo trace and unchanged-source retention checks."""
from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round85';P=R/'crossplatform/round85/prediction366_attempt002'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal=read(P/'SEALED.json');rows=read(H/'experiment/ALL366_PRIVATE.json');accepted=rejected=trials=active=0;ratios=[]
for r in rows:
 f=P/'records_PRIVATE'/(r['id']+'.json');assert sha(f)==seal['files'][f.relative_to(P).as_posix()];rec=read(f);active+=rec['active_evidence'];assert rec['zero_step_max_error']==0
 with np.load(P/'samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as a:
  assert np.isfinite(a['scores']).all() and np.isfinite(a['pi']).all() and abs(a['pi'].sum()-1)<1e-12 and (a['pi']>=0).all()
  assert np.array_equal(a['queries'],a['scores'].argmax(-1))
 for variant,trajectory in rec['variants'].items():
  assert len(trajectory)==3
  for step,s in enumerate(trajectory,1):
   assert s['step']==step and s['trial_count']==len(s['trials']) and len(s['trials'])<=9
   for i,t in enumerate(s['trials']):
    assert t['scale']==1/(2**i) and np.isfinite(t['objective'])
    expected=s['objective_before']-1e-4*t['scale']*.1*min(1.,1/(s['gradient_norm']+1e-6))*s['gradient_norm']**2
    assert abs(expected-t['bound'])<1e-9
    assert t['accepted']==(t['objective']<=t['bound']+1e-10)
    assert not t['accepted'] or i==len(s['trials'])-1
    trials+=1
   if s['accepted']:
    accepted+=1;assert s['trials'][-1]['accepted'] and s['step_scale']==s['trials'][-1]['scale'] and s['objective_after']<=s['objective_before']+1e-10;assert abs(s['objective_after']-s['trials'][-1]['objective'])<1e-10;ratios.append(s['step_scale'])
   else:
    rejected+=1;assert s['step_scale']==0 and abs(s['objective_after']-s['objective_before'])<1e-10
count=read(P/'COUNTS.json');assert accepted==count['actual_optimizer_steps'] and trials==count['line_search_objective_calls'];out=H/'independent_optimization_audit_attempt001';out.mkdir();result={'state':'PASS_R85_ALL_ARMIJO_TRACES_SAME_TARGET_ACCEPTANCE_AND_REJECTION_ROLLBACK_NUMERICS','n':len(rows),'active_inputs':active,'accepted_steps':accepted,'rejected_or_inactive_steps':rejected,'trial_objective_calls':trials,'accepted_objective_increases_over_tolerance':0,'median_accepted_scale':float(np.median(ratios)) if ratios else None,'task_GT_read':False};(out/'AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
