"""Independent complete timing, coverage, fallback and failed-state reuse fidelity."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[2];H=R/'research/round111';O=H/'publication/public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());n=alias=reuse=0
for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json'):
 rr=rd(f)
 with (O/'tasks'/(f.stem.replace('_ALL_ROWS_PRIVATE','')+'.csv')).open() as h:tasks=list(csv.DictReader(h))
 assert len(tasks)==len(rr)
 for r,t in zip(rr,tasks):
  s=r['status'];assert all(t[k]==str(s[k]) for k in ['protected_alias_fallback','seconds']);assert t['readout_amended_cached_state_reuse']==str(s.get('readout_amended_cached_state_reuse',False));n+=1;alias+=s['protected_alias_fallback'];reuse+=s.get('readout_amended_cached_state_reuse',False)
  for k in ['real','randomA','wrong']:
   for q in ['B_available','C_available']:assert t[k+'_'+q]==str(s['families'].get(k,{}).get(q,False))
assert n==768 and alias==44 and reuse==62
x={'state':'PASS_ANONYMOUS_FULL_COVERAGE_TIMING_FALLBACK_REUSED_STATE_FIDELITY','n':n,'protected_alias_fallback':alias,'nonalias_hidden_states_reused':reuse};(H/'independent_public_audit_attempt001/METADATA_AUDIT_RESULT.json').write_text(json.dumps(x,indent=2)+'\n');print(x)
