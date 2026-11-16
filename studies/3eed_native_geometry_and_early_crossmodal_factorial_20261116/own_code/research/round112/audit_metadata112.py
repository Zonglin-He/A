"""Independent complete static edit fields, anonymous timing/coverage and negative controls."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parents[2];H=R/'research/round112';O=H/'publication/public_export_attempt001';rd=lambda p:json.loads(Path(p).read_text());n=alias=edits=0
for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json'):
 name=f.stem.replace('_ALL_ROWS_PRIVATE','');rr=rd(f)
 with (O/'tasks'/(name+'.csv')).open() as h:tasks=list(csv.DictReader(h))
 with (O/'static_edits'/(name+'.csv')).open() as h:ss=list(csv.DictReader(h))
 assert len(tasks)==len(rr);want=[]
 for i,(r,t) in enumerate(zip(rr,tasks)):
  s=r['status'];assert all(t[k]==str(s[k]) for k in ['protected_alias_fallback','seconds']);assert int(t['wrong_binding_changed_visible_seeds'])==s.get('wrong_binding_changed_visible_seeds',0);n+=1;alias+=s['protected_alias_fallback']
  for k in ['real','randomA','wrong']:
   for q in ['B_available','C_available']:assert t[k+'_'+q]==str(s['families'].get(k,{}).get(q,False))
  for arm,x in s['component_edits'].items():want.append({'row':f'ROW{i+1:06d}','source':r['source'],'target':r['target'],'arm':arm,'changed_tensor_names':';'.join(x['changed_tensor_names']),**{k:v for k,v in x.items() if k!='changed_tensor_names'}})
 assert len(want)==len(ss)
 for a,b in zip(ss,want):
  assert set(a)==set(b)
  for k,v in b.items():assert a[k]==str(v) if isinstance(v,(str,bool)) else float(a[k])==v
 edits+=len(ss)
assert n==768 and alias==44 and edits==10136
out={'state':'PASS_ANONYMOUS_FULL_STATIC_EDITS_TIMING_COVERAGE_FALLBACK_ALL_INPUTS','n':n,'protected_alias_fallback':alias,'static_edit_rows':edits};(H/'independent_public_audit_attempt001/METADATA_AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
