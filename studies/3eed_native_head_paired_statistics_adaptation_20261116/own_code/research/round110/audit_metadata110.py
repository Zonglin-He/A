"""Independent complete anonymous state-change and subset/timing fidelity audit."""
from pathlib import Path
import json,csv,argparse
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=2);a=p.parse_args();R=Path(__file__).resolve().parents[2];H=R/'research/round110';O=H/f'publication/public_export_attempt{a.attempt:03d}';rd=lambda p:json.loads(Path(p).read_text());n=ns=0
for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json'):
 rr=rd(f)
 with (O/'tasks'/(f.stem.replace('_ALL_ROWS_PRIVATE','')+'.csv')).open() as h:tasks=list(csv.DictReader(h))
 with (O/'state_change'/(f.stem.replace('_ALL_ROWS_PRIVATE','')+'.csv')).open() as h:states=list(csv.DictReader(h))
 expected=[]
 for i,(r,t) in enumerate(zip(rr,tasks)):
  st=r['status'];assert all(t[k]==str(st[k]) for k in ['protected_alias_fallback','actual_C_available','seconds']);n+=1
  for label,x in st['calibrations'].items():expected.append({'row':f'ROW{i+1:06d}','source':r['source'],'target':r['target'],'calibration':label,**{k:v for k,v in x.items() if k!='eligible_channels_per_layer'},'eligible_BN1':x['eligible_channels_per_layer'][0],'eligible_BN2':x['eligible_channels_per_layer'][1]})
 assert len(states)==len(expected)
 for s,e in zip(states,expected):
  assert set(s)==set(e)
  for k,v in e.items():assert s[k]==str(v) if isinstance(v,(str,bool)) else float(s[k])==v
 ns+=len(states)
assert n==768 and ns==1448
out={'state':'PASS_ANONYMOUS_FULL_BUFFER_CHANGE_TIMING_ACTUAL_C_AND_FALLBACK_FIDELITY','n':n,'state_change_rows':ns};(H/f'independent_public_audit_attempt{a.attempt:03d}/METADATA_AUDIT_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
