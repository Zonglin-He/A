"""Supplementary independent public task-metadata fidelity, no model/scoring."""
from pathlib import Path
import json,csv,hashlib,argparse
ap=argparse.ArgumentParser();ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args();tag=str(a.attempt).zfill(3)
R=Path(__file__).resolve().parents[2];H=R/'research/round109';O=H/('publication/public_export_attempt'+tag);rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/('independent_public_audit_attempt'+tag+'/AUDIT_RESULT.json'))['privacy_findings']==0;n=flags=0;aliases=0
 for direction in rd(R/'crossplatform/round109/MATRIX_SEALED.json')['directions']:
  rr=rd(H/'scores_attempt001'/(direction+'_ALL_ROWS_PRIVATE.json'))
  with (O/'tasks'/(direction+'.csv')).open() as f:pp=list(csv.DictReader(f))
  assert len(rr)==len(pp)==256
  for r,p in zip(rr,pp):
   st=r['status'];assert p['protected_alias_fallback']==str(st['protected_alias_fallback']) and float(p['seconds'])==st['seconds'];aliases+=st['protected_alias_fallback']
   for k in ['real','randomA','wrong']:
    for v in ['B_available','C_available']:assert p[k+'_'+v]==str(st.get('families',{}).get(k,{}).get(v,False));flags+=1
   n+=1
 for f in ['INPUT_SUMMARY.json','PAIR_SUMMARY.json','REGISTRATION_SUMMARY.json','COVERAGE_MODEL_SUMMARY.json','RANGE_COMPARISON.json','CE_COMPONENT_DIAGNOSTIC.json','SUBSET_RESULTS.json','CONCENTRATION_SUMMARY.json','DECISION.json']:assert (O/f).read_bytes()==(H/f).read_bytes()
 assert n==1536 and aliases==110;p=H/('independent_public_audit_attempt'+tag+'/METADATA_AUDIT_RESULT.json');p.write_text(json.dumps({'state':'PASS_ALL_1536_TIMES_FALLBACKS_AND_9216_AVAILABILITY_FLAGS_PUBLIC_FIDELITY','n':n,'availability_flags':flags,'protected_alias_fallback':aliases,'model_or_new_GT_used':False},indent=2)+'\n');print('PUBLIC_META109_PASS',n,flags)
if __name__=='__main__':main()
