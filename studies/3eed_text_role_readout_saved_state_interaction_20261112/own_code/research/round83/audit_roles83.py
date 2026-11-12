"""Saved-only dependency/offset/count audit; does not certify visual identity."""
from core83 import *
import collections
T=B83/'roles2304_attempt001';s=read(T/'SEALED.json');coh=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');summary=[];cases=[]
for i,r in enumerate(read(ROOT/'research/round69/input/ALL2304_PRIVATE.json')):
 f=T/'cases_PRIVATE'/(r['id']+'.json');assert sha(f)==s['files'][f.relative_to(T).as_posix()];x=read(f);off=np.array(x['offsets']);again=roles(r['caption'],x['document'],x['subject'],off,np.array(x['name'],bool),r['id'])
 for k in ['name','attribute','background','mismatch']:assert np.array_equal(again[k],np.array(x[k]))
 assert again['reason']==x['reason'] and again['usable']==x['usable'];assert np.array(x['attribute']).sum()==np.array(x['mismatch']).sum()==x['attribute_count'];assert not np.any(np.array(x['attribute'])&np.array(x['mismatch']));cases.append({'row':'E'+str(i+1).zfill(5),'cohort':coh[r['id']],'platform':r['platform'],'usable':x['usable'],'attribute_enabled':x['usable'] and x['attribute_count']>0,'attribute_subtokens':x['attribute_count'],'reason':x['reason']})
for c in ['VAL1280','TRAIN1024']:
 for pl in ['drone','quad','all']:
  rr=[x for x in cases if x['cohort']==c and (pl=='all' or x['platform']==pl)]
  for reason,n in collections.Counter(x['reason'] for x in rr).items():summary.append({'cohort':c,'platform':pl,'reason':reason,'n':n})
  summary.append({'cohort':c,'platform':pl,'n':len(rr),'usable':sum(x['usable'] for x in rr),'attribute_enabled':sum(x['attribute_enabled'] for x in rr),'UNKNOWN':sum(not x['usable'] for x in rr)})
dump(H83/'ROLE_COVERAGE_ROWS.json',cases);dump(H83/'ROLE_COVERAGE_SUMMARY.json',summary);O=H83/'independent_role_contract_audit_attempt001';O.mkdir();dump(O/'AUDIT_RESULT.json',{'state':'PASS_R83_ALL_CAPTION_ONLY_ROLE_RULE_OFFSET_AND_MISMATCH_COUNT_CONTRACTS','n':2304,'GT_read':False,'visual_identity_certified':False,'role_seal_sha256':sha(T/'SEALED.json')});print([x for x in summary if 'usable' in x])
