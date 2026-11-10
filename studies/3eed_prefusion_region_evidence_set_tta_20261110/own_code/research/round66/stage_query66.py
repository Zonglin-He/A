import argparse
from common66 import *
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['evidence_audit','fit','readout','audit','score','scalar']);a=ap.parse_args();H=ROOT/'research/round66';p=read(H/'experiment/MATCHED_QUERY_DESIGN_LOCK_v2.json');pin(p);p['programs_sha256']=dict(p['programs_sha256']);p['inputs_sha256']=dict(p['inputs_sha256']);stage=a.stage
files={'evidence_audit':['audit_evidence_query66.py','independent66.py'],'fit':['fit_query66.py'],'readout':['readout_query66.py'],'audit':['audit_query66.py'],'score':['score66.py'],'scalar':['audit_scalar66.py']}[stage]
for name in files:p['programs_sha256']['research/round66/'+name]=sha(H/name)
p['inputs_sha256'][p['evidence_root']+'/SEALED.json']=sha(ROOT/p['evidence_root']/'SEALED.json');p['inputs_sha256']['research/round66/independent_evidence_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_evidence_audit_attempt001/AUDIT_RESULT.json')
if stage in ['readout','audit','score','scalar']:p['inputs_sha256'][p['fit_root']+'/SEALED.json']=sha(ROOT/p['fit_root']/'SEALED.json');p['arm_names']=ARMS+['R59_v1_E_same_v2_pool','C_v1_E_same_v2_pool']
if stage in ['audit','score','scalar']:p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
if stage=='audit':p['inputs_sha256']['research/round66/independent_query_evidence_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_query_evidence_audit_attempt001/AUDIT_RESULT.json')
if stage in ['score','scalar']:p['inputs_sha256'][p['independent_audit']]=sha(ROOT/p['independent_audit'])
if stage=='scalar':p['inputs_sha256'][p['score_root']+'/SEALED.json']=sha(ROOT/p['score_root']/'SEALED.json')
names={'evidence_audit':'EVIDENCE_QUERY_AUDIT','fit':'FIT_QUERY','readout':'READOUT_QUERY','audit':'AUDIT_QUERY','score':'SCORE_QUERY','scalar':'SCALAR_QUERY'};path=H/'experiment'/(names[stage]+'_PLAN_LOCK_v2.json');assert not path.exists();dump(path,p);print({'stage':stage,'plan_sha256':sha(path)})
