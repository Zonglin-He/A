import argparse
from common66 import *
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['evidence','fit','readout','audit','score','scalar']);a=ap.parse_args();H=ROOT/'research/round66';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');pin(p);p['programs_sha256']=dict(p['programs_sha256']);p['inputs_sha256']=dict(p['inputs_sha256']);stage=a.stage
files={'evidence':['evidence66.py'],'fit':['fit66.py'],'readout':['readout66.py'],'audit':['audit66.py'],'score':['score66.py'],'scalar':['audit_scalar66.py']}[stage]
for name in files:p['programs_sha256']['research/round66/'+name]=sha(H/name)
if stage in ['evidence','fit','readout','audit','score','scalar']:p['inputs_sha256'][p['capture_root']+'/SEALED.json']=sha(ROOT/p['capture_root']/'SEALED.json')
if stage in ['fit','readout','audit','score','scalar']:p['inputs_sha256'][p['evidence_root']+'/SEALED.json']=sha(ROOT/p['evidence_root']/'SEALED.json')
if stage in ['readout','audit','score','scalar']:p['inputs_sha256'][p['fit_root']+'/SEALED.json']=sha(ROOT/p['fit_root']/'SEALED.json')
if stage in ['audit','score','scalar']:p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
if stage=='audit':
 p['inputs_sha256']['research/round66/independent_evidence_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_evidence_audit_attempt001/AUDIT_RESULT.json');p['programs_sha256']['research/round58/independent58.py']=sha(ROOT/'research/round58/independent58.py');p['programs_sha256']['research/round65/audit65.py']=sha(ROOT/'research/round65/audit65.py')
if stage in ['score','scalar']:p['inputs_sha256'][p['independent_audit']]=sha(ROOT/p['independent_audit'])
if stage=='scalar':p['inputs_sha256'][p['score_root']+'/SEALED.json']=sha(ROOT/p['score_root']/'SEALED.json')
path=H/'experiment'/(stage.upper()+'_PLAN_LOCK.json');assert not path.exists();dump(path,p);print({'stage':stage,'plan_sha256':sha(path)})
