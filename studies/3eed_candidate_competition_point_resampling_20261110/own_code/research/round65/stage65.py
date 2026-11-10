"""Root-pinned sequential audit/scoring stages; no method selection."""
import argparse
from common65 import *
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','score','scalar']);a=ap.parse_args();H=ROOT/'research/round65';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');pin(p);p['inputs_sha256']=dict(p['inputs_sha256']);p['programs_sha256']=dict(p['programs_sha256'])
for rel in [p['prepared_root']+'/SEALED.json',p['new_capture_root']+'/SEALED.json',p['prediction_root']+'/SEALED.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
if a.stage=='audit':
 p['programs_sha256']['research/round65/audit65.py']=sha(H/'audit65.py');name='AUDIT';assert read(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json')['helper_sha256']==sha(H/'audit65.py');p['inputs_sha256']['research/round65/independent_capture_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json')
else:
 p['independent_audit']='research/round65/independent_audit_attempt002/AUDIT_RESULT.json'
 p['inputs_sha256'][p['independent_audit']]=sha(ROOT/p['independent_audit']);p['inputs_sha256']['research/round55/score_dev336_attempt001/SEALED.json']=sha(ROOT/'research/round55/score_dev336_attempt001/SEALED.json');name='SCORE' if a.stage=='score' else 'SCALAR'
 f='score65.py' if a.stage=='score' else 'audit_scalar65.py';p['programs_sha256']['research/round65/'+f]=sha(H/f)
 if a.stage=='scalar':p['inputs_sha256'][p['score_root']+'/SEALED.json']=sha(ROOT/p['score_root']/'SEALED.json')
path=H/'experiment'/(name+'_PLAN_LOCK.json');assert not path.exists();dump(path,p);print({'stage':a.stage,'plan_sha256':sha(path)})
