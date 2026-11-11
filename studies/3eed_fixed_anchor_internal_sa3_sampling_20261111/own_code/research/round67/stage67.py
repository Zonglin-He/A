from common67 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['capture_audit','predict','audit','score','scalar']);a=ap.parse_args();H=ROOT/'research/round67';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);files={'capture_audit':['audit_capture67.py'],'predict':['predict67.py'],'audit':['audit67.py'],'score':['score67.py'],'scalar':['audit_scalar67.py']}[a.stage]
for name in files:p['programs_sha256']['research/round67/'+name]=sha(H/name)
p['inputs_sha256'][p['capture_root']+'/SEALED.json']=sha(ROOT/p['capture_root']/'SEALED.json')
if a.stage in ['predict','audit','score','scalar']:p['inputs_sha256']['research/round67/independent_capture_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json')
if a.stage in ['audit','score','scalar']:p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
if a.stage in ['score','scalar']:p['inputs_sha256'][p['independent_audit']]=sha(ROOT/p['independent_audit'])
if a.stage=='scalar':p['inputs_sha256'][p['score_root']+'/SEALED.json']=sha(ROOT/p['score_root']/'SEALED.json')
path=H/'experiment'/(a.stage.upper()+'_PLAN_LOCK.json');assert not path.exists();dump(path,p);print({'stage':a.stage,'sha256':sha(path)})
