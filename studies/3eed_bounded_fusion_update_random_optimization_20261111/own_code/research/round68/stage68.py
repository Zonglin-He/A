from common68 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['gpu_audit','predict','audit','score','scalar']);a=ap.parse_args();H=ROOT/'research/round68';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);names={'gpu_audit':['audit_gpu68.py','independent_gpu68.py'],'predict':['predict68.py'],'audit':['audit68.py','independent_cpu68.py'],'score':['score68.py'],'scalar':['audit_scalar68.py']}[a.stage]
for name in names:p['programs_sha256']['research/round68/'+name]=sha(H/name)
p['inputs_sha256'][p['random_capture_root']+'/SEALED.json']=sha(ROOT/p['random_capture_root']/'SEALED.json')
if a.stage!='gpu_audit':
 p['inputs_sha256'][p['scaled_root']+'/SEALED.json']=sha(ROOT/p['scaled_root']/'SEALED.json');p['inputs_sha256']['research/round68/independent_capture_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_capture_audit_attempt001/AUDIT_RESULT.json')
if a.stage in ['audit','score','scalar']:p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
if a.stage in ['score','scalar']:p['inputs_sha256'][p['independent_audit']]=sha(ROOT/p['independent_audit'])
if a.stage=='scalar':p['inputs_sha256'][p['score_root']+'/SEALED.json']=sha(ROOT/p['score_root']/'SEALED.json')
path=H/'experiment'/(a.stage.upper()+'_PLAN_LOCK.json');assert not path.exists();dump(path,p);print({'stage':a.stage,'sha256':sha(path)})
