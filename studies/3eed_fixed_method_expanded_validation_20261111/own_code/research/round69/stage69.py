from common69 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['evidence_fit','C_random','predict','audit_gpu','audit_cpu','score','scalar']);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();H=ROOT/'research/round69';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);names={'evidence_fit':['evidence_fit69.py'],'C_random':['C_random69.py'],'predict':['predict69.py'],'audit_gpu':['audit_gpu69.py','independent69.py'],'audit_cpu':['audit_cpu69.py'],'score':['score69.py'],'scalar':['audit_scalar69.py']}[a.stage]
if a.stage=='score':
 import subprocess,sys
 subprocess.run([sys.executable,str(H/'score_parser_preflight69.py')],check=True)
for name in names:p['programs_sha256']['research/round69/'+name]=sha(H/name)
if a.stage=='evidence_fit':root=H/'engineering_capture_attempt001' if a.engineering else ROOT/p['AB_capture_root'];p['inputs_sha256'][root.relative_to(ROOT).as_posix()+'/SEALED.json']=sha(root/'SEALED.json')
else:
 for key in ['AB_capture_root','evidence_root','fit_root']:p['inputs_sha256'][p[key]+'/SEALED.json']=sha(ROOT/p[key]/'SEALED.json')
if a.stage in ['predict','audit_gpu','audit_cpu','score','scalar']:p['inputs_sha256'][p['C_random_root']+'/SEALED.json']=sha(ROOT/p['C_random_root']/'SEALED.json')
if a.stage in ['audit_gpu','audit_cpu','score','scalar']:p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json')
if a.stage=='score':
 for rel in ['research/round5/audit/independent_core_v2.py','tools/obb_iou.py']:p['programs_sha256'][rel]=sha(ROOT/rel)
if a.stage in ['score','scalar']:p['inputs_sha256']['research/round69/independent_audit_attempt001/AUDIT_RESULT.json']=sha(H/'independent_audit_attempt001/AUDIT_RESULT.json')
if a.stage=='scalar':p['inputs_sha256']['research/round69/score_all2304_attempt001/SEALED.json']=sha(H/'score_all2304_attempt001/SEALED.json')
path=H/'experiment'/(a.stage.upper()+'_PLAN_LOCK'+('_ENGINEERING' if a.engineering else '')+'.json');assert not path.exists();dump(path,p);print({'stage':a.stage,'sha256':sha(path)})
