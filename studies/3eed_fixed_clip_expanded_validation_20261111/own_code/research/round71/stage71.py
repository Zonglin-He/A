from common71 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['clip','predict','audit','score']);ap.add_argument('--engineering',action='store_true');a=ap.parse_args();p=read(H/'experiment/DESIGN_LOCK.json')
if a.stage=='clip':
 files=['clip71.py'];roots=[H/'engineering_prepare4_attempt001'] if a.engineering else [ROOT/p['prepared_root']]
elif a.stage=='predict':files=['predict71.py'];roots=[ROOT/p['prepared_root'],ROOT/p['clip_root']]
elif a.stage=='audit':
 p['inputs_sha256']['research/round69/input/SCORING_INDEX_PRIVATE.json']=sha(ROOT/'research/round69/input/SCORING_INDEX_PRIVATE.json');files=['audit71.py','audit_gpu71.py'];roots=[ROOT/p['prepared_root'],ROOT/p['clip_root'],ROOT/p['prediction_root']]
else:
 assert read(H/'independent_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_') and read(H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');files=['score71.py','audit_scalar71.py','analyze71.py','audit_derived71.py','write_report71.py'];roots=[ROOT/p['prediction_root']]
 for f in ['research/round69/score_all2304_attempt001/SEALED.json','research/round55/score_dev336_attempt001/SEALED.json','research/round70/score_dev336_attempt001/SEALED.json','research/round71/independent_audit_attempt001/AUDIT_RESULT.json','research/round71/independent_gpu_audit_attempt001/AUDIT_RESULT.json']:p['inputs_sha256'][f]=sha(ROOT/f)
for f in files:p['programs_sha256']['research/round71/'+f]=sha(H/f)
for root in roots:
 if (root/'SEALED.json').exists():p['inputs_sha256'][root.relative_to(ROOT).as_posix()+'/SEALED.json']=sha(root/'SEALED.json')
 else:
  assert a.stage=='clip' and not a.engineering;p['streaming_preparation']='Per-case ready file read, all consumed input bytes verified against final preparation seal before CLIP output sealing'
path=H/'experiment'/(a.stage.upper()+'_PLAN_LOCK'+('_ENGINEERING' if a.engineering else '')+'.json');assert not path.exists();dump(path,p)
if a.stage=='score':s=read(H/'STATE.json');s['CPU_scoring_authorized']=True;dump(H/'STATE.json',s)
print({'stage':a.stage,'sha256':sha(path)})
