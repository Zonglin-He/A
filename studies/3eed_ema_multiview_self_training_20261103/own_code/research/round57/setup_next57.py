"""Stage-specific root pins after predecessor seals; never reads task labels."""
import argparse
from common57 import *
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['adapt','audit','score','scalar']);a=ap.parse_args();p=read(ROOT/'research/round57/experiment/EXPERIMENT_SPEC_LOCK.json');e=ROOT/'research/round57/experiment'
if a.stage=='adapt':
 for rel in [p['capture_root']+'/SEALED.json',p['association_root']+'/SEALED.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'ADAPTATION_PLAN_LOCK.json';assert not path.exists();dump(path,p)
elif a.stage=='audit':
 p['audit_root']='research/round57/prediction_audit_attempt001';p['programs_sha256']={k:sha(ROOT/k) for k in ['research/round57/audit57.py','research/round57/common57.py']}
 for rel in [p['capture_root']+'/SEALED.json',p['association_root']+'/SEALED.json',p['prediction_root']+'/SEALED.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'PREDICTION_AUDIT_PLAN_LOCK.json';assert not path.exists();dump(path,p)
elif a.stage=='score':
 p={'manifest':p['manifest'],'prediction_root':p['prediction_root'],'independent_prediction_audit':'research/round57/prediction_audit_attempt001/AUDIT_RESULT.json','saved_label_root':'research/round55/score_dev336_attempt001','old_prediction_root':'crossplatform/round55/adapted336_attempt002','split_by_id':'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json','output_root':'research/round57/score_dev336_attempt001','programs_sha256':{},'inputs_sha256':{}}
 for k in ['research/round57/score57.py','research/round57/common57.py','research/round55/score55.py']:p['programs_sha256'][k]=sha(ROOT/k)
 for rel in [p['manifest'],p['prediction_root']+'/SEALED.json',p['independent_prediction_audit'],p['saved_label_root']+'/SEALED.json',p['old_prediction_root']+'/SEALED.json',p['split_by_id'],'research/round57/experiment/EXPERIMENT_SPEC_LOCK.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'SINGLE_POSTSEAL_SCORE_PLAN_LOCK.json';assert not path.exists();dump(path,p);dump(e/'SINGLE_POSTSEAL_SCORE_AUTH.json',{'state':'ROOT_AUTHORIZED_R57_SINGLE_POSTSEAL_SAVED_CANDIDATE_IOU_SCORE','plan_sha256':sha(path),'program_sha256':sha(ROOT/'research/round57/score57.py'),'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json')})
else:
 p={'score_root':'research/round57/score_dev336_attempt001','saved_label_root':'research/round55/score_dev336_attempt001','manifest':p['manifest'],'split_by_id':'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json','output_root':'research/round57/saved_scalar_audit_attempt001','scorer_sha256':sha(ROOT/'research/round57/score57.py'),'programs_sha256':{},'inputs_sha256':{}}
 for k in ['research/round57/audit_scalars57.py','research/round57/common57.py','research/round55/audit_saved_scalars55.py']:p['programs_sha256'][k]=sha(ROOT/k)
 for rel in [p['score_root']+'/SEALED.json',p['saved_label_root']+'/SEALED.json',p['manifest'],p['split_by_id']]:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'SAVED_SCALAR_AUDIT_PLAN_LOCK.json';assert not path.exists();dump(path,p)
print({'stage':a.stage,'plan_sha256':sha(path)})
