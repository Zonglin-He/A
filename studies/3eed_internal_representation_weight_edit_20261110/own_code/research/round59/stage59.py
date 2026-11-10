"""Stage grants and lineage; scoring unavailable until independent prediction audit."""
import argparse
from common59 import *
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['adapt','audit','score','scalar']);a=ap.parse_args();e=ROOT/'research/round59/experiment';p=read(e/'EXPERIMENT_SPEC_LOCK.json')
if a.stage=='adapt':
 for rel in ['research/round59/engineering_attempt001/SEALED.json','research/round59/engineering_attempt001/RESULT.json','research/round59/experiment/DECISION_RULE_LOCK.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'ADAPTATION_PLAN_LOCK.json';assert not path.exists();dump(path,p)
elif a.stage=='audit':
 p['audit_root']='research/round59/prediction_audit_attempt001';p['programs_sha256']={rel:sha(ROOT/rel) for rel in ['research/round59/common59.py','research/round59/audit59.py']};p['inputs_sha256'][p['prediction_root']+'/SEALED.json']=sha(ROOT/p['prediction_root']/'SEALED.json');p['inputs_sha256']['research/round59/experiment/DECISION_RULE_LOCK.json']=sha(ROOT/'research/round59/experiment/DECISION_RULE_LOCK.json');path=e/'PREDICTION_AUDIT_PLAN_LOCK.json';assert not path.exists();dump(path,p)
elif a.stage=='score':
 p={'manifest':p['manifest'],'prediction_root':p['prediction_root'],'independent_prediction_audit':'research/round59/prediction_audit_attempt001/AUDIT_RESULT.json','saved_label_root':'research/round55/score_dev336_attempt001','old_prediction_root':'crossplatform/round55/adapted336_attempt002','r56_saved_score_root':'research/round56/score_dev336_attempt001','r58_saved_score_root':'research/round58/score_dev336_attempt001','split_by_id':'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json','output_root':'research/round59/score_dev336_attempt001','programs_sha256':{},'inputs_sha256':{}}
 for rel in ['research/round59/score59.py','research/round59/common59.py','research/round55/score55.py']:p['programs_sha256'][rel]=sha(ROOT/rel)
 for rel in [p['manifest'],p['prediction_root']+'/SEALED.json',p['independent_prediction_audit'],p['saved_label_root']+'/SEALED.json',p['old_prediction_root']+'/SEALED.json',p['split_by_id'],'research/round59/experiment/EXPERIMENT_SPEC_LOCK.json',p['r56_saved_score_root']+'/SEALED.json',p['r58_saved_score_root']+'/SEALED.json']:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'SINGLE_POSTSEAL_SCORE_PLAN_LOCK.json';assert not path.exists();dump(path,p);dump(e/'SINGLE_POSTSEAL_SCORE_AUTH.json',{'state':'ROOT_AUTHORIZED_R59_SINGLE_POSTSEAL_SAVED_CANDIDATE_IOU_SCORE','program_sha256':sha(ROOT/'research/round59/score59.py'),'plan_sha256':sha(path),'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json')})
else:
 p={'score_root':'research/round59/score_dev336_attempt001','saved_label_root':'research/round55/score_dev336_attempt001','manifest':p['manifest'],'split_by_id':'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json','output_root':'research/round59/saved_scalar_audit_attempt001','scorer_sha256':sha(ROOT/'research/round59/score59.py'),'programs_sha256':{},'inputs_sha256':{}}
 for rel in ['research/round59/audit_scalars59.py','research/round59/common59.py','research/round55/audit_saved_scalars55.py']:p['programs_sha256'][rel]=sha(ROOT/rel)
 for rel in [p['score_root']+'/SEALED.json',p['saved_label_root']+'/SEALED.json',p['manifest'],p['split_by_id']]:p['inputs_sha256'][rel]=sha(ROOT/rel)
 path=e/'SAVED_SCALAR_AUDIT_PLAN_LOCK.json';assert not path.exists();dump(path,p)
print({'stage':a.stage,'plan_sha256':sha(path)})
