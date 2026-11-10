"""Stage pins: complete prefix -> predictions -> independent audit -> score."""
import argparse
from common61 import *
ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['audit','score','scalar']);a=ap.parse_args();e=ROOT/'research/round61/experiment';p=read(e/'EXPERIMENT_SPEC_LOCK.json')
if a.stage=='audit':
 p['programs_sha256']={f:sha(ROOT/f) for f in ['research/round61/common61.py','research/round61/audit61.py','crossplatform/round3/p0/upstream/models/encoder_decoder_layers.py']}
 for f in [p['capture_root']+'/SEALED.json',p['prediction_root']+'/SEALED.json']:p['inputs_sha256'][f]=sha(ROOT/f)
 path=e/'PREDICTION_AUDIT_PLAN_LOCK.json';assert not path.exists();dump(path,p)
elif a.stage=='score':
 p={'manifest':p['manifest'],'prediction_root':p['prediction_root'],'independent_prediction_audit':'research/round61/prediction_audit_attempt001/AUDIT_RESULT.json','saved_label_root':'research/round55/score_dev336_attempt001','old_prediction_root':'crossplatform/round55/adapted336_attempt002','r58_saved_score_root':'research/round58/score_dev336_attempt001','split_by_id':'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json','output_root':'research/round61/score_dev336_attempt001','programs_sha256':{},'inputs_sha256':{}}
 for f in ['research/round61/common61.py','research/round61/score61.py','research/round55/score55.py']:p['programs_sha256'][f]=sha(ROOT/f)
 for f in [p['manifest'],p['prediction_root']+'/SEALED.json',p['independent_prediction_audit'],p['saved_label_root']+'/SEALED.json',p['old_prediction_root']+'/SEALED.json',p['r58_saved_score_root']+'/SEALED.json',p['split_by_id'],'research/round59/score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json','research/round60/score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json','research/round61/experiment/EXPERIMENT_SPEC_LOCK.json']:p['inputs_sha256'][f]=sha(ROOT/f)
 path=e/'SINGLE_POSTSEAL_SCORE_PLAN_LOCK.json';assert not path.exists();dump(path,p);dump(e/'SINGLE_POSTSEAL_SCORE_AUTH.json',{'state':'ROOT_AUTHORIZED_R61_SINGLE_POSTSEAL_SAVED_CANDIDATE_IOU_SCORE','program_sha256':sha(ROOT/'research/round61/score61.py'),'plan_sha256':sha(path),'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json')})
else:
 p={'score_root':'research/round61/score_dev336_attempt001','saved_label_root':'research/round55/score_dev336_attempt001','manifest':p['manifest'],'split_by_id':'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json','output_root':'research/round61/saved_scalar_audit_attempt001','scorer_sha256':sha(ROOT/'research/round61/score61.py'),'programs_sha256':{},'inputs_sha256':{}}
 for f in ['research/round61/common61.py','research/round61/audit_scalars61.py','research/round55/audit_saved_scalars55.py']:p['programs_sha256'][f]=sha(ROOT/f)
 for f in [p['score_root']+'/SEALED.json',p['saved_label_root']+'/SEALED.json',p['manifest'],p['split_by_id']]:p['inputs_sha256'][f]=sha(ROOT/f)
 path=e/'SAVED_SCALAR_AUDIT_PLAN_LOCK.json';assert not path.exists();dump(path,p)
print({'stage':a.stage,'plan_sha256':sha(path)})
