"""Target labels are joined only after sealed inference and independent audit."""
from common76 import *
import itertools
sys.path.insert(0,str(ROOT/'research/round55'));from score55 import summarize
assert read(H/'STATE.json')['CPU_scoring_authorized'];P=BASE/'target336_attempt001';ps=read(P/'SEALED.json');audit=read(H/'independent_target_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(P/'SEALED.json');assert read(H/'independent_source_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');old=ROOT/'research/round55/score_dev336_attempt001';ss=read(old/'SEALED.json');lf=old/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(lf)==ss['files'][lf.name];labels={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(lf)};split=read(ROOT/'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json');status={x['id']:x['arm_statuses'] for x in read(P/'READOUT_STATUS_PRIVATE.json')};tt=[]
for i,r in enumerate(target_rows()):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(P/rel)==ps['files'][rel];z=load(P/rel);assert list(z['arm_names'])==ARMS;qq=dict(zip(ARMS,map(int,z['queries'])));tt.append({'row':f'R{i+1:05d}','id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':'DEV336','split':split[sid],'queries':qq,'IoU':{k:float(labels[sid][qq[k]]) for k in ARMS},'statuses':status[sid]})
refs=['source','independent_frozen','rgb_to_point_frozen','bidirectional_frozen','bidirectional_frozen_budget3','bidirectional_TTA3_wrong_correspondence','R59'];su,sc=summarize(tt,ARMS,refs,336);ng=[]
for x in tt:
 for k,ref in itertools.permutations(ARMS,2):
  v=x['IoU'][k]-x['IoU'][ref]
  if v< -1e-10:ng.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':v*100})
out=H/'score_dev336_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R76_SOURCE_PREPARED_TARGET_POSTSEAL_SAVED_A256_IOU_JOIN','n':336,'arm_names':ARMS,'task_values':len(ARMS)*336,'new_GT_scored_task_values':0,'reused_saved_task_values':len(ARMS)*336,'ordered_negative_comparisons':len(ng),'single_saved_IoU_join':True,'new_raw_target_GT_or_geometry_calls':0,'source_selection_uses_target_labels':False,'prediction_seal_sha256':sha(P/'SEALED.json')}
for name,v in [('ALL_TASK_ROWS_PRIVATE',tt),('TASK_SUMMARY',su),('ALL_SCENE_SUMMARIES_PRIVATE',sc),('ALL_ORDERED_NEGATIVE_COMPARISONS',ng),('SCORING_RECEIPT',rec),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(name+'.json'),v)
seal(out,rec['state']);print(su['ALL336']['balanced_mIoU_percent']);print(rec)
