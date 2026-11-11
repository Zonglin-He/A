"""Saved original-A candidate IoUs only, after all prediction and audit seals."""
from common75 import *
import itertools
sys.path.insert(0,str(ROOT/'research/round55'));from score55 import summarize
assert read(H/'STATE.json')['CPU_scoring_authorized'];P=ROOT/'crossplatform/round75/predicted336_attempt001';ps=read(P/'SEALED.json');audit=read(H/'independent_audit_attempt001/AUDIT_RESULT.json');assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(P/'SEALED.json');assert read(H/'independent_gpu_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');old=ROOT/'research/round55/score_dev336_attempt001';ss=read(old/'SEALED.json');lf=old/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(lf)==ss['files'][lf.name];labels={x['id_PRIVATE']:x['candidate_IoUs_PRIVATE'] for x in read(lf)};splits=read(ROOT/'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json');statuses={r['id']:r['arm_statuses'] for r in read(P/'READOUT_STATUS_PRIVATE.json')};tt=[]
for i,r in enumerate(rows()):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(P/rel)==ps['files'][rel];z=load(P/rel);assert list(z['arm_names'])==ARMS;q=dict(zip(ARMS,map(int,z['queries'])));tt.append({'row':f'R{i+1:05d}','id_PRIVATE':sid,'platform':r['platform'],'scene_PRIVATE':r['scene'],'part':'DEV336','split':splits[sid],'queries':q,'IoU':{k:float(labels[sid][q[k]]) for k in ARMS},'statuses':statuses[sid]})
refs=['source','frozen_mask_probability','frozen_mask_token','frozen_mask_representation','R59','output_ensemble','shuffled_ensemble','held_mask_source'];su,sc=summarize(tt,ARMS,refs,336);neg=[]
for x in tt:
 for k,ref in itertools.permutations(ARMS,2):
  delta=x['IoU'][k]-x['IoU'][ref]
  if delta< -1e-10:neg.append({'row':x['row'],'platform':x['platform'],'part':x['part'],'split':x['split'],'arm':k,'reference':ref,'delta_pp':delta*100})
out=H/'score_dev336_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R75_POSTSEAL_SAVED_A256_IOU_JOIN','n':336,'arm_names':ARMS,'task_values':336*len(ARMS),'new_GT_scored_task_values':0,'reused_saved_task_values':336*len(ARMS),'ordered_negative_comparisons':len(neg),'single_saved_IoU_join':True,'new_raw_GT_or_geometry_scoring_calls':0,'prediction_seal_sha256':sha(P/'SEALED.json')}
for name,v in [('ALL_TASK_ROWS_PRIVATE',tt),('TASK_SUMMARY',su),('ALL_SCENE_SUMMARIES_PRIVATE',sc),('ALL_ORDERED_NEGATIVE_COMPARISONS',neg),('SCORING_RECEIPT',rec),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(name+'.json'),v)
seal(out,rec['state']);print(su['ALL336']['balanced_mIoU_percent']);print(rec)
