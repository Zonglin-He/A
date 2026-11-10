"""One postseal saved-label evaluation: exact old256 candidate IoUs, no raw GT read."""
import argparse,itertools,sys,time
from common59 import *
sys.path.insert(0,str(ROOT/'research/round55'));from score55 import summarize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);a=ap.parse_args();p=read(a.plan);auth=read(a.authorization);assert auth['state']=='ROOT_AUTHORIZED_R59_SINGLE_POSTSEAL_SAVED_CANDIDATE_IOU_SCORE' and auth['plan_sha256']==sha(a.plan) and auth['program_sha256']==sha(__file__);pin(p)
 pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');audit=read(ROOT/p['independent_prediction_audit']);assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(pred/'SEALED.json') and auth['prediction_seal_sha256']==sha(pred/'SEALED.json');assert ps['arm_names']==ARMS and audit['n']==336
 # Old target labels become accessible only after sealed-prediction/audit grants.
 old=ROOT/p['saved_label_root'];ss=read(old/'SEALED.json')
 def member(name):assert sha(old/name)==ss['files'][name];return read(old/name)
 labels=member('ALL_CANDIDATE_IOUS_PRIVATE.json');oldt=member('ALL_TASK_ROWS_PRIVATE.json');rows=read(ROOT/p['manifest']);assert len(rows)==len(labels)==len(oldt)==336;split=read(ROOT/p['split_by_id']);oldpred=ROOT/p['old_prediction_root'];ops=read(oldpred/'SEALED.json');statuses=read(pred/'READOUT_STATUS_PRIVATE.json');assert sha(pred/'READOUT_STATUS_PRIVATE.json')==ps['files']['READOUT_STATUS_PRIVATE.json'];out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True);start=time.perf_counter();tasks=[]
 old56=ROOT/p['r56_saved_score_root'];ss56=read(old56/'SEALED.json');assert sha(old56/'ALL_TASK_ROWS_PRIVATE.json')==ss56['files']['ALL_TASK_ROWS_PRIVATE.json'];r56=read(old56/'ALL_TASK_ROWS_PRIVATE.json');assert len(r56)==336
 old58=ROOT/p['r58_saved_score_root'];ss58=read(old58/'SEALED.json');assert sha(old58/'ALL_TASK_ROWS_PRIVATE.json')==ss58['files']['ALL_TASK_ROWS_PRIVATE.json'];r58=read(old58/'ALL_TASK_ROWS_PRIVATE.json')
 for i,(row,label,base,st) in enumerate(zip(rows,labels,oldt,statuses)):
  sid=row['id'];assert label['id_PRIVATE']==base['id_PRIVATE']==st['id']==sid and label['platform']==base['platform']==row['platform'] and base['scene_PRIVATE']==row['scene'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel] and sha(oldpred/rel)==ops['files'][rel];z=load(pred/rel);original=load(oldpred/rel,['source_boxes','source_scores']);assert np.array_equal(z['source_boxes'],original['source_boxes']) and np.array_equal(z['source_scores'],original['source_scores']);assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);ious=np.array(label['candidate_IoUs_PRIVATE']);assert ious.shape==(256,) and np.isfinite(ious).all() and (ious>=0).all() and (ious<=1).all();q=dict(zip(ARMS,map(int,z['queries'])));v={arm:float(ious[q[arm]]) for arm in ARMS};assert q['source']==base['queries']['source'] and v['source']==base['IoU']['source']
  assert r56[i]['id_PRIVATE']==sid and q['r56_one']==r56[i]['queries']['tta1'] and v['r56_one']==r56[i]['IoU']['tta1']
  assert r58[i]['id_PRIVATE']==sid
  for arm,key in [('r58_online_ema_teacher','backtrack_online_ema_teacher_post'),('r58_online_ema_student','backtrack_online_ema_student_post')]:assert q[arm]==r58[i]['queries'][key] and v[arm]==r58[i]['IoU'][key]
  tasks.append({'row':f'R{i+1:05d}','id_PRIVATE':sid,'platform':row['platform'],'scene_PRIVATE':row['scene'],'part':'DEV336','split':split[sid],'queries':q,'IoU':v,'statuses':st['arm_statuses']})
 refs=REFS;summary,scenes=summarize(tasks,ARMS,refs,336);negative=[]
 for row in tasks:
  for arm,reference in itertools.product(ARMS,repeat=2):
   delta=row['IoU'][arm]-row['IoU'][reference]
   if arm!=reference and delta< -1e-10:negative.append({'row':row['row'],'part':row['part'],'split':row['split'],'platform':row['platform'],'arm':arm,'reference':reference,'delta_pp':float(delta*100)})
 for name,value in [('ALL_TASK_ROWS_PRIVATE',tasks),('TASK_SUMMARY',summary),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',negative),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(name+'.json'),value)
 receipt={'state':'PASS_R59_SINGLE_POSTSEAL_SAVED_LABEL_ORIGINAL_BOX_TASK_SCORE','n':336,'arm_names':ARMS,'task_values':336*len(ARMS),'new_GT_scored_task_values':0,'reused_saved_task_values':336*len(ARMS),'ordered_negative_comparisons':len(negative),'single_GT_scoring_pass':True,'new_raw_GT_or_geometry_scoring_calls':0,'all_values_reused_from_old_fixed_candidate_IoUs':True,'unknown_fallback_and_no_update_rows_remain_full_denominator':True,'source_256_boxes_not_changed':True,'balanced_platforms_not_micro_selection':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'independent_prediction_audit_sha256':sha(ROOT/p['independent_prediction_audit']),'source_saved_label_seal_sha256':sha(old/'SEALED.json'),'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'seconds':time.perf_counter()-start,'no_protected_FINAL_specific_artifacts':True};dump(out/'SCORING_RECEIPT.json',receipt);seal(out,receipt['state']);print(receipt,flush=True)
if __name__=='__main__':main()
