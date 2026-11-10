"""One stage-specific postseal saved-IoU evaluation; labels never enter runtime."""
import argparse,itertools,sys,time
from common64 import *
sys.path.insert(0,str(ROOT/'research/round55'));from score55 import summarize
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);pred=ROOT/p['prediction_root'];ps=read(pred/'SEALED.json');audit=read(ROOT/p['independent_audit']);assert audit['state'].startswith('PASS_') and audit['prediction_seal_sha256']==sha(pred/'SEALED.json');arms=p['arm_names'];assert ps['arm_names']==arms and ps['n']==336
 old=ROOT/'research/round55/score_dev336_attempt001';oldseal=read(old/'SEALED.json');labelsfile=old/'ALL_CANDIDATE_IOUS_PRIVATE.json';assert sha(labelsfile)==oldseal['files'][labelsfile.name];labels=read(labelsfile);base=read(old/'ALL_TASK_ROWS_PRIVATE.json');manifest=read(ROOT/p['manifest']);split=read(ROOT/'research/round51/experiment/DEV_SPLIT_BY_ID_PRIVATE.json');statuses=read(pred/'READOUT_STATUS_PRIVATE.json');assert sha(pred/'READOUT_STATUS_PRIVATE.json')==ps['files']['READOUT_STATUS_PRIVATE.json'];out=ROOT/p['score_root'];assert not out.exists();out.mkdir(parents=True);tasks=[];start=time.perf_counter()
 for i,(row,label,prior,st) in enumerate(zip(manifest,labels,base,statuses)):
  sid=row['id'];assert sid==label['id_PRIVATE']==prior['id_PRIVATE']==st['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(pred/rel)==ps['files'][rel];z=load(pred/rel);canonical=load(ROOT/'crossplatform/round55/adapted336_attempt002'/rel,['source_boxes','source_scores']);assert np.array_equal(z['source_boxes'],canonical['source_boxes']) and np.array_equal(z['source_scores'],canonical['source_scores']);assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);q=dict(zip(arms,map(int,z['queries'])));ious=np.array(label['candidate_IoUs_PRIVATE']);values={arm:float(ious[q[arm]]) for arm in arms}
  if 'source' in arms:assert q['source']==prior['queries']['source'] and values['source']==prior['IoU']['source']
  tasks.append({'row':f'R{i+1:05d}','id_PRIVATE':sid,'platform':row['platform'],'scene_PRIVATE':row['scene'],'part':'DEV336','split':split[sid],'queries':q,'IoU':values,'statuses':st['arm_statuses']})
 assert len(tasks)==336;summary,scenes=summarize(tasks,arms,p['reference_arms'],336);negative=[]
 for t in tasks:
  for arm,ref in itertools.product(arms,repeat=2):
   d=t['IoU'][arm]-t['IoU'][ref]
   if arm!=ref and d< -1e-10:negative.append({'row':t['row'],'part':t['part'],'split':t['split'],'platform':t['platform'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
 for name,value in [('ALL_TASK_ROWS_PRIVATE',tasks),('TASK_SUMMARY',summary),('ALL_SCENE_SUMMARIES_PRIVATE',scenes),('ALL_ORDERED_NEGATIVE_COMPARISONS',negative),('SHARED_CALIBRATION_SELECTION',None)]:dump(out/(name+'.json'),value)
 receipt={'state':'PASS_R64_'+p['mode'].upper()+'_POSTSEAL_ORIGINAL_CANDIDATE_SAVED_IOU_SCORE','n':336,'arm_names':arms,'task_values':len(arms)*336,'new_GT_scored_task_values':0,'reused_saved_task_values':len(arms)*336,'ordered_negative_comparisons':len(negative),'single_GT_scoring_pass':True,'new_raw_GT_or_geometry_scoring_calls':0,'unknown_fallback_and_no_update_rows_remain_full_denominator':True,'prediction_seal_sha256':sha(pred/'SEALED.json'),'independent_audit_sha256':sha(ROOT/p['independent_audit']),'program_sha256':sha(__file__),'plan_sha256':sha(a.plan),'seconds':time.perf_counter()-start};dump(out/'SCORING_RECEIPT.json',receipt);seal(out,receipt['state']);print(receipt)
if __name__=='__main__':main()
