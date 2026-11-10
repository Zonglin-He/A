"""Independent saved-only reaggregation and old-candidate-IoU lookup verification."""
import argparse,sys,itertools,time
from common59 import *
sys.path.insert(0,str(ROOT/'research/round55'));from audit_saved_scalars55 import aggregate,check_tree

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);start=time.perf_counter();root=ROOT/p['score_root'];ss=read(root/'SEALED.json');old=ROOT/p['saved_label_root'];os=read(old/'SEALED.json');rows=read(ROOT/p['manifest']);splits=read(ROOT/p['split_by_id'])
 def member(base,seal,name):assert sha(base/name)==seal['files'][name];return read(base/name)
 tasks=member(root,ss,'ALL_TASK_ROWS_PRIVATE.json');labels=member(old,os,'ALL_CANDIDATE_IOUS_PRIVATE.json');oldtasks=member(old,os,'ALL_TASK_ROWS_PRIVATE.json');receipt=member(root,ss,'SCORING_RECEIPT.json');assert receipt['new_GT_scored_task_values']==0 and receipt['task_values']==len(ARMS)*336==receipt['reused_saved_task_values'];assert receipt['program_sha256']==p['scorer_sha256']
 for i,(row,t,oldrow,label) in enumerate(zip(rows,tasks,oldtasks,labels)):
  assert t['id_PRIVATE']==label['id_PRIVATE']==oldrow['id_PRIVATE']==row['id'];assert t['platform']==row['platform']==oldrow['platform'];assert t['scene_PRIVATE']==row['scene']==oldrow['scene_PRIVATE'] and t['row']==f'R{i+1:05d}' and t['split']==splits[row['id']];assert set(t['IoU'])==set(t['queries'])==set(t['statuses'])==set(ARMS)
  for arm in ARMS:assert t['IoU'][arm]==label['candidate_IoUs_PRIVATE'][t['queries'][arm]]
  assert t['queries']['source']==oldrow['queries']['source'] and t['IoU']['source']==oldrow['IoU']['source']
 summary,scenes=aggregate(tasks,ARMS,REFS,336);err=check_tree(summary,member(root,ss,'TASK_SUMMARY.json'));sceneerr=check_tree(scenes,member(root,ss,'ALL_SCENE_SUMMARIES_PRIVATE.json'));expected=[]
 for t in tasks:
  for arm,ref in itertools.product(ARMS,repeat=2):
   d=t['IoU'][arm]-t['IoU'][ref]
   if arm!=ref and d< -1e-10:expected.append({'row':t['row'],'part':t['part'],'split':t['split'],'platform':t['platform'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
 assert expected==member(root,ss,'ALL_ORDERED_NEGATIVE_COMPARISONS.json')
 out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True);res={'state':'PASS_R59_INDEPENDENT_SAVED_LABEL_VALUES_AGGREGATES_SCENES_NEGATIVES_AUDIT','n':336,'arm_names':ARMS,'task_values':len(ARMS)*336,'all_values_exact_saved_candidate_IoU_reuse':True,'ordered_negative_comparisons_exact':len(expected),'independent_summary_max_abs_error_pp':err,'independent_scene_max_abs_error_pp':sceneerr,'calibration_recomputed':False,'new_GT_geometry_or_models_or_raw_media':False,'source_saved_label_seal_sha256':sha(old/'SEALED.json'),'scoring_seal_sha256':sha(root/'SEALED.json'),'seconds':time.perf_counter()-start,'program_sha256':sha(__file__),'plan_sha256':sha(a.plan)};dump(out/'AUDIT_RESULT.json',res);print(res,flush=True)
if __name__=='__main__':main()
