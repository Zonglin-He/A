"""One authorized post-seal R50 old336 task score, all seven configurations."""
from pathlib import Path
import sys,json,hashlib,time,collections,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment';OUT=ROOT/'research/round50/independent_score/dev336_attempt001';PRED=ROOT/'crossplatform/round50/predicted336_attempt001'
ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
sys.path.insert(0,str(ROOT/'research/round5/audit'))
from independent_core_v2 import candidate_ious,transform_gt
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def scoring_index(rows):
 import ijson
 path=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json';receipt=read(ROOT/'crossplatform/round5/baseline_engineering/PREPARATION_RECEIPT.json');assert sha(path)==receipt['scoring_index_sha256'];wanted={r['id']:r for r in rows};out={}
 with path.open('rb') as f:
  for entry in ijson.items(f,'item'):
   if entry['id'] not in wanted:continue
   r=wanted[entry['id']];assert entry['platform']==r['platform'] and entry['scene']==r['scene'] and Path(entry['meta_path']).resolve()==Path(r['metapath']).resolve() and type(entry['object_index']) is int and entry['object_index']>=0
   out[r['id']]={k:entry[k] for k in ('meta_path','object_index','metadata_container_sha256')}
 assert set(out)==set(wanted);return out
def summarize(tasks):
 out={};scenes=[]
 for part in ('ALL336','D208','V64','M64'):
  rr=tasks if part=='ALL336' else [r for r in tasks if r['split']==part];ss={}
  for pl in ('drone','quad'):
   pr=[r for r in rr if r['platform']==pl];ss[pl]={}
   for arm in ARMS:
    vv=np.array([r['IoU'][arm] for r in pr]);source=np.array([r['IoU']['source'] for r in pr]);old=np.array([r['IoU']['old_G'] for r in pr]);ds=vv-source;do=vv-old
    ss[pl][arm]={'n':len(pr),'mIoU_percent':float(vv.mean()*100),'Acc25_percent':float((vv>.25).mean()*100),'Acc50_percent':float((vv>.5).mean()*100),'gain_vs_source_pp':float(ds.mean()*100),'gain_vs_old_G_pp':float(do.mean()*100),'improved_vs_source':int((ds>1e-10).sum()),'worsened_vs_source':int((ds< -1e-10).sum()),'improved_vs_old_G':int((do>1e-10).sum()),'worsened_vs_old_G':int((do< -1e-10).sum()),'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in pr),'query_changes_vs_old_G':sum(r['queries'][arm]!=r['queries']['old_G'] for r in pr),'scene_clusters':len({r['scene_PRIVATE'] for r in pr})}
   for scene in sorted({r['scene_PRIVATE'] for r in pr}):
    sr=[r for r in pr if r['scene_PRIVATE']==scene];scenes.append({'part':part,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),'mIoU_percent':{arm:float(np.mean([r['IoU'][arm] for r in sr])*100) for arm in ARMS}})
  ss['balanced_mIoU_percent']={arm:float(np.mean([ss[pl][arm]['mIoU_percent'] for pl in ('drone','quad')])) for arm in ARMS};out[part]=ss
 return out,scenes
def main():
 start=time.perf_counter();auth=read(EXP/'ROOT_SINGLE_DEV_SCORING_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R50_SINGLE_POSTSEAL_OLD336_DEV_TASK_SCORE'
 assert auth['program_sha256']==sha(__file__) and auth['prediction_seal_sha256']==sha(PRED/'SEALED.json') and auth['scoring_plan_sha256']==sha(EXP/'DEV_SCORING_PLAN_LOCK.json')
 plan=read(EXP/'DEV_SCORING_PLAN_LOCK.json');assert plan['arms']==ARMS and plan['n']==336 and plan['selection_rule']=='shared_k_max_wideentry_newW_balanced_mIoU_tie_1e-10pp_k10_then_best_three_candidate_same_tie_priority'
 assert plan['candidate_tie_priority']==['wideentry_oldW','oldentry_newW','wideentry_newW']
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert 'round46' not in rel and 'FINAL' not in rel and sha(ROOT/rel)==h
 for field in ('prep_audit','prediction_audit'):
  fp=ROOT/plan[field];assert auth[field+'_sha256']==sha(fp) and read(fp)['state'].startswith('PASS_')
 seal=read(PRED/'SEALED.json')
 for rel,h in seal['files'].items():assert sha(PRED/rel)==h
 assert not OUT.exists();OUT.mkdir(parents=True);rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');assert len(rows)==336 and len({r['id'] for r in rows})==336
 split={r['id']:p for p in ('D208','V64','M64') for r in read(ROOT/'research/round24/input'/(p+'_PRIVATE.json'))};assert set(split)=={r['id'] for r in rows}
 idx=scoring_index(rows);oldtasks={r['id_PRIVATE']:r for r in read(ROOT/'research/round49/independent_audit/ALL336_TASK_ROWS_PRIVATE.json')};containers={};tasks=[]
 try:
  for i,r in enumerate(rows):
   z=load(PRED/'samples_PRIVATE'/(r['id']+'.npz'));assert z['arm_names'].tolist()==ARMS and z['queries'].shape==(7,) and z['selected_original_boxes'].shape==(7,6)
   assert np.array_equal(z['selected_original_boxes'],z['source_boxes'][z['queries']]);ix=idx[r['id']];mp=ix['meta_path']
   if mp not in containers:assert sha(mp)==ix['metadata_container_sha256'];containers[mp]=read(mp)
   obj=containers[mp]['ground_info'][ix['object_index']];assert ' '.join(obj['caption'].replace(',',' ,').split())==r['caption'];gt=transform_gt(obj['bbox_3d'],r['platform']);boxes=z['selected_original_boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);values=candidate_ious(boxes,gt)[0]
   assert abs(float(values[0])-oldtasks[r['id']]['IoU']['legacy_V0P0R0'])<1e-12 and abs(float(values[1])-oldtasks[r['id']]['IoU']['legacy_V1P1R1'])<1e-12
   tasks.append({'row':f'D{i+1:04d}','id_PRIVATE':r['id'],'platform':r['platform'],'split':split[r['id']],'scene_PRIVATE':r['scene'],'queries':dict(zip(ARMS,map(int,z['queries']))),'IoU':dict(zip(ARMS,map(float,values)))})
 except Exception as exc:
  dump(OUT/'PARTIAL_ROWS_PRIVATE.json',tasks);dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':len(tasks),'no_automatic_GT_rerun':True});raise
 dump(OUT/'ALL336_TASK_ROWS_PRIVATE.json',tasks);summary,scenes=summarize(tasks);neg=[]
 for row in tasks:
  for arm,ref in itertools.product(ARMS,repeat=2):
   d=row['IoU'][arm]-row['IoU'][ref]
   if arm!=ref and d< -1e-10:neg.append({'row':row['row'],'platform':row['platform'],'split':row['split'],'arm':arm,'reference':ref,'delta_pp':float(d*100)})
 m=summary['ALL336']['balanced_mIoU_percent'];k=3 if m['wideentry_newW_k3']>m['wideentry_newW_k10']+1e-10 else 10
 selected={'wideentry_oldW':'wideentry_oldW','oldentry_newW':f'oldentry_newW_k{k}','wideentry_newW':f'wideentry_newW_k{k}'}
 maximum=max(m[selected[name]] for name in plan['candidate_tie_priority']);candidate=next(name for name in plan['candidate_tie_priority'] if maximum-m[selected[name]]<=1e-10);winner=selected[candidate];gain=m[winner]-m['old_G'];main=['source','old_G','wideentry_oldW',f'oldentry_newW_k{k}',f'wideentry_newW_k{k}']
 selection={'state':'R50_DEV_LABEL_SELECTED_SHARED_K_AND_CANDIDATE_ALL_TRIALS_RETAINED','selected_shared_k':k,'k_primary_target':'wideentry_newW_ALL336_balanced_mIoU','k_tie_priority':10,'numeric_tie_tolerance_pp':1e-10,'candidate_tie_priority':plan['candidate_tie_priority'],'selected_candidate_family':candidate,'selected_candidate_arm':winner,'selected_candidate_balanced_mIoU_percent':m[winner],'selected_gain_vs_old_G_pp':gain,'selected_gain_vs_source_pp':m[winner]-m['source'],'full_gate_positive_balanced_net_vs_old_G':bool(gain>1e-10),'no_zero_harm_or_per_case_or_blind_proxy_gate':True,'main_five_arms':main,'all_seven_unique_trials_reported':True,'all_labels_used_only_after_predictions_and_audits_sealed':True,'full_execution_requires_new_root_lock_and_grant':True}
 effects={}
 for part,ss in summary.items():
  effects[part]={}
  for pl in ('drone','quad','balanced'):
   mm=ss['balanced_mIoU_percent'] if pl=='balanced' else {a:ss[pl][a]['mIoU_percent'] for a in ARMS};base=mm['old_G'];d={'coverage_oldW_minus_old_G_pp':mm['wideentry_oldW']-base}
   for kk in (3,10):
    o=mm[f'oldentry_newW_k{kk}'];w=mm[f'wideentry_newW_k{kk}'];d.update({f'newW_oldentry_k{kk}_minus_old_G_pp':o-base,f'newW_wideentry_k{kk}_minus_wide_oldW_pp':w-mm['wideentry_oldW'],f'coverage_newW_k{kk}_pp':w-o,f'entry_W_interaction_k{kk}_pp':w-o-mm['wideentry_oldW']+base})
   effects[part][pl]=d
 dump(OUT/'TASK_SUMMARY.json',summary);dump(OUT/'ALL_SCENE_SUMMARIES_PRIVATE.json',scenes);dump(OUT/'ALL_ORDERED_NEGATIVE_COMPARISONS.json',neg);dump(OUT/'FACTORIAL_CONTRASTS.json',effects);dump(OUT/'DEV_SELECTION_AND_FULL_GATE.json',selection)
 receipt={'state':'PASS_R50_SINGLE_OLD336_SEVEN_CONFIGURATION_TASK_SCORE','n':336,'arms':7,'task_values':2352,'ordered_negative_comparisons':len(neg),'scene_summary_rows':len(scenes),'source_and_oldG_IoU_reproduced_n':336,'selected_shared_k':k,'full_gate_PASS':bool(gain>1e-10),'no_models_or_parameter_or_box_change_or_protected_FINAL_access':True,'only_one_GT_scoring_pass':True,'program_sha256':sha(__file__),'prediction_seal_sha256':sha(PRED/'SEALED.json'),'seconds':time.perf_counter()-start};dump(OUT/'SCORING_RECEIPT.json',receipt);dump(OUT/'SEALED.json',{'state':receipt['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'selection_is_dev_label_based_not_independent_confirmation':True});print(json.dumps({'receipt':receipt,'selection':selection,'ALL336_balanced_mIoU':m},ensure_ascii=False))
if __name__=='__main__':main()
