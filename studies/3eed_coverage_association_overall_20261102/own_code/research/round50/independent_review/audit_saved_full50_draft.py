"""UNEXECUTED independent reaggregation of sealed full5912 saved scalars.

Stdlib only. No author score/evaluator import, NumPy, GT or metadata file,
manifest, media, point cloud, prediction vectors, model or network access.
The only quality observations are the already saved 17736 IoU scalars.
"""
from pathlib import Path
import collections,hashlib,itertools,json,math,statistics,time

ROOT=Path(__file__).resolve().parents[3];EXP=ROOT/'research/round50/experiment'
PLAN=EXP/'FULL_SAVED_SCALAR_AUDIT_PLAN_LOCK.json'
AUTH=EXP/'ROOT_INDEPENDENT_FULL_SAVED_SCALAR_AUDIT_AUTHORIZATION.json'
SCORE=ROOT/'research/round50/independent_score/full5912_attempt001'
OUT=ROOT/'research/round50/independent_scalar_recheck/full5912_attempt001'
GATE=EXP/'FULL_GATE_AND_TTA_ADVANCE_BEFORE_FULL_GT.json'
FREEZE=EXP/'FULL_METHOD_FREEZE_BEFORE_FULL_MEDIA.json'
DEV_SUMMARY=ROOT/'research/round50/independent_score/dev336_attempt001/TASK_SUMMARY.json'
ARMS=['source','old_G','wideentry_oldW'];PARTS=['ALL5912','DEV336','NONDEV5576']
FILES=['ALL5912_TASK_ROWS_PRIVATE.json','TASK_SUMMARY.json','ALL_SCENE_SUMMARIES_PRIVATE.json',
 'ALL_ORDERED_NEGATIVE_COMPARISONS.json','SCORING_RECEIPT.json']
ERROR=0.;COMPARISONS=0;READS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);READS.append(p.relative_to(ROOT).as_posix())
 return json.loads(p.read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def same(a,b,path='$'):
 global ERROR,COMPARISONS
 if isinstance(a,dict):
  assert isinstance(b,dict) and set(a)==set(b),(path,'keys')
  for k in a:same(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  assert isinstance(b,list) and len(a)==len(b),(path,'length')
  for j,(x,y) in enumerate(zip(a,b)):same(x,y,path+f'[{j}]')
 elif type(a) in (int,float):
  assert type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),path
  delta=abs(a-b);ERROR=max(ERROR,delta);COMPARISONS+=1
  assert delta<=1e-10,(path,a,b,delta)
 else:assert type(a) is type(b) and a==b,(path,a,b)
def mean(x):return statistics.fmean(x)
def aggregate(tasks):
 summary={};scenes=[];effects={}
 for part in PARTS:
  rows=tasks if part=='ALL5912' else [r for r in tasks if r['part']==part]
  assert len(rows)=={'ALL5912':5912,'DEV336':336,'NONDEV5576':5576}[part]
  platforms={};paired={}
  for pl in ('drone','quad'):
   rr=[r for r in rows if r['platform']==pl];assert rr
   platforms[pl]={}
   for arm in ARMS:
    values=[r['IoU'][arm] for r in rr]
    source_delta=[r['IoU'][arm]-r['IoU']['source'] for r in rr]
    old_delta=[r['IoU'][arm]-r['IoU']['old_G'] for r in rr]
    platforms[pl][arm]={'n':len(rr),'mIoU_percent':mean(values)*100,
     'Acc25_percent':sum(v>.25 for v in values)/len(values)*100,
     'Acc50_percent':sum(v>.5 for v in values)/len(values)*100,
     'gain_vs_source_pp':mean(source_delta)*100,'gain_vs_old_G_pp':mean(old_delta)*100,
     'improved_vs_source':sum(x>1e-10 for x in source_delta),
     'worsened_vs_source':sum(x< -1e-10 for x in source_delta),
     'improved_vs_old_G':sum(x>1e-10 for x in old_delta),
     'worsened_vs_old_G':sum(x< -1e-10 for x in old_delta),
     'query_changes_vs_source':sum(r['queries'][arm]!=r['queries']['source'] for r in rr),
     'query_changes_vs_old_G':sum(r['queries'][arm]!=r['queries']['old_G'] for r in rr),
     'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
   for scene in sorted({r['scene_PRIVATE'] for r in rr}):
    sr=[r for r in rr if r['scene_PRIVATE']==scene]
    scenes.append({'part':part,'platform':pl,'scene_PRIVATE':scene,'n':len(sr),
     'mIoU_percent':{arm:mean(r['IoU'][arm] for r in sr)*100 for arm in ARMS}})
  platforms['balanced_mIoU_percent']={arm:mean(platforms[pl][arm]['mIoU_percent'] for pl in ('drone','quad')) for arm in ARMS}
  platforms['micro_mIoU_percent']={arm:mean(r['IoU'][arm] for r in rows)*100 for arm in ARMS}
  summary[part]=platforms
  for scope in ('drone','quad','balanced','micro'):
   m=(platforms[scope+'_mIoU_percent'] if scope in ('balanced','micro') else
      {arm:platforms[scope][arm]['mIoU_percent'] for arm in ARMS})
   paired[scope]={'wideentry_oldW_minus_old_G_pp':m['wideentry_oldW']-m['old_G'],
    'wideentry_oldW_minus_source_pp':m['wideentry_oldW']-m['source'],
    'old_G_minus_source_pp':m['old_G']-m['source']}
  effects[part]=paired
 negatives=[]
 for r in tasks:
  for arm,reference in itertools.permutations(ARMS,2):
   delta=r['IoU'][arm]-r['IoU'][reference]
   if delta< -1e-10:negatives.append({'row':r['row'],'part':r['part'],'platform':r['platform'],
    'arm':arm,'reference':reference,'delta_pp':delta*100})
 return summary,scenes,negatives,effects

def main():
 start=time.perf_counter();auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_SAVED_FULL_SCALAR_REAGGREGATION_NO_GT'
 assert auth['program_sha256']==sha(__file__) and auth['audit_plan_sha256']==sha(PLAN)
 assert plan['n']==5912 and plan['arms']==ARMS and plan['saved_IoU_only_no_GT_or_new_evaluation'] is True
 assert plan['score_root']==SCORE.relative_to(ROOT).as_posix()
 assert plan['full_gate_lock']==GATE.relative_to(ROOT).as_posix()
 assert plan['output']==OUT.relative_to(ROOT).as_posix()
 allowed=[SCORE/'SEALED.json',*[SCORE/n for n in FILES],GATE,FREEZE,DEV_SUMMARY]
 assert set(plan['inputs_sha256'])=={p.relative_to(ROOT).as_posix() for p in allowed}
 for p in allowed:assert sha(p)==plan['inputs_sha256'][p.relative_to(ROOT).as_posix()]
 assert auth['score_seal_sha256']==sha(SCORE/'SEALED.json')
 gate=read(GATE);freeze=read(FREEZE)
 assert gate['state']=='ROOT_LOCKED_R50_FULL_ADVANCE_RULE_BEFORE_ANY_FULL_GT'
 assert gate['scientific_freeze_sha256']==sha(FREEZE)
 rule=gate['registered_full_gate'];assert rule=={'candidate':'wideentry_oldW','scope':'ALL5912',
  'metric':'two_platform_equal_weight_mIoU_percent','references':['old_G','source'],'strict_minimum_gain_pp':1e-10}
 assert gate['platform_or_case_zero_harm_gate'] is False and gate['no_new_configuration_selection_after_full_GT'] is True
 assert freeze['selected_configuration']=='wideentry_oldW' and freeze['comparators']==ARMS
 saved=read(SCORE/'SEALED.json');assert set(saved['files'])==set(FILES)
 for name,h in saved['files'].items():assert sha(SCORE/name)==h
 tasks=read(SCORE/'ALL5912_TASK_ROWS_PRIVATE.json')
 assert len(tasks)==5912 and len({r['row'] for r in tasks})==len({r['id_PRIVATE'] for r in tasks})==5912
 assert collections.Counter(r['platform'] for r in tasks)=={'drone':2984,'quad':2928}
 assert collections.Counter(r['part'] for r in tasks)=={'DEV336':336,'NONDEV5576':5576}
 assert {pl:len({r['scene_PRIVATE'] for r in tasks if r['platform']==pl}) for pl in ('drone','quad')}=={'drone':2,'quad':3}
 for r in tasks:
  assert r['platform'] in ('drone','quad') and r['part'] in ('DEV336','NONDEV5576')
  assert set(r['IoU'])==set(r['queries'])==set(ARMS)
  assert all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1 for v in r['IoU'].values())
  assert all(type(q) is int and 0<=q<256 for q in r['queries'].values())
 dev=[r for r in tasks if r['part']=='DEV336']
 assert collections.Counter(r['platform'] for r in dev)=={'drone':168,'quad':168}
 assert not OUT.exists();OUT.mkdir(parents=True)
 try:
  summary,scenes,negative,effects=aggregate(tasks)
  assert len(scenes)==15
  stored_summary=read(SCORE/'TASK_SUMMARY.json')
  same(summary,stored_summary,'TASK_SUMMARY')
  same(scenes,read(SCORE/'ALL_SCENE_SUMMARIES_PRIVATE.json'),'SCENES')
  same(negative,read(SCORE/'ALL_ORDERED_NEGATIVE_COMPARISONS.json'),'NEGATIVES')
  receipt=read(SCORE/'SCORING_RECEIPT.json')
  assert receipt['state']=='PASS_R50_SINGLE_LOCKED_FULL5912_THREE_ARM_TASK_SCORE'
  assert receipt['n']==5912 and receipt['arms']==3 and receipt['task_values']==17736
  assert receipt['ordered_negative_comparisons']==len(negative) and receipt['DEV336_all_three_IoUs_reproduced']==336
  assert receipt['DEV336_metadata_seed_equal_n']+receipt['DEV336_metadata_seed_mismatch_n']==336
  assert receipt['source_boxes_not_refined'] and receipt['no_k_newrho_or_TTA_in_full_method'] and receipt['no_new_method_selection_after_full_labels']
  old=read(DEV_SUMMARY)
  # Full-to-DEV exact path/caption identity was already audited before score.
  # This check uses only saved aggregate values, never assumes full/old IDs
  # match and never reopens either source cohort or its GT metadata.
  for pl in ('drone','quad'):
   for arm in ARMS:same(summary['DEV336'][pl][arm],old['ALL336'][pl][arm],'DEV_BRIDGE.'+pl+'.'+arm)
  same(summary['DEV336']['balanced_mIoU_percent'],{a:old['ALL336']['balanced_mIoU_percent'][a] for a in ARMS},'DEV_BALANCED_BRIDGE')
  m=summary['ALL5912']['balanced_mIoU_percent'];gain={ref:m['wideentry_oldW']-m[ref] for ref in rule['references']}
  full_gate=all(g>rule['strict_minimum_gain_pp'] for g in gain.values())
  saved_m=stored_summary['ALL5912']['balanced_mIoU_percent']
  assert full_gate==all(saved_m['wideentry_oldW']-saved_m[ref]>rule['strict_minimum_gain_pp'] for ref in rule['references']), 'Gate boundary differs between independent fmean and saved NumPy reduction; do not choose the more favorable rounding'
  decision={'state':'R50_FULL_PRELOCKED_ADVANCE_GATE_INDEPENDENTLY_RECOMPUTED_FROM_SAVED_SCALARS',
   'selected_candidate':'wideentry_oldW','selected_candidate_has_no_k':True,'scope':'ALL5912',
   'metric':rule['metric'],'references':rule['references'],'strict_minimum_gain_pp':rule['strict_minimum_gain_pp'],
   'candidate_balanced_mIoU_percent':m['wideentry_oldW'],'gains_vs_references_pp':gain,'full_gate_PASS':full_gate,
   'full_score_seal_sha256':sha(SCORE/'SEALED.json'),'registered_gate_lock_sha256':sha(GATE),
   'no_platform_scene_Acc_nonDEV_or_individual_negative_hard_gate':True,
   'no_new_method_or_k_selection_after_full_GT':True,
   'gate_predicate_agrees_between_independent_fmean_and_original_numpy_reduction':True,
   'gate_PASS_only_permits_separate_locked_DEV_TTA_not_evidence_TTA_already_effective':True}
  dump(OUT/'RECOMPUTED_TASK_SUMMARY.json',summary)
  dump(OUT/'RECOMPUTED_SCENE_SUMMARIES_PRIVATE.json',scenes)
  dump(OUT/'RECOMPUTED_FROZEN_PAIRWISE_DELTAS.json',effects)
  dump(OUT/'REGISTERED_FULL_GATE_RECHECK.json',decision)
  negative_counts={pl:sum(r['platform']==pl for r in negative) for pl in ('drone','quad')}
  for part in PARTS:
   expected=tasks if part=='ALL5912' else [r for r in tasks if r['part']==part]
   assert summary[part]['drone']['source']['n']+summary[part]['quad']['source']['n']==len(expected)
 except BaseException as exc:
  dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),
   'no_GT_or_evaluator_import_or_new_IoU':True,'no_automatic_scoring_rerun':True});raise
 permitted_reads={AUTH.relative_to(ROOT).as_posix(),PLAN.relative_to(ROOT).as_posix(),*[p.relative_to(ROOT).as_posix() for p in allowed]}
 assert set(READS)<=permitted_reads
 result={'state':'PASS_R50_INDEPENDENT_SAVED_FULL_SCALARS_AND_PRELOCKED_GATE_REAGGREGATION',
  'n':5912,'arms':3,'IoU_values_reaggregated':17736,'ordered_negative_comparisons_reproduced':len(negative),
  'ordered_negative_comparisons_by_platform':negative_counts,'scene_summary_rows_reproduced':len(scenes),
  'scene_arm_values_reproduced':len(scenes)*3,'known_scene_counts':{'drone':2,'quad':3},
  'scope_counts':{'ALL5912':5912,'DEV336':336,'NONDEV5576':5576},
  'numeric_comparisons':COMPARISONS,'max_abs_scalar_error':ERROR,
  'selected_candidate':'wideentry_oldW','full_gate_PASS':full_gate,'full_gate_references':['old_G','source'],
  'strict_minimum_gain_pp':1e-10,'DEV336_three_arm_summary_bridge_reproduced':True,
  'no_old_to_full_ID_or_alias_equality_assumed':True,'source_groups_not_independent_physical_instances':True,
  'historically_exposed_target_no_new_scene_independent_confirmation_or_CI_claim':True,
  'no_GT_metadata_manifest_media_predictions_model_network_or_IoU_recomputation':True,
  'no_author_scorer_or_geometry_import':True,'read_file_count':len(set(READS)),
  'score_seal_sha256':sha(SCORE/'SEALED.json'),'registered_full_gate_lock_sha256':sha(GATE),
  'program_sha256':sha(__file__),'audit_plan_sha256':sha(PLAN),'seconds':time.perf_counter()-start}
 dump(OUT/'AUDIT_RESULT.json',result)
 dump(OUT/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
 print(json.dumps(result))

if __name__=='__main__':main()
