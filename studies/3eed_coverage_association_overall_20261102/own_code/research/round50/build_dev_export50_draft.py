"""Local-only DEV export helper, invoked by a later root-authorized publisher.

No task evaluation, model, media, GT or network operations. All inputs are
saved DEV scalars/receipts, source code, and the generated DEV narrative.
"""
from pathlib import Path
import csv,hashlib,io,json
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round50';S=H/'independent_score/dev336_attempt001';A=H/'independent_review'
ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
OWN_CODE=['crossplatform/round50/pool_weights50.py','crossplatform/round50/prepare50.py','crossplatform/round50/visual50.py','crossplatform/round50/predict50.py','research/round50/independent_review/audit_prepared50.py','research/round50/independent_review/audit_predictions50.py','research/round50/independent_review/score_dev50.py','research/round50/independent_review/audit_saved_dev50.py','research/round50/independent_review/diagnose_prediction_numeric50.py','research/round50/independent_review/source_metadata_inventory50.py','research/round50/finalize_dev50_draft.py','research/round50/build_dev_export50_draft.py']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def text(p,s):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def safe(s):return s.replace(str(ROOT),'WORKSPACE_ROOT').replace(str(ROOT.parent),'LOCAL_HOME')
def csvfile(p,rows):
 assert rows;stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);text(p,stream.getvalue())
def project_receipt(x):
 return {k:v for k,v in x.items() if v is None or type(v) in (int,float,bool) or (k=='state' and isinstance(v,str))}
def build_dev(out,expected_input_sha256,include_narratives=True):
 out=Path(out);assert out.exists() and not (out/'dev').exists()
 # Root supplies exact pins for every requested input; a missing future file
 # cannot be silently replaced with an earlier version or inferred result.
 def get(rel):
  assert rel in expected_input_sha256 and 'round46' not in rel and 'FINAL' not in rel and sha(ROOT/rel)==expected_input_sha256[rel]
  return read(ROOT/rel)
 sr='research/round50/independent_score/dev336_attempt001/'
 seal=get(sr+'SEALED.json')
 for rel,h in seal['files'].items():assert sha(S/rel)==h
 rows=get(sr+'ALL336_TASK_ROWS_PRIVATE.json');assert len(rows)==336
 scene_names={pl:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==pl}) for pl in ('drone','quad')};mapping={r['row']:f'DEV{i+1:04d}' for i,r in enumerate(rows)}
 def scene(pl,private):return pl+'_scene'+str(scene_names[pl].index(private)+1).zfill(2)
 public=[]
 for r in rows:
  assert set(r['IoU'])==set(r['queries'])==set(ARMS)
  for arm in ARMS:public.append({'row':mapping[r['row']],'split':r['split'],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'arm':arm,'IoU':r['IoU'][arm],'changed_vs_source':r['queries'][arm]!=r['queries']['source'],'changed_vs_old_G':r['queries'][arm]!=r['queries']['old_G']})
 assert len(public)==2352;csvfile(out/'dev/results/ALL2352_TASK_SCALARS.csv',public)
 neg=get(sr+'ALL_ORDERED_NEGATIVE_COMPARISONS.json');assert len(neg)==1758
 csvfile(out/'dev/results/ALL1758_ORDERED_NEGATIVE_COMPARISONS.csv',[{**{k:v for k,v in r.items() if k!='row'},'row':mapping[r['row']]} for r in neg])
 scenes=get(sr+'ALL_SCENE_SUMMARIES_PRIVATE.json');sc=[]
 for r in scenes:
  for arm in ARMS:sc.append({'split':r['part'],'platform':r['platform'],'scene_alias':scene(r['platform'],r['scene_PRIVATE']),'n':r['n'],'arm':arm,'mIoU_percent':r['mIoU_percent'][arm]})
 assert len(sc)==140;csvfile(out/'dev/results/ALL_SCENE_ARM_SCALARS.csv',sc)
 for name in ('TASK_SUMMARY.json','FACTORIAL_CONTRASTS.json','DEV_SELECTION_AND_FULL_GATE.json'):dump(out/'dev/results'/name,get(sr+name))
 pr=get('research/round50/independent_review/prepared_audit_attempt001/ANONYMOUS_ROWS.json')
 assert len(pr)==336
 fields=['platform','old_pool_count','wide_pool_count','new_masks','new_taxonomy_compatibility_recovered','all_checks_PASS']
 dump(out/'dev/results/ALL336_ENTRANCE_AND_TAXONOMY_FLAGS.json',[{'row':f'DEV{i+1:04d}',**{k:r[k] for k in fields}} for i,r in enumerate(pr)])
 for rel,label in [('research/round50/independent_review/prepared_audit_attempt001/AUDIT_RESULT.json','PREP'),('research/round50/independent_review/prediction_audit_attempt002/AUDIT_RESULT.json','PREDICTION'),('research/round50/independent_scalar_recheck/dev336_attempt001/AUDIT_RESULT.json','SAVED_SCALAR'),(sr+'SCORING_RECEIPT.json','SCORE'),('crossplatform/round50/visual336_attempt001/COUNTS.json','VISUAL_COSTS')]:dump(out/'dev/audit'/(label+'.json'),project_receipt(get(rel)))
 if include_narratives:
  for rel,dest in [('research/round50/REPORT_DEV.md','dev/REPORT.md'),('research/round50/DECISION_DEV.json','dev/DECISION.json')]:
   assert rel in expected_input_sha256 and sha(ROOT/rel)==expected_input_sha256[rel];text(out/dest,safe((ROOT/rel).read_text()))
 for rel in OWN_CODE:
  assert rel in expected_input_sha256 and sha(ROOT/rel)==expected_input_sha256[rel];text(out/'own_code'/rel,safe((ROOT/rel).read_text()))
 runtime=get('research/round50/independent_review/VISUAL_RUNTIME_STATIC_ADDENDUM.json')
 dump(out/'dev/history/RUNTIME_FAILURE_SUMMARY.json',{'state':runtime['state'],'failures':runtime['failed_attempts_preserved'],'rejected_started_attempts_additional_to_valid4956':True,'old_model_prompt_crops_unchanged':runtime['no_prompt_crop_modelweights_formula_or_GT_selection_changed'],'runtime_returned_to_original_registered_interpreter':True,'private_callbindings_logs_and_thirdparty_dependency_snapshot_not_exported':True})
 repair=get('research/round50/independent_review/prediction_audit_preexecution_revision002/REPAIR_CONTRACT.json')
 allowed=['state','row1_all_variants_bitexact','row2_only_wideentry_newW_k3_k10_C_layout_scores_differed','maximum_float64_score_difference','recreated_original_allocation_and_explicit_F_bitexact','source_oldG_wideoldW_and_oldentry_newW_controls_bitexact','repair','tolerance_weakened','author_predictions_or_method_changed','GT_read_or_score_run','old_output_and_code_and_grant_and_console_log_preserved']
 dump(out/'dev/history/INDEPENDENT_AUDITOR_LAYOUT_REPAIR.json',{k:repair[k] for k in allowed})
 for rel,label in [('research/round50/preexecution_revision001/pool_weights50.py','naive_W_never_executed.py'),('research/round50/independent_review/prediction_audit_preexecution_revision002/audit_predictions50.py','failed_independent_auditor_original.py')]:
  assert rel in expected_input_sha256 and sha(ROOT/rel)==expected_input_sha256[rel];text(out/'dev/history/own_code'/label,safe((ROOT/rel).read_text()))
 dump(out/'dev/SCHEMA.json',{'scope':'old exposed336,seven fixed configurations; shared k selected withDEV labels; full/TTA results not invented','task_scalars':2352,'ordered_negative_comparisons':1758,'scene_arm_rows':140,'entrance_scalar_rows':336,'no_raw_IDs_scene_names_captions_queries_masks_coordinates_vectors_features_or_weights':True,'only_saved_scalars_no_new_GT_evaluation':True,'future_full_candidate':'wideentry_oldW','future_full_candidate_has_no_k':True})
 return {'task_values':2352,'negative_comparisons':1758,'scene_arm_rows':140,'entrance_rows':336,'new_files':len([p for p in (out/'dev').rglob('*') if p.is_file()])}
