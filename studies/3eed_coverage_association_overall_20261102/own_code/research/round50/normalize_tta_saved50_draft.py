"""UNEXECUTED saved-only TTA report adapter; no scoring/model/evaluator imports.

Consumes only the closed, root-pinned score/audit JSON and CSV set. Removes
raw IDs, scenes, queries and all tensors. It neither selects a method nor
recomputes IoU; all six arms, 2016 task scalars and 732 negatives are retained.
"""
from pathlib import Path
import argparse,hashlib,json,math,re,time

ROOT=Path(__file__).resolve().parents[2]
SCORE='research/round50/independent_score/episodic336_attempt001'
SAVED='research/round50/independent_scalar_recheck/episodic336_attempt001'
OUTPUT='research/round50/TTA_NORMALIZED_SAVED_RESULTS.json'
RECEIPT='research/round50/TTA_NORMALIZATION_RECEIPT.json'
ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
MODE='ACTUAL_NONZERO_EPISODIC_STATES_WITH_INSTALLED_CLOSURE'
FILES=['ALL336_TASK_ROWS_PRIVATE.json','TASK_SUMMARY.json','SCENE_SUMMARIES_PRIVATE.json',
 'ALL_ORDERED_NEGATIVE_COMPARISONS.json','DEV_STEP_SELECTION_AND_GATE.json',
 'ALL2016_ANONYMOUS_TASK_SCALARS.csv','ALL_ANONYMOUS_ORDERED_NEGATIVES.csv',
 'CONDITION_COUNTS.json','SCORING_RECEIPT.json']
BASE_INPUTS=[SCORE+'/SEALED.json',*[SCORE+'/'+n for n in FILES],SAVED+'/AUDIT_RESULT.json',SAVED+'/SEALED.json']
DENIED={'id','id_PRIVATE','alias','aliases','scene_PRIVATE','caption','subject_query','queries',
 'query_indices','query_index','point_indices','mask_indices','mask_index','source_boxes','source_scores',
 'selected_original_boxes','boxes_PRIVATE','box_PRIVATE','coordinates','features','tokens','h','weights',
 'imagepath','pointpath','metapath','raw','input_key','logical_binding','value_path'}


def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts
 assert p.as_posix().startswith('research/round50/') and 'FINAL' not in p.as_posix()
 return ROOT/p


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(rel):return json.loads(local(rel).read_text())


def clean(value):
 if isinstance(value,dict):
  assert not set(value)&DENIED
  for x in value.values():clean(x)
 elif isinstance(value,list):
  for x in value:clean(x)
 elif type(value) is float:assert math.isfinite(value)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--plan',required=True);p.add_argument('--authorization',required=True);args=p.parse_args()
 start=time.perf_counter();pp=Path(args.plan).resolve();ap=Path(args.authorization).resolve()
 assert pp.parent==ap.parent==ROOT/'research/round50/experiment'
 plan=json.loads(pp.read_text());auth=json.loads(ap.read_text())
 assert auth['state']=='ROOT_AUTHORIZED_R50_SAVED_ONLY_TTA_NORMALIZATION'
 assert auth['program_sha256']==sha(__file__) and auth['plan_sha256']==sha(pp)
 assert plan['n']==336 and plan['arm_names']==ARMS and plan['saved_only_no_new_GT_models_or_scoring'] is True
 assert plan['output']==OUTPUT and plan['measurement_mode']==MODE
 numeric_result=plan['independent_numeric_audit']
 assert numeric_result.startswith('research/round50/independent_review/') and Path(numeric_result).name=='AUDIT_RESULT.json'
 expected=BASE_INPUTS+[numeric_result]
 assert len(set(expected))==13 and set(plan['inputs_sha256'])==set(expected)
 for rel,h in plan['inputs_sha256'].items():assert sha(local(rel))==h,rel
 seal=read(SCORE+'/SEALED.json');assert set(seal['files'])==set(FILES)
 for name,h in seal['files'].items():assert sha(local(SCORE+'/'+name))==h
 scalar=read(SAVED+'/AUDIT_RESULT.json');ss=read(SAVED+'/SEALED.json')
 assert ss['files']['AUDIT_RESULT.json']==sha(local(SAVED+'/AUDIT_RESULT.json'))
 assert scalar['state']=='PASS_R50_INDEPENDENT_SAVED_EPISODIC_SCALARS_SELECTION_AND_MEASUREMENT_REAGGREGATION'
 assert scalar['n']==336 and scalar['arms']==6 and scalar['IoU_values_reaggregated']==2016
 assert scalar['score_seal_sha256']==sha(local(SCORE+'/SEALED.json'))
 assert scalar['independent_numeric_audit_result']==numeric_result
 assert scalar['independent_numeric_audit_sha256']==sha(local(numeric_result))
 numeric=read(numeric_result);condition=read(SCORE+'/CONDITION_COUNTS.json')
 gate=read(SCORE+'/DEV_STEP_SELECTION_AND_GATE.json');receipt=read(SCORE+'/SCORING_RECEIPT.json')
 assert numeric['state']=='PASS_R50_INDEPENDENT_EPISODIC_NUMERIC_AND_CONTRACT_AUDIT_NO_GT'
 assert numeric['n']==336 and numeric['arms']==6 and numeric['no_GT_read'] is True
 assert numeric['nonzero_parameter_element_count']>0 and numeric['installed_closure_requirement_satisfied'] is True
 assert numeric['actual_adapted_closure_verified'] is True and numeric['TTA_effect_actually_tested'] is True
 assert receipt['measurement_mode']==scalar['measurement_mode']==gate['measurement_mode']==MODE
 assert receipt['independent_audit_sha256']==sha(local(numeric_result))
 assert receipt['prediction_seal_sha256']==numeric['prediction_seal_sha256']
 assert receipt['task_values']==2016 and receipt['ordered_negative_comparisons']==732
 assert gate['gate_PASS']==scalar['gate_PASS']==receipt['gate_PASS']==False
 assert gate['selected_step']==scalar['selected_step'] and gate['selected_arm']==scalar['selected_arm']
 rows=read(SCORE+'/ALL336_TASK_ROWS_PRIVATE.json');summary=read(SCORE+'/TASK_SUMMARY.json')
 scenes=read(SCORE+'/SCENE_SUMMARIES_PRIVATE.json');neg=read(SCORE+'/ALL_ORDERED_NEGATIVE_COMPARISONS.json')
 assert len(rows)==336 and len({r['row'] for r in rows})==336 and len(neg)==732
 mapping={r['row']:f'TTA{j+1:04d}' for j,r in enumerate(rows)}
 scene_names={pl:sorted({r['scene_PRIVATE'] for r in rows if r['platform']==pl}) for pl in ('drone','quad')}
 assert {pl:len(x) for pl,x in scene_names.items()}=={'drone':2,'quad':3}
 scene_map={(pl,s):pl.upper()+'_SCENE_'+str(j+1).zfill(2) for pl,names in scene_names.items() for j,s in enumerate(names)}
 task=[]
 for row in rows:
  assert set(row['IoU'])==set(ARMS)
  for arm in ARMS:
   value=row['IoU'][arm];assert type(value) in (int,float) and math.isfinite(value) and 0<=value<=1
   task.append({'row':mapping[row['row']],'platform':row['platform'],'split':row['split'],
    'scene_alias':scene_map[row['platform'],row['scene_PRIVATE']], 'arm':arm,'IoU':value,
    'accepted':row['accepted'],'nonzero_s1':row['nonzero_parameter_updates']['1'],
    'nonzero_s3':row['nonzero_parameter_updates']['3']})
 negative=[{'row':mapping[r['row']],'platform':r['platform'],'split':r['split'],
   'arm':r['arm'],'reference':r['reference'],'delta_pp':r['delta_pp']} for r in neg]
 scene_rows=[{'split':r['split'],'platform':r['platform'],'scene_alias':scene_map[r['platform'],r['scene_PRIVATE']],
   'n':r['n'],'mIoU_percent':r['mIoU_percent']} for r in scenes]
 assert len(task)==2016 and len(negative)==732 and set(summary)=={'ALL336','D208','V64','M64'}
 numeric_fields=['nonzero_parameter_element_count','installed_closure_requirement_satisfied',
  'actual_adapted_closure_verified','TTA_effect_actually_tested','actual_adapted_closure_episodes',
  'actual_full_grounder_closure_forward_checks','all_saved_heads_source_bytes_identical','all_no_update_outputs_equal_s0']
 diagnostics={'condition_counts':condition,'numeric_audit':{k:numeric[k] for k in numeric_fields},
  'all_exposed_splits_summary':summary,'all336_is_primary_denominator':True,
  'B_is_used_in_current_input_fitting_not_unseen_validation':True,
  'gate_failed_STOP_no_full_TTA':True}
 assert condition['n']==336 and condition['accepted_n']==sum(r['accepted'] for r in rows)
 assert condition['rejected_n']==336-condition['accepted_n']
 value={'state':'COMPLETED_R50_DEV_EPISODIC_SAMEINFO_SAVED_RESULTS','n':336,'arm_names':ARMS,
  'task_rows':task,'negative_comparisons':negative,'scene_rows':scene_rows,'summary':summary['ALL336'],
  'update_diagnostics':diagnostics,'selection_and_gate':gate,'measurement_mode':MODE}
 clean(value)
 # An explicit privacy projection, not copying arbitrary producer objects.
 encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
 for name in {r['scene_PRIVATE'] for r in rows}:
  assert not re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',encoded)
 for ident in {r['id_PRIVATE'] for r in rows}:assert ident not in encoded
 assert not local(OUTPUT).exists() and not local(RECEIPT).exists()
 local(OUTPUT).write_text(encoded)
 result={'state':'PASS_R50_SAVED_ONLY_TTA_RESULTS_NORMALIZED_FOR_FINAL_REPORT','n':336,'arms':6,
  'task_values':len(task),'negative_comparisons':len(negative),'scene_alias_counts':{'drone':2,'quad':3},
  'all_exposed_split_summaries_retained':True,'measurement_mode':MODE,'gate_PASS':False,
  'output_sha256':sha(local(OUTPUT)),'score_seal_sha256':sha(local(SCORE+'/SEALED.json')),
  'saved_scalar_audit_sha256':sha(local(SAVED+'/AUDIT_RESULT.json')),
  'numeric_audit_sha256':sha(local(numeric_result)),'program_sha256':sha(__file__),
  'plan_sha256':sha(pp),'GT_models_or_new_scoring_calls':0,'seconds':time.perf_counter()-start}
 local(RECEIPT).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(result))


if __name__=='__main__':main()
