"""Synthetic builder guards/projections only; never build or read actual scores."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,tempfile
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('test_R14_finite_builder',ROOT/'tools/build_round14_export.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def check():
 checks={}
 with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='builder_synthetic_') as td:
  p=Path(td);r=p/'readiness.json';b.write(r,{'schema':'R14_finite_public_export_readiness_v1','build_authorized':False})
  try:b.readiness(r,b.sha(r))
  except ValueError as err:assert 'No root build authorization' in str(err)
  else:raise AssertionError('Unready export authorized')
  checks['not_ready_rejects_before_task_or_private_payload_read']=True
  good=p/'public';good.mkdir();b.write(good/'metrics.json',{'case_alias':'quad_pair_001','iou':.2,'scope':'synthetic only'})
  b.verify_public_tree(good,{'opaque_raw_input_id'}, {'physical_scene_secret','full literal caption secret'})
  b.write(good/'bad.json',{'Nested':{'Selected_Mask_Index_by_Order':[1,2]}})
  try:b.verify_public_tree(good,set(),set())
  except ValueError:pass
  else:raise AssertionError('Nested actual index key survived')
  (good/'bad.json').unlink();checks['public_tree_recursive_actual_index_rejection']=True
  (good/'bad.md').write_text('physical_scene_secret')
  try:b.verify_public_tree(good,set(),{'physical_scene_secret'})
  except ValueError:pass
  else:raise AssertionError('Raw scene leak survived')
  (good/'bad.md').unlink();checks['raw_scene_marker_leak_rejected']=True
  projected=b.report_projection('[kept](helper.py) ![local figure](private.jpg)',p/'REPORT.md',{(p/'helper.py').resolve():'code/helper.py'})
  assert '[kept](code/helper.py)' in projected and 'private.jpg' not in projected and 'local figure' in projected
  checks['report_links_project_without_exposing_private_targets']=True
  record=p/'audit.json';b.write(record,{'state':'passed_synthetic_only','n':3,'raw_answer_PRIVATE':'must not copy'})
  audit=b.audit_projection({'path':str(record),'sha256':b.sha(record),'required_assertions':{'n':3},'scalar_field_map':{'state':'state','n':'n'}})
  assert audit['n']==3 and 'raw_answer_PRIVATE' not in audit
  checks['audit_actual_scalar_and_sha_projection_no_fake_PASS']=True
  row={'population':'all','arm':'a','reference':'source','selection':'synthetic signed extremes',
     'largest_negative_PRIVATE':[{'pair_id':'secret_pair','delta_iou':-.1}],
     'largest_positive_PRIVATE':[{'pair_id':'secret_pair','delta_iou':.2}]}
  tails=b.public_tails({'signed_tails_PRIVATE':[row]},'quad',{'secret_pair':'quad_pair_001'})
  assert 'secret_pair' not in json.dumps(tails) and tails[0]['largest_negative'][0]['delta_iou']==-.1
  checks['negative_tail_alias_preserved_no_raw_pair_ID']=True
  event_data={
   'drone_formal001_interrupted':('root_interrupted_before_GT',{'completed_cases':5,'expected_cases':64,'progress_reused_calls':20}),
   'drone_formal001_closed_inventory':('root_interrupted_before_GT',{'actual_closed_calls':39,'new_model_calls':15,'reused_calls':24}),
   'drone_format_fix_review':('format_fix_only',{'passed':True}),
   'drone_engineering001':('engineering_interface_failure',{'completed_cases':2}),
   'drone_engineering002':('engineering002_pass',{'completed_cases':2}),
   'drone_formal002_budget_bottleneck':('visual_generation_budget_reference_contract',{'visual_calls':128,'visual_cap':64,'visual_no_eos_n':61,'valid_visual_calls':0,'combined_source_fallback_cases':64}),
   'drone_attempt003_preGT_authorization':('preGT_budget_reference_interface_fix',{'no_GT_at_event':True}),
   'drone_engineering003':('nonstring_reference_slot_discards_valid_target',{'completed_cases':2,'expected_cases':2,'new_model_calls':10,'reused_calls':4,'visual_calls':4,'visual_eos_n':4,'formal_started':False}),
   'drone_attempt004_preGT_authorization':('preGT_reference_slot_type_interface_fix',{'no_GT_at_event':True}),
   'drone_engineering004_metadata_failure':('engineering_count_variable_shadow',{'completed_cases':2,'new_model_calls':10,'reused_calls':4,'reported_seal_cases':4,'actual_case_files':2,'formal_started':False}),
   'drone_engineering004b':('engineering004b_metadata_only_rename_pass',{'completed_cases':2,'new_model_calls':10,'reused_calls':4,'checked_count':1142,'outputs_identical':True,'algorithm_changed':False})}
  events=[]
  for key,(reason,data) in event_data.items():
   path=p/(key+'.json');b.write(path,data)
   events.append({'event_alias':key,'reason_code':reason,'path':str(path),'sha256':b.sha(path),'numeric_field_map':{k:k for k in data}})
  hist=b.history_projection(events);by={x['event_alias']:x for x in hist}
  assert by['drone_formal001_interrupted']['progress_reused_calls']==20
  assert by['drone_formal001_closed_inventory']['reused_calls']==24
  assert by['drone_engineering003']['formal_started'] is False
  checks['actual_closed_inventory_not_confused_with_progress_and_no_formal003']=True
  extra=p/'extra.json';b.write(extra,{'no_formal':True,'count':22,'counts':{}})
  sources={'primary':{'n':2},'extra':b.read(extra)}
  assert b.history_scalar(sources,{'source':'extra','pointer':'no_formal','op':'not'}) is False
  assert b.history_scalar(sources,{'source':'extra','pointer':'count','op':'equals','equals':22}) is True
  assert b.history_scalar(sources,{'source':'extra','pointer':['counts','absent_event'],'op':'count_default_zero'})==0
  try:b.history_scalar(sources,{'source':'extra','pointer':'count','op':'eval'})
  except ValueError:pass
  else:raise AssertionError('Arbitrary transform allowed')
  events[-1]['additional_sources']={'engineering':{'path':str(extra),'sha256':b.sha(extra),'required_assertions':{'count':22}}}
  events[-1]['numeric_field_map']['outputs_identical']={'source':'engineering','pointer':'count','op':'equals','equals':22}
  assert b.history_projection(events)[-1]['outputs_identical'] is True
  events[-1]['additional_sources']['engineering']['sha256']='0'*64
  try:b.history_projection(events)
  except ValueError:pass
  else:raise AssertionError('Unverified secondary provenance allowed')
  checks['typed_multisource_history_requires_each_SHA_no_arbitrary_formula']=True
  generic_literal='/'+'home'+'/'
  assert b.code_contains_private_host_path('prefix = '+repr(generic_literal)) is False
  assert b.code_contains_private_host_path('host = '+repr(generic_literal+'private_user/data')) is True
  assert b.code_contains_private_host_path('# '+generic_literal+'private_user/data') is True
  (good/'sentinel.py').write_text('prefix = '+repr(generic_literal))
  b.verify_public_tree(good,set(),set())
  b.write(good/'bad_host.json',{'actual_path':generic_literal+'private_user/data'})
  try:b.verify_public_tree(good,set(),set())
  except ValueError:pass
  else:raise AssertionError('Data/report host path exception incorrectly enabled')
  (good/'bad_host.json').unlink()
  checks['generic_code_privacy_literals_only_real_code_or_data_hostpaths_still_rejected']=True
  # Generic helpers never create an output package or call network/Git tools.
  assert not (p/'package').exists();checks['no_build_or_upload_in_fixture']=True
 return {'state':'passed_synthetic_R14_builder_readiness_privacy_and_provenance_checks','checks':checks,
  'actual_scores_GT_model_raw_media_or_public_package_accessed':False,
  'program_sha256':{'tools/build_round14_export.py':b.sha(ROOT/'tools/build_round14_export.py'),
    'research/round14/publication/check_builder_cpu.py':b.sha(Path(__file__))}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=check()
 if a.output.exists():raise FileExistsError(a.output)
 b.write(a.output,r);print(json.dumps({'state':r['state'],'checks':len(r['checks'])}))
