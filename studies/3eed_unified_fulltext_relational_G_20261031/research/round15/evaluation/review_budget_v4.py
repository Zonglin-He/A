"""Independent budget-only diff/config audit; no actual running outputs read."""
from pathlib import Path
import ast,dataclasses,hashlib,json,sys
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'crossplatform/round15')]
import semantic_graph_v3 as v3,semantic_graph_v4 as v4
from semantic_model_runner_v4 import parse_complete_visual,parse_complete_text

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=Path(__file__).parent;p=d/'DECODE_BUDGET_V4_INDEPENDENT_REVIEW.json'
 if p.exists():raise FileExistsError(p)
 old=R/'crossplatform/round15/semantic_graph_v3.py';new=old.with_name('semantic_graph_v4.py');a=ast.parse(old.read_text());b=ast.parse(new.read_text());c={}
 cls=next(x for x in b.body if isinstance(x,ast.ClassDef) and x.name=='SemanticConfig')
 field=next(x for x in cls.body if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and x.target.id=='visual_max_new_tokens')
 assert field.value.value==2048;field.value=ast.Constant(512)
 c['entire_core_AST_equal_except_visual_budget512_to2048']=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 orun=R/'crossplatform/round15/semantic_model_runner_v3.py';nrun=orun.with_name('semantic_model_runner_v4.py');oa=ast.parse(orun.read_text());na=ast.parse(nrun.read_text())
 node=next(x for x in na.body if isinstance(x,ast.ImportFrom) and x.module=='semantic_graph_v4');node.module='semantic_graph_v3'
 c['entire_runner_AST_equal_except_core_module_import']=ast.dump(oa,include_attributes=False)==ast.dump(na,include_attributes=False)
 cfg3=dataclasses.asdict(v3.DEFAULT_CONFIG);cfg4=dataclasses.asdict(v4.DEFAULT_CONFIG);changed={k:(cfg3[k],cfg4[k]) for k in cfg3 if cfg3[k]!=cfg4[k]}
 c['defaults_only_one_visual_budget_changed']=changed=={'visual_max_new_tokens':(512,2048)}
 c['text_budget_stays1024']=cfg4['text_max_new_tokens']==1024
 c['prompt_v2_unchanged']=sha(R/'crossplatform/round15/semantic_runner_v2.py')=='6afe95cc495b1e8b3cf0ff0aebeca4c2657ce236aa09f7ab02cfb8a892c0cc9f'
 c['thin_prompt_adapter_unchanged']=sha(R/'crossplatform/round15/semantic_runner_v3.py')=='0fd71b53f15fcf98ff3bdbdd7f2c08eaf04f64cb1372f883a016da94ac96af48'
 planpath=R/'research/round15/experiment/SEMANTIC_PLAN_V4_LOCK.json';plan=json.loads(planpath.read_text());prev=json.loads((R/'research/round15/experiment/SEMANTIC_PLAN_V3_LOCK.json').read_text())
 c['plan_default_config_exact']=plan['semantic_config']==json.loads(json.dumps(cfg4))
 for k in ('engineering_case_ids','manifest_sha256','case_index_sha256','frame_index_sha256','cache_seals_sha256','model_files_sha256','expected_role_counts','expected_samples','formal_expected_calls','prompt_payload_version','scene_config','panel_config'):
  c['same_plan_input_or_method:'+k]=plan[k]==prev[k]
 for rel,h in plan['program_sha256'].items():c['bound_code:'+rel]=sha(R/rel)==h
 c['both_budget_files_explicitly_locked']=plan['program_sha256'].get(str(new.relative_to(R)))==sha(new) and plan['program_sha256'].get(str(nrun.relative_to(R)))==sha(nrun)
 c['GT_permission_still_false']=plan['GT_read_permitted'] is False
 raw='{"status":"UNKNOWN","roles":[],"attribute_checks":[]}'
 c['nonEOS_JSON_still_no_prefix_salvage']=parse_complete_visual({'raw_answer_PRIVATE':raw,'ended_eos':False},[],{'status':'UNKNOWN_text_graph'})['status']=='NA_visual_output_budget_nonEOS'
 source=nrun.read_text();c['actual_callsite_uses_frozen_visual_default']='SEM_CONFIG.visual_max_new_tokens' in source and 'SEM_CONFIG.text_max_new_tokens' in source
 addonpath=R/'research/round15/experiment/DECODE_BUDGET_V4_ADDENDUM.json';ad=json.loads(addonpath.read_text())
 c['48engineer_plus612formal_equals660within720']=ad['total_new_engineering_calls']==48 and ad['formal_calls']==612 and ad['total_calls_if_complete']==660 and ad['global_call_ceiling']==720
 c['echo_warning_and_closed_entry_not_erased']=ad['demonstration_echo_observed_in_V3']==2 and ad['entry_remains_closed_until_current_version_passes'] is True
 c={k:bool(v) for k,v in c.items()};assert all(c.values()),[k for k,v in c.items() if not v]
 out={'state':'PASS_R15_V4_DECODE_BUDGET_ONLY_CPU_DIFF_AND_PLAN_CONTRACT_NOT_ENGINEERING_COVERAGE',
 'plan_sha256':sha(planpath),'addendum_sha256':sha(addonpath),'core_sha256':sha(new),'runner_sha256':sha(nrun),
 'auditor_sha256':sha(__file__),'checks':c,'check_count':len(c),'issues':[],
 'new_or_running_engineering_outputs_read':False,'model_GPU_taskGT_IoU_sensor_or_rawcaption_read':False,
 'actual_new_call_budget_check_scope':'Static default, frozen plan and exact invocation site agree on text1024/visual2048; actual completed call records require later sealed engineering audit.',
 'old_v1_v2_v3_modified':False,'formal_entry_validated':False,
 'limits':['Larger output budget cannot establish echo or semantic errors are solved.',
 'EOS and strict JSON are engineering coverage only, not correct target identity.',
 'Prompt and guards unchanged, so invalid role/head/enum/frame/direction remains rejected under the same code.',
 'No actual output quality or task effect was evaluated in this bounded review.']}
 p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'state':out['state'],'checks':len(c),'sha256':sha(p)}))
if __name__=='__main__':main()
