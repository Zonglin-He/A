"""Independent literal-binding-only review. Sealed old text regressions allowed;
no new model, sensors, labels or task IoU. Never edits prior implementation."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
R=Path(__file__).resolve().parents[3];sys.path[:0]=[str(R/'crossplatform/round15')]
import semantic_graph as old
import semantic_graph_v3 as new

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def main(a):
 dest=Path(__file__).with_name('LITERAL_BINDING_V3_INDEPENDENT_REVIEW.json')
 if dest.exists():raise FileExistsError(dest)
 c={};core=R/'crossplatform/round15/semantic_graph_v3.py';runner=R/'crossplatform/round15/semantic_model_runner_v3.py';prompt=R/'crossplatform/round15/semantic_runner_v2.py'
 c['prompt_v2_bytes_unchanged']=sha(prompt)=='6afe95cc495b1e8b3cf0ff0aebeca4c2657ce236aa09f7ab02cfb8a892c0cc9f'
 before=functions(core.with_name('semantic_graph.py'));after=functions(core)
 for k in before:
  if k!='parse_caption_graph':c['unchanged_AST:'+k]=before[k]==after[k]
 c['only_three_new_binding_helpers']=set(after)-set(before)=={'_exact_occurrences','attribute_quote','viewpoint_quote_all'}
 text='the black car and a black truck; in the image, far in the image'
 ent=new.entity_parse({'quote':'the black car','head':'car'},text,'T')
 q=new.attribute_quote(text,'black',ent);c['duplicate_attr_unique_in_entity_only']=q is not None and q['span']==[4,9] and q['global_match_count']==2
 double=dict(ent,quote={'text':text,'span':[0,len(text)]})
 c['two_hits_in_same_entity_not_guessed']=new.attribute_quote(text,'black',double) is None
 c['unknown_entity_duplicate_attr_not_guessed']=new.attribute_quote(text,'black',{'status':'UNKNOWN','quote':None}) is None
 c['absent_attr_unknown']=new.attribute_quote(text,'red',ent) is None
 c['unique_attr_old_rule_unchanged']=new.attribute_quote(text,'truck',ent)==old.quote(text,'truck')
 c['nonword_substring_not_color_hit']=new.attribute_quote('a blackberry and a black car','black',{'status':'UNKNOWN','quote':None})==old.quote('a blackberry and a black car','black')
 view=new.viewpoint_quote_all(text,'the image');c['viewpoint_all_occurrences_no_chosen_unique']=view['spans']==[[36,45],[54,63]] and view['occurrence_count']==2 and view['unique_occurrence_claimed'] is False and 'span' not in view
 c['ordinary_relation_duplicate_stays_unknown']=new.quote('left of a car then left of a car','left of a car') is None
 c['duplicate_main_quote_stays_unknown']=new.entity_parse({'quote':'black','head':'black'},text,'T')['status']=='UNKNOWN'
 oldrun=ast.parse(runner.with_name('semantic_model_runner_v2.py').read_text());newrun=ast.parse(runner.read_text())
 c['runner_body_length_unchanged']=len(oldrun.body)==len(newrun.body)
 for i,(x,y) in enumerate(zip(oldrun.body,newrun.body)):
  if isinstance(x,ast.ImportFrom) and x.module=='semantic_graph':
   c['runner_import_only_change']=isinstance(y,ast.ImportFrom) and y.module=='semantic_graph_v3' and ast.dump(ast.Module(body=x.names,type_ignores=[]))==ast.dump(ast.Module(body=y.names,type_ignores=[]))
  elif isinstance(x,ast.ImportFrom) and x.module=='semantic_runner_v2':
   c['runner_thin_prompt_import']=isinstance(y,ast.ImportFrom) and y.module=='semantic_runner_v3' and ast.dump(ast.Module(body=x.names,type_ignores=[]))==ast.dump(ast.Module(body=y.names,type_ignores=[]))
  else:c['runner_AST_node:'+str(i)]=ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False)
 replay=Path(__file__).parent/'v3_binding_cpu_replay/LITERAL_BINDING_V3_CPU_FIXTURE.json';author=R/'research/round15/association/LITERAL_BINDING_V3_CPU_FIXTURE.json'
 c['old3_contract_failures_stillNA_and_existing_one_scoped_bindings_recovered']=sha(replay)==sha(author)
 lock=json.loads(a.lock.read_text())
 import semantic_runner_v2 as p2,semantic_runner_v3 as p3
 c['thin_adapter_reexports_same_prompt_functions']=all(getattr(p2,k) is getattr(p3,k) for k in ('text_messages','visual_messages','linguistic_payload','json_safe'))
 for rel,h in lock['program_sha256'].items():c['lock_program:'+rel]=sha(R/rel)==h
 c={k:bool(v) for k,v in c.items()};assert all(c.values()),[k for k,v in c.items() if not v]
 result={'state':'PASS_R15_V3_LITERAL_BINDING_ONLY_NO_PROMPT_GUARD_OR_METHOD_RELAXATION',
 'lock_sha256':sha(a.lock),'core_sha256':sha(core),'runner_sha256':sha(runner),'prompt_v2_sha256':sha(prompt),
 'addendum_sha256':sha(R/'research/round15/experiment/QUOTE_BINDING_BUGFIX_ADDENDUM.json'),
 'auditor_sha256':sha(__file__),'checks':c,'check_count':len(c),'independent_author_fixture_replay_sha256':sha(replay),
 'prior_v2_engineering_seal_sha256':'b97fc9e528de98d97107273a0442418bed83d96cbeba48b88699e1d010be94a5','prior_v2_all32_members_hashed_before_existing_text_reparse':True,
 'new_model_GPU_sensor_GT_or_taskscore_accessed':False,'existing_sealed_engineering_text_and_inputcaption_reparse':True,
 'scientific_limits':['This restores literal provenance for already strictly bound roles; it does not repair wrong roles, nonliteral heads, invalid enums or an incorrect direction.',
 'All viewpoint occurrences are retained without selecting a unique occurrence. Their association with a model-supplied predicate remains an unverified language hypothesis.',
 'Prompt bytes remain v2; resulting visual hypothesis contents can differ because correctly bound attributes/viewpoints are no longer UNKNOWN. This is an intended downstream input effect, not new prompt tuning.',
 'Global-unique attributes retain the original policy, even when outside the entity phrase; this review does not claim all attribute-entity semantics are certified.',
 'Existing three structural-invalid engineering texts remain NA; one parsed text recovery is engineering coverage, not target identity or useful geometric supervision.'],
 'old_v1_v2_source_and_evidence_modified':False,'formal_execution_or_task_quality_validated':False,'blocking_issues':[]}
 dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'state':result['state'],'checks':len(c),'sha256':sha(dest)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);main(p.parse_args())
