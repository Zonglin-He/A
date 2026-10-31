"""Literal-binding v3 only. No GT/model/media/new sensor data."""
from pathlib import Path
import ast,hashlib,json,sys,copy
from semantic_graph_v3 import *
from semantic_model_runner_v3 import parse_complete_text,derive
D=Path(__file__).resolve().parents[2]/'research/round15/association'
def main():
 caption='a black car left of another black car in the image; a sign in the image'
 raw={'status':'RESOLVED','target':{'quote':'a black car','head':'car'},'references':[{'entity':'R1','quote':'another black car','head':'car','distinct_from_target':True}],'attributes':[{'id':'A1','entity':'T','quote':'black','kind':'color'},{'id':'A2','entity':'R1','quote':'black','kind':'color'}],'relations':[{'id':'L1','subject':'T','object':'R1','quote':'left of another black car','predicate':'LEFT_OF','frame':'camera_image','viewpoint_quote':'the image'}]}
 g=parse_caption_graph(json.dumps(raw),caption);assert g['status']=='PARSED_text_graph'
 spans=[x['quote']['span'] for x in g['attributes']];assert spans[0]!=spans[1] and all(x['quote']['binding']=='unique_within_strict_entity_quote' for x in g['attributes'])
 view=g['relations'][0]['viewpoint_quote'];assert view['occurrence_count']==2 and len(view['spans'])==2 and not view['unique_occurrence_claimed'] and 'span' not in view
 t=g['entities'][0];copy_ent=dict(t,quote={'text':caption,'span':[0,len(caption)]});assert attribute_quote(caption,'black',copy_ent) is None
 assert attribute_quote(caption,'black',{'status':'UNKNOWN','quote':None}) is None
 absent=raw.copy();absent['target']={'quote':'black car','head':'car'};assert parse_caption_graph(json.dumps(absent),caption)['status']=='NA_text_graph_contract'
 assert viewpoint_quote_all(caption,'from the viewer') is None
 # Predicate/frame/negation guards and the ordinary relation-quote uniqueness
 # policy remain strict, independent of the repeated-viewpoint source binding.
 obj1={'mask_centroid_norm2':[.2,.5],'mask_bbox_norm4':[.15,.4,.25,.6],'touches_image_border':False};obj2={'mask_centroid_norm2':[.8,.5],'mask_bbox_norm4':[.75,.4,.85,.6],'touches_image_border':False}
 assert relation_evidence(g['relations'][0],obj1,obj2)['status']=='SUPPORT'
 wrong=dict(g['relations'][0],predicate='RIGHT_OF');assert relation_evidence(wrong,obj1,obj2)['status']=='UNKNOWN'
 assert quote('left of a car then left of a car','left of a car') is None
 # Exact real engineering text regression only, no sensors/task labels.
 B=Path(__file__).resolve().parents[2];plan=json.loads((B/'research/round15/experiment/SEMANTIC_PLAN_V2_LOCK.json').read_text());rows={r['id']:r for r in json.loads(Path(plan['manifest']).read_text())};cases={c['case_alias']:c for c in json.loads(Path(plan['case_index']).read_text())};P=B/'crossplatform/round15/semantic_engineering4_attempt002/calls_PRIVATE';actual={}
 for alias in ('reg_drone_00033','reg_quad_00056','fresh_drone_00001','fresh_quad_00001'):
  call=json.loads((P/(alias+'__text.json')).read_text());case=cases[alias];out=parse_complete_text(call['result'],rows[case['id']]['caption']);actual[alias]=out
 assert all(actual[a]['status']=='NA_text_graph_contract' for a in ('reg_drone_00033','reg_quad_00056','fresh_drone_00001'))
 one=actual['fresh_quad_00001'];assert one['status']=='PARSED_text_graph' and all(a['status']=='PARSED' for a in one['attributes']) and all(r['viewpoint_quote'] is not None and len(r['viewpoint_quote']['spans'])==2 for r in one['relations'])
 wrong=next(r for r in one['relations'] if r['id']=='L5');assert relation_evidence(wrong,obj1,obj2)['status']=='UNKNOWN'
 # Normalize ONLY the documented binding changes; every other guard,
 # enum, relation/matching function must remain identical to frozen v1.
 original_core=ast.parse((B/'crossplatform/round15/semantic_graph.py').read_text());v3core=ast.parse((B/'crossplatform/round15/semantic_graph_v3.py').read_text())
 class UndoBinding(ast.NodeTransformer):
  def visit_FunctionDef(self,node):
   if node.name in ('_exact_occurrences','attribute_quote','viewpoint_quote_all'):return None
   return self.generic_visit(node)
  def visit_Assign(self,node):
   if any(isinstance(t,ast.Name) and t.id=='entity_lookup' for t in node.targets):return None
   return self.generic_visit(node)
  def visit_Call(self,node):
   node=self.generic_visit(node)
   if isinstance(node.func,ast.Name) and node.func.id=='attribute_quote':node.func.id='quote';node.args=node.args[:2]
   if isinstance(node.func,ast.Name) and node.func.id=='viewpoint_quote_all':node.func.id='quote'
   return node
 v3core=UndoBinding().visit(v3core)
 original_core.body=original_core.body[1:];v3core.body=v3core.body[1:]
 assert ast.dump(original_core,include_attributes=False)==ast.dump(v3core,include_attributes=False)
 old=ast.parse((B/'crossplatform/round15/semantic_model_runner_v2.py').read_text());new=ast.parse((B/'crossplatform/round15/semantic_model_runner_v3.py').read_text())
 for a,b in zip(old.body,new.body):
  if isinstance(a,ast.ImportFrom) and a.module=='semantic_graph':assert isinstance(b,ast.ImportFrom) and b.module=='semantic_graph_v3' and ast.dump(ast.Module(body=a.names,type_ignores=[]))==ast.dump(ast.Module(body=b.names,type_ignores=[]))
  elif isinstance(a,ast.ImportFrom) and a.module=='semantic_runner_v2':assert isinstance(b,ast.ImportFrom) and b.module=='semantic_runner_v3' and [x.name for x in a.names]==[x.name for x in b.names]
  else:assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 receipt={'state':'PASS_R15_V3_LITERAL_BINDING_CPU_NO_GT_NO_MODEL','checks':['global_unique_attribute_unchanged','repeated_attribute_only_unique_in_strict_entity_span','ambiguous_within_entity_or_unknown_entity_stays_UNKNOWN','target_entity_head_schema_enum_guards_unchanged','repeated_viewpoint_allspans_no_unique_claim','ordinary_relation_quote_global_unique_unchanged','direction_polarity_frame_guards_unchanged','actual_old3_contract_failures_remain_NA','actual_existing_freshQ_scoped_attrs4and5allspan_viewpoints_recovered','actual_existing_wrong_direction_still_UNKNOWN','runner_v3_AST_only_core_and_thin_prompt_import_changed','v2_prompt_bytes_reused_unmodified','entire_core_AST_equal_after_only_declared_binding_undo'],'existing_engineering_text_reparse_not_new_inference_or_taskGT':True,'program_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),B/'crossplatform/round15/semantic_graph_v3.py',B/'crossplatform/round15/semantic_model_runner_v3.py',B/'crossplatform/round15/semantic_runner_v2.py',B/'crossplatform/round15/semantic_runner_v3.py')},'new_model_GPU_taskGT_score_or_raw_sensor_media_read':False}
 (D/'LITERAL_BINDING_V3_CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['state'])
if __name__=='__main__':main()
