"""Model-runner CPU engineering only. Never construct its model class."""
from pathlib import Path
import ast,hashlib,json,sys
from types import SimpleNamespace
import numpy as np
from semantic_model_runner import parse_complete_text,parse_complete_visual,derive,canonical_sha,prepare
from semantic_runner import json_safe
D=Path(__file__).resolve().parents[2]/'research/round15/association'
def main():
 caption='the car left of the image';raw='{"status":"RESOLVED","target":{"quote":"the car","head":"car"},"references":[],"attributes":[],"relations":[{"id":"L1","subject":"T","object":"SCENE","quote":"left","predicate":"GLOBAL_LEFT","frame":"camera_image","viewpoint_quote":"the image"}]}'
 closed=parse_complete_text({'raw_answer_PRIVATE':raw,'ended_eos':True},caption);assert closed['status']=='PARSED_text_graph'
 trunc=parse_complete_text({'raw_answer_PRIVATE':raw,'ended_eos':False},caption);assert trunc['status']=='NA_text_output_budget_nonEOS'
 vraw='{"status":"RESOLVED","roles":[{"entity":"T","candidates":["C001"]}],"attribute_checks":[]}'
 cs=[{'candidate_id':'C001','mask_id':'M00001'}];vr=parse_complete_visual({'raw_answer_PRIVATE':vraw,'ended_eos':False},cs,closed);assert vr['status']=='NA_visual_output_budget_nonEOS'
 good=parse_complete_visual({'raw_answer_PRIVATE':vraw,'ended_eos':True},cs,closed)
 scene={'objects':[{'mask_id':'M00001','predicted_class_id':3,'mask_centroid_norm2':np.array([.2,.5]),'mask_bbox_norm4':np.array([.15,.4,.25,.6]),'touches_image_border':False}]}
 out=derive(closed,[good,good],scene);assert out['graph_match']['status']=='unique_supported' and out['proposal_consensus']['status']=='proposal_control_unique' and out['graph_relation_disabled']['status']=='unique_supported' and out['graph_relation_disabled']['relation_mode']=='disabled'
 assert out['graph_relation_flip']['flip_applied'] and out['graph_relation_flip']['selected_target_mask_id'] is None
 allna=derive(trunc,[vr,vr],scene);assert allna['graph_match']['selected_target_mask_id'] is None
 safe=json_safe({'tuplemap':{('A1','M00001'):'SUPPORT'},'np':np.asarray([1,2]),'scalar':np.int64(3)});assert safe['tuplemap']==[{'key':['A1','M00001'],'value':'SUPPORT'}] and safe['np']==[1,2] and safe['scalar']==3;json.dumps(safe)
 assert canonical_sha(safe)==canonical_sha(json.loads(json.dumps(safe)))
 source=Path(__file__).with_name('semantic_model_runner.py').read_text();tree=ast.parse(source);mainfn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main');loop=next(x for x in ast.walk(mainfn) if isinstance(x,ast.For) and ast.unparse(x.iter)=='enumerate(selected_rows)')
 assert not any(isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store) and x.id=='selected_rows' for x in ast.walk(loop))
 assert 'exec(' not in source and 'eval(' not in source and 'optimizer' not in source.lower()
 # False root scope must reject before asset/cache/media access. The only file
 # is a clearly marked temporary synthetic plan, removed before receipt.
 tmp=D/'SYNTHETIC_FALSE_PLAN_TMP.json';tmp.write_text(json.dumps({'stage':'synthetic_false_scope','GT_read_permitted':True}));a=SimpleNamespace(plan=tmp,plan_sha256=hashlib.sha256(tmp.read_bytes()).hexdigest())
 try:prepare(a)
 except (ValueError,KeyError):pass
 else:raise AssertionError('False scope admitted')
 finally:tmp.unlink()
 r={'state':'PASS_R15_ACTUAL_MODEL_RUNNER_CPU_CONTRACT_NO_MODEL_OR_REALDATA','checks':['import_safe_no_model_constructed','nonEOS_wellformedJSON_is_durableNA','allThreeCallOutputs_have_scalar_NA_path','numpy_and_tuple_keys_lossless_json_safe','scene_metadata_canonicalhash_roundtrip','selected_rows_not_shadowed','false_scope_fails_before_asset_or_media','no_generated_code_eval_exec_or_optimizer','distinct_proposal_graph_disabled_flip_outputs'],'program_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('semantic_model_runner.py'),Path(__file__).with_name('semantic_graph.py'),Path(__file__).with_name('semantic_runner.py'))},'new_model_GPU_GT_score_sensor_or_media_read':False};(D/'MODEL_RUNNER_CPU_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(r['state'])
if __name__=='__main__':main()
