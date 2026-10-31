"""V2 prompt-schema repair CPU tests; no actual model/GT/sensor execution."""
from pathlib import Path
import ast,hashlib,json,copy
from semantic_graph import parse_caption_graph,parse_visual_graph
from semantic_runner_v2 import TEXT_EXAMPLE,TEXT_EXAMPLE_CAPTION,VISUAL_EXAMPLE,text_messages,visual_messages,linguistic_payload
from semantic_model_runner_v2 import parse_complete_text,canonical_sha
D=Path(__file__).resolve().parents[2]/'research/round15/association'
def main():
 g=parse_caption_graph(json.dumps(TEXT_EXAMPLE),TEXT_EXAMPLE_CAPTION);assert g['status']=='PARSED_text_graph'
 cand=[{'candidate_id':'C%03d'%i,'mask_id':'M%05d'%i} for i in range(1,6)]
 p=parse_visual_graph(json.dumps(VISUAL_EXAMPLE),cand,g);assert p['status']=='PARSED_visual_graph' and p['role_proposals']['T']==['M00002']
 m=text_messages('UNIQUE_CURRENT_DESCRIPTION_SENTINEL');assert len(m)==4 and m[-1]['content'].count('UNIQUE_CURRENT_DESCRIPTION_SENTINEL')==1 and set(json.loads(m[2]['content']))=={'status','target','references','attributes','relations'}
 bad={'status':'NA_text_graph_contract','reason':'UNTRUSTED_INTERNAL_ERROR_SENTINEL','entities':[],'attributes':[],'relations':[]};san=linguistic_payload(bad);assert 'reason' not in san and 'UNTRUSTED_INTERNAL_ERROR_SENTINEL' not in json.dumps(san)
 vm=visual_messages('CURRENT_LITERAL_QUERY',bad,cand);last=vm[-1]['content'];assert len([x for x in last if x['type']=='image'])==2 and 'UNTRUSTED_INTERNAL_ERROR_SENTINEL' not in json.dumps(vm);assert set(json.loads(vm[2]['content']))=={'status','roles','attribute_checks'}
 assert canonical_sha(m)==canonical_sha(json.loads(json.dumps(m)))
 assert parse_complete_text({'raw_answer_PRIVATE':json.dumps(TEXT_EXAMPLE),'ended_eos':False},TEXT_EXAMPLE_CAPTION)['status']=='NA_text_output_budget_nonEOS'
 # Method/selection functions are byteidentical AST to the original runner;
 # only prompt import, explicit version guard and transport receipt fields differ.
 root=Path(__file__).parent;old=ast.parse((root/'semantic_model_runner.py').read_text());new=ast.parse((root/'semantic_model_runner_v2.py').read_text())
 for name in ('scene_metadata','canonical_sha','derive','parse_complete_text','parse_complete_visual','main'):
  a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);b=copy.deepcopy(next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name))
  if name=='main':
   for node in ast.walk(b):
    if isinstance(node,ast.Dict):
     pairs=[(k,v) for k,v in zip(node.keys,node.values) if not(isinstance(k,ast.Constant) and k.value=='prompt_payload_version')];node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
  assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
 receipt={'state':'PASS_R15_ONE_SCHEMA_PROMPT_REPAIR_V2_CPU_NO_MODEL','checks':['complete_text_example_accepted_by_unchanged_parser','complete_visual3key_example_accepted_by_unchanged_parser','last_current_caption_not_changed','two_actual_images_only_no_fake_example_image','NA_error_reason_not_sent_as_visual_language_evidence','text_visual_budget_and_strictEOS_unchanged','core_matching_and_selection_AST_unchanged_except_version_receipt','messages_canonicalhash_roundtrip'],'program_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),root/'semantic_runner_v2.py',root/'semantic_model_runner_v2.py',root/'semantic_graph.py')},'new_GPU_model_GT_score_or_sensor_media_read':False,'fictional_examples_not_target_evidence':True}
 (D/'PROMPT_V2_CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['state'])
if __name__=='__main__':main()
