"""Root-executed R15 prompt-schema v2 once repair; original methods untouched.
No target GT/scoring, source-weight update, generated code or silent retries.
Root actual plan/hash+authorization+source/mask seals precede all model/media IO.
"""
from pathlib import Path
import argparse,collections,hashlib,json,os,sys,time,traceback
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1]
sys.path[:0]=[str(D),str(B/'crossplatform/round13')]
from semantic_graph import DEFAULT_CONFIG as SEM_CONFIG,asdict,parse_caption_graph,parse_visual_graph,combine_visual_graphs,match_scene_graph,proposal_consensus,match_relation_flip_diagnostic
from semantic_runner_v2 import text_messages,visual_messages,json_safe
from scene_observation import build_scene_observation,DEFAULT_CONFIG as SCENE_CONFIG
from association import DEFAULT_CONFIG as PANEL_CONFIG,array_sha,numbered_candidates,render_numbered_panel
from association_runner import FrozenVLM,atomic_json
from input_io import read,sha,verify_seal,load_case

class SemanticFrozenVLM(FrozenVLM):
 def predict_messages(self,messages,images,max_new_tokens):
  t=self.torch;text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  kw={'text':[text],'return_tensors':'pt','padding':False,'truncation':False}
  if images:kw['images']=images
  ins=self.processor(**kw).to('cuda')
  if ins.input_ids.shape[1]>8192:raise RuntimeError('Locked8192prompt cap exceeded; no silent truncation')
  grid=ins['image_grid_thw'].tolist() if 'image_grid_thw' in ins else []
  if len(images) not in (0,2) or len(grid)!=len(images) or not all(g[0]==1 and g[1]*g[2]//4<=1024 for g in grid):raise RuntimeError('Exactly0or2staticimages with <=1024visualtokens each')
  t.cuda.synchronize();start=time.perf_counter()
  with t.inference_mode():out=self.model.generate(**ins,do_sample=False,max_new_tokens=max_new_tokens,use_cache=True)
  t.cuda.synchronize();tail=out[0,ins.input_ids.shape[1]:];raw=self.processor.tokenizer.decode(tail,skip_special_tokens=True,clean_up_tokenization_spaces=False);eos=self.model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
  return {'raw_answer_PRIVATE':raw,'serialized_prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'serialized_prompt_PRIVATE':text,'messages_sha256':canonical_sha(messages),'input_ids_sha256':array_sha(ins.input_ids.detach().cpu().numpy()),'prompt_payload_version':'schema_fewshot_v2_once','image_grid_thw':grid,'input_tokens':int(ins.input_ids.shape[1]),'generated_tokens':len(tail),'max_new_tokens':max_new_tokens,'ended_eos':any(int(x) in eos for x in tail),'seconds_generate':time.perf_counter()-start,'greedy_no_retry_no_tools':True}

def scene_metadata(scene):
 fields=('mask_id','native_instance_id','array_index','predicted_class_id','detector_score','mask_sha256','mask_area_pixels','mask_geometry_defined','mask_centroid_defined','mask_extent_defined','mask_bbox_defined','mask_centroid_norm2','mask_extent_norm2','mask_bbox_norm4','touches_image_border','n_visible','n_exclusive','n_ambiguous_overlap','depth_camera','support_sha256','support_raw_ids_sha256')
 return json_safe({k:scene[k] for k in ('schema','platform','image_size_wh','image_sha256','raw_points_input_sha256','camera_input_sha256','visibility_sha256','mask_input_sha256','config','config_sha256','diagnostics')}|{'objects':[{k:o[k] for k in fields} for o in scene['objects']]})
def canonical_sha(obj):return hashlib.sha256(json.dumps(json_safe(obj),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def derive(graph,parsed,scene):
 visual=combine_visual_graphs(parsed)
 return {'text_graph':graph,'visual_parsed':parsed,'combined_visual':visual,'proposal_consensus':proposal_consensus(visual),'graph_match':match_scene_graph(graph,scene,visual),'graph_relation_disabled':match_scene_graph(graph,scene,visual,relation_check=False),'graph_relation_flip':match_relation_flip_diagnostic(graph,scene,visual),'source_q0_boxes_scores_used_for_semantic_selection':False,'all_mask_inventory_preserved':True,'agreement_is_not_correctness':True}
def parse_complete_text(result,caption):
 return parse_caption_graph(result['raw_answer_PRIVATE'],caption) if result['ended_eos'] is True else {'status':'NA_text_output_budget_nonEOS','entities':[],'attributes':[],'relations':[]}
def parse_complete_visual(result,candidates,graph):
 return parse_visual_graph(result['raw_answer_PRIVATE'],candidates,graph) if result['ended_eos'] is True else {'status':'NA_visual_output_budget_nonEOS','role_proposals':{},'attribute_claims':{}}
def prepare(a):
 if sha(a.plan)!=a.plan_sha256:raise ValueError('Actual root plan hash required')
 p=read(a.plan)
 if p['stage']!='R15_unified_full_description_semantic_graph' or p['GT_read_permitted'] is not False or p['GPU_execution_permitted'] is not True or p['model_execution_permitted'] is not True:raise ValueError('Root execution scope')
 if p.get('prompt_payload_version')!='schema_fewshot_v2_once':raise ValueError('Explicit one pre-GT schema repair plan required')
 if p['expected_samples']!=204 or p['formal_expected_calls']!=612 or p['semantic_config']!=json_safe(asdict(SEM_CONFIG)) or p['scene_config']!=asdict(SCENE_CONFIG) or p['panel_config']!=asdict(PANEL_CONFIG):raise ValueError('Fixed counts/config')
 for rel,h in p['program_sha256'].items():
  if sha(B/rel)!=h:raise ValueError('Frozen program changed '+rel)
 if p['program_sha256'].get(str(Path(__file__).relative_to(B)))!=sha(__file__):raise ValueError('Actual runner must be locked')
 if sha(p['authorization_path'])!=p['authorization_sha256']:raise ValueError('Root authorization bytes')
 data={}
 for k in ('manifest','case_index','frame_index'):
  if sha(p[k])!=p[k+'_sha256']:raise ValueError('Input-only binding '+k)
  data[k]=read(p[k])
 rows=data['manifest'];cases={c['id']:c for c in data['case_index']}
 if len(rows)!=204 or len(cases)!=204 or len({r['id'] for r in rows})!=204 or set(cases)!={r['id'] for r in rows}:raise ValueError('Complete204unique inputs')
 if dict(collections.Counter(c['role'] for c in cases.values()))!=p['expected_role_counts'] or sorted(p['expected_role_counts'].values())!=[64,140]:raise ValueError('Primary64/regression140 separate')
 if any(r['platform'] not in ('drone','quad') for r in rows):raise ValueError('Same two-platform core only')
 expected_seals={str(Path(c[k])/'SEALED.json') for c in cases.values() for k in ('source_dir','mask_dir')}
 if len(expected_seals)!=8 or set(p['cache_seals_sha256'])!=expected_seals:raise ValueError('Exact8source/mask producers')
 for rel,h in p['cache_seals_sha256'].items():
  f=B/rel
  if sha(f)!=h:raise ValueError('Capture seal changed')
  verify_seal(f.parent)
 modeldir=Path(p['model_dir']);needed={'chat_template.json','config.json','generation_config.json','merges.txt','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','vocab.json'}
 if set(p['model_files_sha256'])!=needed or any(sha(modeldir/f)!=h for f,h in p['model_files_sha256'].items()):raise ValueError('Exact11local public asset hashes')
 if read(modeldir/'config.json')['model_type']!='qwen2_5_vl' or set(read(modeldir/'model.safetensors.index.json')['weight_map'].values())!={'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors'}:raise ValueError('Local safetensors/Qwen architecture')
 selected_rows=list(rows)
 if a.limit is not None:
  ids=p['engineering_case_ids']
  if a.limit!=4 or len(ids)!=4 or len(set(ids))!=4 or not set(ids)<=set(cases):raise ValueError('Exactly4root preselected engineering inputs')
  lookup={r['id']:r for r in rows};selected_rows=[lookup[i] for i in ids]
 if a.output.exists():raise ValueError('New attempt directory required; no automatic resume or overwrite')
 return p,data,selected_rows,cases

def main(a):
 start=time.perf_counter();plan,data,selected_rows,cases=prepare(a);out=a.output;out.mkdir(parents=True);atomic_json(out/'registered_plan.json',plan)
 for f in ('calls_PRIVATE','cases_PRIVATE','scene_metadata_PRIVATE','panels_PRIVATE'):(out/f).mkdir()
 n_calls=0;engine=None;expected_n=len(selected_rows);statuses=collections.Counter();roles=collections.Counter()
 def save_call(name,payload):
  path=out/'calls_PRIVATE'/(name+'.json');atomic_json(path,json_safe(payload))
  with (out/'calls.jsonl').open('a') as f:f.write(json.dumps({'path':str(path.relative_to(out)),'sha256':sha(path)})+'\n');f.flush();os.fsync(f.fileno())
 try:
  t=time.perf_counter();engine=SemanticFrozenVLM(Path(plan['model_dir']),PANEL_CONFIG);atomic_json(out/'MODEL_LOAD_RECEIPT.json',{'strict_local_safetensors_eval':True,'loading_info':engine.loading_info,'seconds':time.perf_counter()-t,'model_files_sha256':plan['model_files_sha256']})
  for case_i,row in enumerate(selected_rows):
   case=cases[row['id']];alias=case['case_alias']
   if not isinstance(alias,str) or not alias or any(not(x.isalnum() or x=='_') for x in alias):raise ValueError('Safe private case alias')
   frame,unused_source_cache,bindings=load_case(row,case,data['frame_index']);inst=frame['instances'];h,w=frame['rgb'].shape[:2]
   scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=inst,image_sha256=bindings['image_sha256'])
   metadata=scene_metadata(scene);metadata_hash=canonical_sha(metadata);atomic_json(out/'scene_metadata_PRIVATE'/(alias+'.json'),metadata)
   candidates=[numbered_candidates(row['id'],inst['masks_bool'],inst['boxes_xyxy'],permutation=i) for i in range(2)]
   for inventory in candidates:
    for c in inventory:c['mask_id']='M%05d'%int(inst['instance_indices'][c['mask_index']])
   if any({c['mask_id'] for c in inventory}!={o['mask_id'] for o in scene['objects']} for inventory in candidates):raise ValueError('Candidate↔whole mask inventory exact bridge')
   from PIL import Image
   original=Image.fromarray(frame['rgb']);panels=[render_numbered_panel(frame['rgb'],inventory) for inventory in candidates]
   for i,panel in enumerate(panels):panel.save(out/'panels_PRIVATE'/(alias+'__order'+str(i)+'.png'))
   text=engine.predict_messages(text_messages(row['caption']),[],SEM_CONFIG.text_max_new_tokens);n_calls+=1
   graph=parse_complete_text(text,row['caption'])
   save_call(alias+'__text',{'result':text,'parsed':graph,'bindings':bindings,'scene_metadata_sha256':metadata_hash})
   visual=[]
   for order in range(2):
    result=engine.predict_messages(visual_messages(row['caption'],graph,candidates[order]),[original,panels[order]],SEM_CONFIG.visual_max_new_tokens);n_calls+=1
    parsed=parse_complete_visual(result,candidates[order],graph)
    visual.append(parsed);save_call(alias+'__visual_order'+str(order),{'result':result,'parsed':parsed,'bindings':bindings,'scene_metadata_sha256':metadata_hash,'candidate_mapping_PRIVATE':candidates[order],'original_RGB_array_sha256':array_sha(frame['rgb']),'panel_RGB_array_sha256':array_sha(np.asarray(panels[order]))})
   result=derive(graph,visual,scene);payload={'sample_id_PRIVATE':row['id'],'case_alias':alias,'role':case['role'],'platform':row['platform'],'bindings':bindings,'scene_metadata_sha256':metadata_hash,'semantics':result};atomic_json(out/'cases_PRIVATE'/(alias+'.json'),json_safe(payload));statuses[result['graph_match']['status']]+=1;roles[case['role']]+=1
   atomic_json(out/'PROGRESS.json',{'completed_cases':case_i+1,'expected_cases':expected_n,'model_calls':n_calls,'scope':'engineering4' if a.limit else 'formal204'})
   if (case_i+1)%5==0 or a.limit:print(json.dumps({'event':'R15_semantic_progress','completed':case_i+1,'expected':expected_n,'calls':n_calls}),flush=True)
  if not engine.verify_unchanged() or n_calls!=expected_n*3:raise ValueError('Frozen model/call closure')
  files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='SEALED.json'}
  atomic_json(out/'SEALED.json',{'state':'sealed_R15_semantic_engineering4_before_GT' if a.limit else 'sealed_R15_semantic204_three_calls_before_GT','n_cases':expected_n,'n_model_calls':n_calls,'role_counts':dict(roles),'literal_graph_status_counts':dict(statuses),'program_sha256':plan['program_sha256'],'plan_sha256':sha(a.plan),'manifest_sha256':plan['manifest_sha256'],'cache_seals_sha256':plan['cache_seals_sha256'],'model_files_sha256':plan['model_files_sha256'],'model_frozen_parameters_buffers_unchanged':True,'no_generated_code_GT_model_update_or_sourceROI':True,'scene_metadata_rebuild_hash_required_downstream':True,'prompt_payload_version':'schema_fewshot_v2_once','seconds_process':time.perf_counter()-start,'peak_allocated_bytes':int(engine.torch.cuda.max_memory_allocated()),'files':files})
  print(json.dumps({'state':'R15_semantic_sealed','n':expected_n,'calls':n_calls,'seal_sha256':sha(out/'SEALED.json')}))
 except Exception as e:
  atomic_json(out/'FAILURE.json',{'state':'R15_semantic_attempt_failed_preserved','error':str(e),'traceback_PRIVATE':traceback.format_exc(),'actual_calls_completed':n_calls,'no_automatic_retry_or_answer_repair':True});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);main(p.parse_args())
