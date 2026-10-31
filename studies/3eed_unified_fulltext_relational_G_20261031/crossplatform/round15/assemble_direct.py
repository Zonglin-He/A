"""R15 root-executed eight direct arms after semantic/source/mask sealing.

Import-safe pure composition + guarded CPU CLI. No model, target labels, new task
IoU, parameter adaptation, generated code, legacy G endpoint or answer repair.
"""
from pathlib import Path
import argparse,collections,hashlib,json,os,sys,time,traceback
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1]
sys.path[:0]=[str(D),str(B/'crossplatform/round13'),str(B/'crossplatform/round7/experiment')]
from scene_observation import build_scene_observation,DEFAULT_CONFIG as SCENE_CONFIG
from joint_geometry import generate_joint_targets,DEFAULT_CONFIG as JOINT_CONFIG
from input_io import read,sha,verify_seal,load_case
from runner_helpers import eval_boxes
from semantic_model_runner import scene_metadata,canonical_sha
from semantic_runner import json_safe
from dataclasses import asdict

ARM_NAMES=('source','proposal_assoc','proposal_geo','graph_assoc','graph_geo','graph_geo_noRGB','graph_geo_relation_disabled','graph_geo_relation_flipped_diagnostic')
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'

def atomic_json(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(json_safe(obj),indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def compose_direct_arrays(*,scene,source_pool,text_scores,q_source,semantics):
 """All branches share the same fresh source pool and independent scene.

 A nonapplicable flip control exactly reuses literal graph arrays rather than
 cancelling the update. Geometry fallback stays associated q, never source q0.
 """
 required={'proposal_consensus','graph_match','graph_relation_disabled','graph_relation_flip'}
 if not isinstance(semantics,dict) or not required<=set(semantics):raise ValueError('All four actual semantic branches required')
 pool=np.asarray(source_pool,np.float32);scores=np.asarray(text_scores)
 if pool.shape!=(256,6) or scores.shape!=(256,) or int(q_source)!=int(scores.argmax()):raise ValueError('Exact256 fresh source pool/argmax bridge')
 base=dict(scene=scene,boxes_raw6=pool,text_scores=scores,q0=int(q_source),config=JOINT_CONFIG)
 proposal=generate_joint_targets(**base,graph_match=semantics['proposal_consensus'],semantic_branch='proposal_control')
 graph=generate_joint_targets(**base,graph_match=semantics['graph_match'],semantic_branch='graph')
 disabled=generate_joint_targets(**base,graph_match=semantics['graph_relation_disabled'],semantic_branch='graph')
 flip_info=semantics['graph_relation_flip']
 if not isinstance(flip_info.get('flip_applied'),bool):raise ValueError('Explicit diagnostic flip applicability required')
 if flip_info['flip_applied']:
  flip=generate_joint_targets(**base,graph_match=flip_info,semantic_branch='graph');reused=False
 else:
  # Producer must already state exact-fullgraph semantics under nonapplicability.
  for key in ('status','selected_target_mask_id','search_complete'):
   if flip_info[key]!=semantics['graph_match'][key]:raise ValueError('Nonapplicable flip cannot silently cancel or change full graph')
  if flip_info.get('nonapplicable_policy')!='exact_original_fullgraph_output_not_source_cancellation':raise ValueError('Explicit nonapplicable flip policy required')
  flip=graph;reused=True
 branches={'proposal':proposal,'graph':graph,'relation_disabled':disabled,'relation_flip':flip}
 selection=[(None,0),('proposal',1),('proposal',2),('graph',1),('graph',2),('graph',3),('relation_disabled',2),('relation_flip',2)]
 boxes=np.tile(pool[q_source],(8,1));queries=np.full(8,int(q_source),np.int64);status=['source']*8
 supports=np.full(8,'0'*64,dtype='U64');masks=np.full(8,'0'*64,dtype='U64');accepted=np.zeros(8,bool)
 for j,(name,index) in enumerate(selection):
  if name is None:continue
  branch=branches[name];boxes[j]=branch['arm_boxes_raw6'][index];queries[j]=branch['arm_query_index'][index];status[j]=branch['arm_status'][index]
  accepted[j]=branch['association_accepted']
  if branch['association_accepted']:
   supports[j]=branch['diagnostics']['selected_support_sha256'];masks[j]=branch['diagnostics']['selected_mask_sha256']
 assert queries[3]==queries[4]==queries[5]
 assert np.array_equal(boxes[3],pool[queries[3]]) and np.array_equal(boxes[1],pool[queries[1]])
 assert np.array_equal(boxes[0],pool[q_source])
 assert np.array_equal(supports[3:6],np.repeat(supports[3],3)) and np.array_equal(masks[3:6],np.repeat(masks[3],3))
 assert np.all(accepted[3:6]==accepted[3])
 if reused:assert queries[7]==queries[4] and np.array_equal(boxes[7],boxes[4])
 return {'arm_names':np.asarray(ARM_NAMES),'arm_query_index':queries,'arm_boxes_raw6':boxes,
  'arm_boxes_eval6':eval_boxes(boxes),'arm_status':np.asarray(status),
  'source_boxes_raw6':pool.copy(),'source_text_scores256':scores.copy(),'q_source':np.array(q_source,np.int64),
  'arm_geometry_anchor_raw6':pool[queries].copy(),'actual_changed_axes':boxes!=pool[q_source],
  'geometry_increment_axes':boxes!=pool[queries],'arm_association_accepted':accepted,
  'arm_support_sha256':supports,'arm_mask_sha256':masks,
  'trace':{'branches':branches,'relation_flip_applicable':flip_info['flip_applied'],
   'nonapplicable_flip_exact_fullgraph_array_reuse':reused,
   'graph_assoc_and_geo_same_query_anchor_support_and_gate':True,
   'all_no_parameter_update':True,'no_old_G_endpoint_used':True,
   'relation_flip_is_diagnostic_not_method_candidate':True}}

def prepare(a):
 if sha(a.plan)!=a.plan_sha256:raise ValueError('Actual root plan SHA required')
 p=read(a.plan)
 if p['stage']!='R15_direct8_from_sealed_semantics' or p['GT_read_permitted'] is not False or p['CPU_geometry_execution_permitted'] is not True or p['model_execution_permitted'] is not False:raise ValueError('Root CPU-only/no-GT execution scope required')
 if p['source_checkpoint_sha256']!=SOURCE_SHA or p['arm_names']!=list(ARM_NAMES):raise ValueError('Exact frozen source and eight roles required')
 if p['scene_config']!=asdict(SCENE_CONFIG) or p['joint_config']!=asdict(JOINT_CONFIG):raise ValueError('Exact prelocked configuration required')
 required_programs={'crossplatform/round15/assemble_direct.py','crossplatform/round15/scene_observation.py','crossplatform/round15/joint_geometry.py','crossplatform/round15/semantic_graph.py','crossplatform/round15/semantic_runner.py','crossplatform/round15/semantic_model_runner.py','crossplatform/round13/input_io.py','crossplatform/round13/geometry_observation.py','crossplatform/round13/association.py','crossplatform/round13/association_runner.py','crossplatform/round7/camera_projection.py','crossplatform/round7/experiment/runner_helpers.py'}
 if not required_programs<=set(p['program_sha256']):raise ValueError('Explicit loaded program closure required')
 for rel,h in p['program_sha256'].items():
  if sha(B/rel)!=h:raise ValueError('Frozen program closure changed '+rel)
 if p['program_sha256'].get(str(Path(__file__).relative_to(B)))!=sha(__file__):raise ValueError('Actual assembler must be bound')
 if sha(p['authorization_path'])!=p['authorization_sha256']:raise ValueError('Root authorization binding')
 data={}
 for key in ('manifest','case_index','frame_index'):
  if sha(p[key])!=p[key+'_sha256']:raise ValueError('Bound input-only registry '+key)
  data[key]=read(p[key])
 rows=data['manifest'];cases={c['id']:c for c in data['case_index']}
 if len(rows)!=204 or len(cases)!=204 or len({r['id'] for r in rows})!=204 or {r['id'] for r in rows}!=set(cases):raise ValueError('Full registered204 source cases required')
 sem=Path(p['semantic_directory'])
 if sha(sem/'SEALED.json')!=p['semantic_seal_sha256']:raise ValueError('Actual semantic seal required before any scene reconstruction')
 seal=verify_seal(sem);sp=read(sem/'registered_plan.json')
 if seal['state'] not in ('sealed_R15_semantic204_three_calls_before_GT','sealed_R15_semantic_engineering4_before_GT') or not seal['model_frozen_parameters_buffers_unchanged'] or not seal['no_generated_code_GT_model_update_or_sourceROI']:raise ValueError('Frozen actual semantic producer scope')
 for key in ('manifest','case_index','frame_index'):
  if p[key+'_sha256']!=sp[key+'_sha256']:raise ValueError('Assembler inputs differ from actual semantic inputs')
 if any(p['program_sha256'].get(rel)!=h for rel,h in seal['program_sha256'].items()):raise ValueError('Actual semantic program closure must also be bound without changing producer code')
 if seal['manifest_sha256']!=p['manifest_sha256'] or seal['n_model_calls']!=seal['n_cases']*3 or seal['n_cases']!=p['expected_samples']:raise ValueError('Complete semantic case/call coverage')
 if seal['n_cases']==204:selected=list(rows)
 elif seal['n_cases']==4:
  wanted=sp['engineering_case_ids'];lookup={r['id']:r for r in rows}
  if len(wanted)!=4 or len(set(wanted))!=4:raise ValueError('Preselected engineering4 only')
  selected=[lookup[i] for i in wanted]
 else:raise ValueError('Only formal204 or preselected engineering4')
 actual_roles=dict(collections.Counter(cases[r['id']]['role'] for r in selected))
 if actual_roles!=p['expected_role_counts'] or actual_roles!=seal['role_counts']:raise ValueError('Exact fresh/regression role denominators from actual semantic producer')
 expected={str(Path(c[k])/'SEALED.json') for c in cases.values() for k in ('source_dir','mask_dir')}
 if set(p['cache_seals_sha256'])!=expected or p['cache_seals_sha256']!=seal['cache_seals_sha256']:raise ValueError('Exact source/mask producer closure')
 for rel,h in p['cache_seals_sha256'].items():
  f=B/rel
  if sha(f)!=h:raise ValueError('Source/mask seal changed')
  ss=verify_seal(f.parent)
  if 'source_checkpoint_sha256' in ss and ss['source_checkpoint_sha256']!=SOURCE_SHA:raise ValueError('Waymo-only source100 provenance')
 if a.output.exists():raise ValueError('New output directory only; preserve failed attempts')
 return p,data,cases,selected,sem,seal

def main(a):
 start=time.perf_counter();p,data,cases,rows,sem,semseal=prepare(a);out=a.output;out.mkdir(parents=True);atomic_json(out/'registered_plan.json',p)
 (out/'cases_PRIVATE').mkdir();(out/'trace_PRIVATE').mkdir();counts=collections.Counter();roles=collections.Counter()
 try:
  for i,row in enumerate(rows):
   case=cases[row['id']];alias=case['case_alias']
   if not alias or any(not(c.isalnum() or c=='_') for c in alias):raise ValueError('Safe preregistered case alias')
   payload_path=sem/'cases_PRIVATE'/(alias+'.json');payload=read(payload_path)
   if semseal['files'][str(payload_path.relative_to(sem))]!=sha(payload_path) or payload['sample_id_PRIVATE']!=row['id'] or payload['case_alias']!=alias or payload['role']!=case['role']:raise ValueError('Actual sealed semantic case identity')
   frame,cache,bindings=load_case(row,case,data['frame_index'])
   if bindings!=payload['bindings']:raise ValueError('Source/mask/image/raw observation binding changed since semantics')
   h,w=frame['rgb'].shape[:2];scene=build_scene_observation(raw_points=frame['raw_points'],camera_spec=frame['camera'],image_size_wh=(w,h),instances=frame['instances'],image_sha256=bindings['image_sha256'])
   metadata=scene_metadata(scene);metadata_path=sem/'scene_metadata_PRIVATE'/(alias+'.json')
   if canonical_sha(metadata)!=payload['scene_metadata_sha256'] or canonical_sha(metadata)!=canonical_sha(read(metadata_path)) or semseal['files'][str(metadata_path.relative_to(sem))]!=sha(metadata_path):raise ValueError('Independent scene metadata rebuild bridge failed')
   result=compose_direct_arrays(scene=scene,source_pool=cache['boxes_raw6'][0],text_scores=cache['text_scores'][0],q_source=int(cache['query_index']),semantics=payload['semantics'])
   name='cases_PRIVATE/'+alias+'.npz';npz={k:v for k,v in result.items() if k!='trace'}
   npz.update(sample_id=np.asarray(row['id']),case_alias=np.asarray(alias),case_role=np.asarray(case['role']),platform=np.asarray(row['platform']),semantic_case_sha256=np.asarray(sha(payload_path)),scene_metadata_sha256=np.asarray(canonical_sha(metadata)),source_cache_sha256=np.asarray(bindings['source_cache_sha256']),mask_cache_sha256=np.asarray(bindings['mask_cache_sha256']))
   np.savez_compressed(out/name,**npz)
   trace={'sample_id_PRIVATE':row['id'],'case_alias':alias,'case_role':case['role'],'platform':row['platform'],'bindings':bindings,'scene_metadata_sha256':canonical_sha(metadata),'semantics_member_sha256':sha(payload_path),'source_q0':int(cache['query_index']),'arm_query_index':result['arm_query_index'],'arm_status':result['arm_status'],'geometry_trace':result['trace']}
   atomic_json(out/'trace_PRIVATE'/(alias+'.json'),trace)
   roles[case['role']]+=1
   for arm,changed in zip(ARM_NAMES,np.any(result['actual_changed_axes'],axis=1)):counts[arm+'_changed']+=int(changed)
   if (i+1)%10==0:print(json.dumps({'event':'R15_direct_progress','completed':i+1,'expected':len(rows)}),flush=True)
  files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and f.name!='SEALED.json'}
  atomic_json(out/'SEALED.json',{'state':'sealed_R15_eight_direct_arms_before_GT','n_samples':len(rows),'n_cases':len(rows),'n_arms':8,'arm_names':ARM_NAMES,'role_counts':dict(roles),'mechanism_counts':dict(counts),'source_checkpoint_sha256':SOURCE_SHA,'semantic_seal_sha256':sha(sem/'SEALED.json'),'plan_sha256':sha(a.plan),'manifest_sha256':p['manifest_sha256'],'case_index_sha256':p['case_index_sha256'],'frame_index_sha256':p['frame_index_sha256'],'program_sha256':p['program_sha256'],'cache_seals_sha256':p['cache_seals_sha256'],'all_direct_no_model_parameter_update_GT_or_new_taskIoU':True,'no_old_G_endpoint_used':True,'graph_geometry_increment_same_associated_anchor_query_gate_support':True,'scene_metadata_rebuilt_matches_actual_semantic_inputs':True,'seconds_process':time.perf_counter()-start,'files':files})
  print(json.dumps({'state':'R15_direct8_sealed','n':len(rows),'seal_sha256':sha(out/'SEALED.json')}))
 except Exception as exc:
  atomic_json(out/'FAILURE.json',{'state':'R15_direct_attempt_failed_preserved','error':str(exc),'traceback_PRIVATE':traceback.format_exc(),'no_automatic_retry_GT_or_method_change':True});raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',type=Path,required=True);main(parser.parse_args())
