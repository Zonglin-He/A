"""Root-executed R14 Quad cached A-only regression transfer, no target-GT API.

This file does not create source/G predictions. It reuses frozen source caches
and R13 A direct targets. One optional B sensor/text eval-forward validates cache
transport, AFTER a true A-only step, and never informs fit/gates/LR/query choice.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,random,re,sys,time,traceback
import numpy as np
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
sys.path[:0]=[str(D),str(D.parent/'round7'),str(D.parent/'round7/experiment')]
from runner_helpers import sha,read,eval_boxes,to_json,array_sha
from quad_transfer import ARM_NAMES,DEFAULT_CONFIG,run_transfer_episode
STRICT_FLAGS={'cudnn_allow_tf32':False,'matmul_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
PAIR_KEYS={'pair_id','scene','A_id','B_id','A_cache_path','B_cache_path','A_geometry_path','B_observation_path',
           'A_image_sha256','A_point_sha256','B_image_sha256','B_point_sha256'}


def fingerprint(model):
 h=hashlib.sha256()
 for name,v in model.state_dict().items():
  x=v.detach().cpu().contiguous();h.update(name.encode()+b'\0');h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(to_json(obj),indent=2,allow_nan=False)+'\n')
def bound(path,digest):
 if sha(path)!=digest:raise ValueError('Binding mismatch '+str(path))
def verify_artifact(item):
 directory=Path(item['directory']);bound(directory/'SEALED.json',item['seal_sha256']);s=read(directory/'SEALED.json')
 if not s.get('files'):raise ValueError('Empty allowlist')
 for name,h in s['files'].items():
  rel=Path(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe sealed path')
  bound(directory/rel,h)
 if s.get('source_checkpoint_sha256',SOURCE_SHA)!=SOURCE_SHA:raise ValueError('Source mismatch')
 if any(name.startswith('source_cache/') for name in s['files']):
  if not s.get('source_unchanged') or not s.get('source_repeat_bitexact') or s.get('strict_FP32_flags')!=STRICT_FLAGS:raise ValueError('Frozen strict-FP32 source cache provenance')
 return directory,s

def completed_run_files(output,hook_done):
 """Only actual hook execution creates its optional receipt; zero-prefix valid."""
 names=['registered_plan.json','CACHE_BRIDGES.jsonl','TRACE_PRIVATE.jsonl','TRACE_PRIVATE.json']
 if hook_done:names.append('FIRST_NONZERO_A_FULLMODEL_B_CHECK.json')
 return {name:sha(output/name) for name in names}

def load_model(context):
 import torch
 factoryfile=ROOT/'crossplatform/round6/factory.py';bound(factoryfile,context['r6_factory_sha256'])
 runtime=Path(context['runtime_dir']).resolve()
 if runtime!=(D/'runtime_quad').resolve():raise ValueError('Runtime must remain in new R14 namespace')
 (runtime/'data').mkdir(parents=True,exist_ok=True)
 assets=(D.parent/'upstream/data/roberta_base').resolve();link=runtime/'data/roberta_base'
 if not link.exists():link.symlink_to(assets,target_is_directory=True)
 if link.resolve()!=assets:raise ValueError('Generic RoBERTa runtime link mismatch')
 spec=importlib.util.spec_from_file_location('R14_readonly_source100_factory',factoryfile)
 factory=importlib.util.module_from_spec(spec);spec.loader.exec_module(factory)
 for rel,h in context['upstream_code_sha256'].items():bound(factory.U/rel,h)
 for rel,h in context['roberta_sha256'].items():bound(assets/rel,h)
 checkpoint=Path(context['source_checkpoint_path']);bound(checkpoint,context['source_checkpoint_sha256'])
 if context['source_checkpoint_sha256']!=SOURCE_SHA:raise ValueError('Only audited Waymo-only source100')
 previous=Path.cwd()
 try:os.chdir(runtime);model=factory.model()
 finally:os.chdir(previous)
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 if state['epoch']!=100 or state['trained_platforms']!=['waymo'] or state['source_provenance']['joint_task_checkpoint_initialization']:raise ValueError('Source provenance')
 if state['source_provenance']['plan_sha256']!=context['source_training_plan_sha256']:raise ValueError('Training plan')
 if not all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point()):raise ValueError('Nonfinite source')
 model.load_state_dict(state['model'],strict=True);entries=len(state['model']);del state
 model.requires_grad_(False).cuda().eval()
 if fingerprint(model)!=context['source_model_state_fingerprint']:raise ValueError('Full source fingerprint')
 return model,entries

def load_cache(path,device):
 import torch
 from regression_adapter import SourceCache
 with np.load(path,allow_pickle=False) as z:
  fields={k:torch.from_numpy(z[k].copy()).to(device) for k in ('base_xyz','center_features','size_features','boxes_raw6','text_scores')}
  cache=SourceCache(**fields,query_index=int(z['query_index']),selected_parameter_sha256=str(z['selected_parameter_sha256'].item()))
  receipt={k:z[k].copy() for k in ('sample_id','point_sample_sha256','raw_indices_sha256','actual_model_text_sha256','source_checkpoint_sha256')}
 if str(receipt['source_checkpoint_sha256'].item())!=SOURCE_SHA:raise ValueError('Cache source')
 return cache,receipt


def prepare(plan):
 if plan['stage']!='R14_Quad_distinct_content_Aonly_regression_transfer' or not plan['GPU_execution_permitted'] or plan['GT_read_permitted']:raise ValueError('Root execution scope')
 if plan['strict_FP32_flags']!=STRICT_FLAGS or plan['expected_pairs']!=76 or plan['source_checkpoint_sha256']!=SOURCE_SHA:raise ValueError('Fixed protocol')
 if plan['fractions']!=[.5,1.] or plan['parameter_kernel_atol_m']!=2e-4 or plan['A_fit_coordinate_atol_m']!=5e-5 or not plan['fit_full6D'] or plan['same_object_or_known_ego_transform_claimed']:raise ValueError('Fit/control scope')
 for rel,h in plan['program_sha256'].items():bound(ROOT/rel,h)
 if plan['program_sha256']['crossplatform/round14/run_quad_transfer.py']!=sha(Path(__file__)):raise ValueError('Runner not locked')
 for key in ('pairs','model_context','B_manifest','A_geometry_trace'):
  bound(plan[key+'_path'],plan[key+'_sha256'])
 pairs=read(plan['pairs_path']);context=read(plan['model_context_path']);bmanifest=read(plan['B_manifest_path']);brows={r['id']:r for r in bmanifest}
 if len(pairs)!=76 or len({r['pair_id'] for r in pairs})!=76:raise ValueError('Pairs coverage')
 if any(set(p)!=PAIR_KEYS for p in pairs):raise ValueError('Strict pair schema')
 if any(not isinstance(p['pair_id'],str) or re.fullmatch(r'[A-Za-z0-9_-]+',p['pair_id']) is None for p in pairs):raise ValueError('Safe pair ID required')
 if len({p['A_id'] for p in pairs})!=76 or len({p['B_id'] for p in pairs})!=76:raise ValueError('One-to-one')
 if {p['A_id'] for p in pairs}&{p['B_id'] for p in pairs}:raise ValueError('A/B query overlap')
 # All172 audited A/B source pools are globally unique by image AND point hash;
 # actual76 selected pairs must retain independent content on both modalities.
 for field in ('image_sha256','point_sha256'):
  hashes=[p[side+'_'+field] for p in pairs for side in ('A','B')]
  if len(set(hashes))!=len(hashes):raise ValueError('Duplicate physical '+field)
 artifacts={}
 for item in plan['sealed_artifacts']:
  base,s=verify_artifact(item);artifacts[base.resolve()]=s
 def is_sealed(path):
  p=Path(path).resolve()
  return any(p.is_relative_to(base) and str(p.relative_to(base)) in s['files'] and sha(p)==s['files'][str(p.relative_to(base))] for base,s in artifacts.items())
 capture_traces={}
 for base,seal in artifacts.items():
  if 'trace.json' in seal['files']:
   tr=read(base/'trace.json')
   if isinstance(tr,list) and all(isinstance(x,dict) and 'id' in x and 'image_sha256' in x and 'point_file_sha256' in x for x in tr):
    capture_traces[base]={x['id']:x for x in tr}
 for pair in pairs:
  for key in ('A_cache_path','B_cache_path','A_geometry_path','B_observation_path'):
   if not is_sealed(pair[key]):raise ValueError('Pair member not in registered source/G allowlists')
  for side in ('A','B'):
   cachepath=Path(pair[side+'_cache_path']).resolve()
   matches=[tr[pair[side+'_id']] for base,tr in capture_traces.items() if cachepath.is_relative_to(base) and pair[side+'_id'] in tr]
   if len(matches)!=1 or matches[0]['image_sha256']!=pair[side+'_image_sha256'] or matches[0]['point_file_sha256']!=pair[side+'_point_sha256']:raise ValueError('Content hash table differs from sealed source-capture provenance')
  if pair['B_id'] not in brows or brows[pair['B_id']]['platform']!='quad' or brows[pair['B_id']]['scene']!=pair['scene']:raise ValueError('B pure-input join')
 gbase=Path(plan['A_geometry_trace_path']).parent.resolve()
 if gbase not in artifacts or artifacts[gbase].get('state')!='sealed_R13_fixedq0_inequality_vs_source_and_surface_direct_no_GT' or not artifacts[gbase].get('all_q_fixed') or not artifacts[gbase].get('no_neural_update_or_GPU_or_GT'):raise ValueError('Original pre-GT R13 fixed-q0 G provenance')
 trace=read(plan['A_geometry_trace_path']);byA={x['id']:x for x in trace}
 if not all(p['A_id'] in byA and byA[p['A_id']]['platform']=='quad' and byA[p['A_id']]['scene_PRIVATE']==p['scene'] for p in pairs):raise ValueError('A gate trace join')
 return pairs,context,brows,byA


def full_B_hook_from_cache(pair,row,cache_receipt,plan,output):
 """Return read-only validator. No image, metadata, G, GT or fit access."""
 import torch
 from regression_adapter import text_bbf_scores,first_noun_mask
 with np.load(pair['B_observation_path'],allow_pickle=False) as z:
  points=z['sampled_points_model'].copy();ph=z['point_sample_sha256'].copy();ih=z['raw_indices_sha256'].copy()
 if points.shape!=(16384,3) or points.dtype!=np.float32 or not np.array_equal(ph,cache_receipt['point_sample_sha256']) or not np.array_equal(ih,cache_receipt['raw_indices_sha256']):raise ValueError('B source observation input bridge')
 if array_sha(points)!=ph.tobytes().hex():raise ValueError('B sampled points hash')
 text=row['caption']+plan['sentinel']
 if hashlib.sha256(text.encode()).digest()!=cache_receipt['actual_model_text_sha256'].tobytes():raise ValueError('B exact model text')
 def hook(model,cache_B,expected_pool):
  tok=model.tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True)
  try:tokenmask=first_noun_mask(text,tok['offset_mapping'][0])
  except ValueError:
   if plan['noun_fallback']!='all_content_tokens':raise
   off=tok['offset_mapping'][0];tokenmask=off[:,1]>off[:,0]
  torch.cuda.synchronize();start=time.perf_counter()
  with torch.no_grad():
   out=model({'point_clouds':torch.from_numpy(points)[None].to(expected_pool.device),'text':[text]})
   raw=torch.cat((out['last_center'],out['last_pred_size']),-1)
   scores=text_bbf_scores(out,tokenmask)
  torch.cuda.synchronize()
  record={'state':'observed_readonly_fullmodel_B_after_A_step_before_restore',
     'B_id_PRIVATE':pair['B_id'],'B_q_source_fixed':cache_B.query_index,
     'B_source_current_text_argmax':int(scores.argmax()),'query_reselected_for_output':False,
     'all256_bbox_max_abs_error_vs_adapted_cache_head_m':float((raw-expected_pool).abs().max()),
     'selected_bbox_max_abs_error_vs_adapted_cache_head_m':float((raw[0,cache_B.query_index]-expected_pool[0,cache_B.query_index]).abs().max()),
     'source_scores_max_abs_error':float((scores-cache_B.text_scores).abs().max()),
     'coordinate_atol_m':2e-4,'seconds':time.perf_counter()-start,
     'passed':bool((raw-expected_pool).abs().max()<=2e-4),
     'B_used_for_loss_gate_LR_or_G':False,'sample_point_sha256':ph.tobytes().hex(),
     'model_text_sha256':cache_receipt['actual_model_text_sha256'].tobytes().hex()}
  # Always persist observed values before any caller engineering pass/fail.
  dump(output/'FIRST_NONZERO_A_FULLMODEL_B_CHECK.json',record)
  return record
 return hook


def main(a):
 bound(a.plan,a.plan_sha256);plan=read(a.plan);pairs,context,brows,gtrace=prepare(plan)
 if a.limit is not None:
  if a.limit!=2 or not plan['allow_fixed_prefix2_engineering']:raise ValueError('Only fixed first2 preflight permitted')
  pairs=pairs[:2]
 engineering=a.limit is not None
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
 import torch
 from regression_adapter import parameter_sha,raw_boxes
 random.seed(plan['seed']);np.random.seed(plan['seed']);torch.manual_seed(plan['seed']);torch.set_num_threads(plan.get('torch_num_threads',4))
 torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 if not torch.cuda.is_available():raise RuntimeError('Root GPU execution requires actual CUDA')
 started=time.perf_counter();files={};traces=[];hook_done=False
 try:
  model,entries=load_model(context);basefp=fingerprint(model);head=parameter_sha(model)
  for i,pair in enumerate(pairs):
   tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   ca,ra=load_cache(pair['A_cache_path'],'cuda');cb,rb=load_cache(pair['B_cache_path'],'cuda')
   if str(ra['sample_id'].item())!=pair['A_id'] or str(rb['sample_id'].item())!=pair['B_id']:raise ValueError('Cache pair IDs')
   if ca.selected_parameter_sha256!=head or cb.selected_parameter_sha256!=head:raise ValueError('Source selected head')
   with torch.no_grad():a_replay=raw_boxes(model,ca);b_replay=raw_boxes(model,cb)
   bridges={'A_bitexact':bool(torch.equal(a_replay,ca.boxes_raw6)),'B_bitexact':bool(torch.equal(b_replay,cb.boxes_raw6)),
      'A_max_abs_error':float((a_replay-ca.boxes_raw6).abs().max()),'B_max_abs_error':float((b_replay-cb.boxes_raw6).abs().max())}
   with (a.output/'CACHE_BRIDGES.jsonl').open('a') as f:f.write(json.dumps({'pair_id':pair['pair_id'],**bridges})+'\n')
   if not bridges['A_bitexact'] or not bridges['B_bitexact']:raise ValueError('Source cached Conv replay is not bitexact; preserve failure')
   with np.load(pair['A_geometry_path'],allow_pickle=False) as z:
    names=z['arm_names'].tolist();sourceA=z['arm_boxes_raw6'][names.index('source')].copy();target=z['arm_boxes_raw6'][names.index('B_inequality_full')].copy();gq=int(z['q_source']);gstatus=str(z['arm_status'][names.index('B_inequality_full')])
   if gq!=ca.query_index or not np.array_equal(sourceA,ca.boxes_raw6[0,ca.query_index].cpu().numpy()):raise ValueError('Frozen A G/source bridge')
   gate=bool(gtrace[pair['A_id']]['inequality']['gate_passed'])
   nonzero=bool(gate and np.any(target!=sourceA) and np.all(sourceA[3:]>0) and np.all(target[3:]>0))
   hook=full_B_hook_from_cache(pair,brows[pair['B_id']],rb,plan,a.output) if nonzero and not hook_done else None
   result=run_transfer_episode(model,ca,cb,target,gate,full_B_forward_hook=hook)
   if hook is not None:hook_done=True
   raw=result['B_boxes_raw6'].cpu().numpy();qB=cb.query_index
   if not np.array_equal(raw[0],cb.boxes_raw6[0,qB].cpu().numpy()):raise ValueError('B fixed source output')
   name=pair['pair_id']+'.npz'
   np.savez_compressed(a.output/name,pair_id=np.array(pair['pair_id']),A_id=np.array(pair['A_id']),B_id=np.array(pair['B_id']),
     arm_names=np.array(ARM_NAMES),arm_query_index=np.full(7,qB,np.int64),arm_boxes_raw6=raw,arm_boxes_eval6=eval_boxes(raw),arm_status=np.array(result['arm_status']),
     source_boxes_raw6=cb.boxes_raw6[0].cpu().numpy(),source_text_scores256=cb.text_scores[0].cpu().numpy(),q_source=np.array(qB),
     A_q_source=np.array(ca.query_index),A_source6=sourceA,A_target6=target,A_frozen_G_status=np.array(gstatus),A_gate=np.array(gate),A_nonzero_residual=np.array(nonzero),
     A_source_cache_sha256=np.array(sha(pair['A_cache_path'])),B_source_cache_sha256=np.array(sha(pair['B_cache_path'])),
     A_geometry_sha256=np.array(sha(pair['A_geometry_path'])),B_point_sample_sha256=rb['point_sample_sha256'],B_raw_indices_sha256=rb['raw_indices_sha256'])
   files[name]=sha(a.output/name)
   tr={k:v for k,v in result.items() if k not in ('B_boxes_raw6','source_text_scores_B_immutable')}
   tr.update(pair_id=pair['pair_id'],A_id_PRIVATE=pair['A_id'],B_id_PRIVATE=pair['B_id'],scene_PRIVATE=pair['scene'],
      A_frozen_G_status=gstatus,cache_bridges=bridges,seconds=time.perf_counter()-tick,peak_allocated_bytes=int(torch.cuda.max_memory_allocated()))
   traces.append(to_json(tr))
   with (a.output/'TRACE_PRIVATE.jsonl').open('a') as f:f.write(json.dumps(to_json(tr),allow_nan=False)+'\n')
   # Full observed numeric failures are durable before checking engineering gates.
   if not result['engineering_checks_pass'] or result['full_B_forward_hook'] is not None and not result['full_B_forward_hook']['passed']:
    raise ValueError('R14 measured engineering equivalence/restore/fullforward failure; do not relax tolerances')
   if parameter_sha(model)!=head or any(p.requires_grad or p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):raise ValueError('Episode source state not restored')
   if (i+1)%10==0 or i+1==len(pairs):print(json.dumps({'completed':i+1,'pairs':len(pairs),'seconds':time.perf_counter()-started,'nonzero_A_sofar':sum(x['A_update_executed'] for x in traces)}),flush=True)
  if not hook_done and not engineering:raise ValueError('No nonzero A available for locked fullmodel B engineering check; do not manufacture a target')
  if fingerprint(model)!=basefp:raise ValueError('Full source state fingerprint changed')
  dump(a.output/'TRACE_PRIVATE.json',traces)
  files.update(completed_run_files(a.output,hook_done))
  seal={'state':'sealed_R14_fixed_prefix2_engineering_no_GT' if engineering else 'sealed_R14_Quad76_Aonly_transfer7arms_before_B_GT','n_pairs':len(pairs),'n_arms':7,'arm_names':ARM_NAMES,
    'source_checkpoint_sha256':SOURCE_SHA,'source_model_state_fingerprint':basefp,'model_state_entries':entries,
    'plan_sha256':sha(a.plan),'pairs_sha256':plan['pairs_sha256'],'model_context_sha256':plan['model_context_sha256'],
    'program_sha256':plan['program_sha256'],'strict_FP32_flags':STRICT_FLAGS,
    'all_B_fixed_own_source_q':True,'B_in_loss_gate_LR_or_G':False,'A_and_B_content_disjoint':True,
    'source_state_restored':True,'full_B_forward_check_done':hook_done,'source_training_teacher_EMA_or_GT_read':False,
    'scope':'fixed_first2_order_engineering_not_taskscore' if engineering else 'formal_all76_rerun_separate_output_no_GT',
    'first_nonzero_A_fullmodel_B_check_missing_due_fixed_prefix_zero':bool(engineering and not hook_done),
    'nonzero_A_n':sum(x['A_update_executed'] for x in traces),'seconds':time.perf_counter()-started,'files':files}
  dump(a.output/'SEALED.json',seal);print(json.dumps({'state':seal['state'],'seal_sha256':sha(a.output/'SEALED.json')}))
 except Exception as exc:
  dump(a.output/'FAILURE.json',{'state':'failed_R14_transfer_attempt_preserved','exception':type(exc).__name__,'message':str(exc),
       'traceback_PRIVATE':traceback.format_exc(),'plan_sha256':sha(a.plan),'seconds':time.perf_counter()-started,
       'no_metric_or_tolerance_repair_or_retry_performed':True})
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);main(p.parse_args())
