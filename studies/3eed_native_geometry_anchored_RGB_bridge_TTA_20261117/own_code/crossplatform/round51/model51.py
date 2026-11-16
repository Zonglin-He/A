"""R51 single Waymo-only 3EED interface. Import performs no model/sample I/O.

Root owns every execution grant, source bridge and GPU lease. There is no
image/VLM/detector cache reader. Different text phrases have independent query
sets: language enters the cross encoder before top-k query generation.
"""
from pathlib import Path
import hashlib,importlib.util,os,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE_SHA='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
SENTINEL=' . not mentioned'
PREFIX_KEYS=('prefix_query','prefix_vis','prefix_lang','prefix_pos','prefix_query_mask','prefix_text_padding')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fingerprint(model):
 h=hashlib.sha256()
 for name,value in model.state_dict().items():
  a=value.detach().cpu().contiguous()
  h.update(name.encode()+b'\0');h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
 return h.hexdigest()


def strict_runtime():
 import torch
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 return {'matmul_allow_tf32':False,'cudnn_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}


def load_source(source_plan):
 """Same factory/strict state as R15, but only R51's runtime is created."""
 import torch
 plan=source_plan;assert plan['source_checkpoint_sha256']==SOURCE_SHA
 factory_path=ROOT/'crossplatform/round6/factory.py'
 assert sha(factory_path)==plan['r6_factory_sha256']
 assets=(ROOT/'crossplatform/upstream/data/roberta_base').resolve()
 for rel,h in plan['roberta_sha256'].items():assert sha(assets/rel)==h
 upstream=ROOT/'crossplatform/round3/p0/upstream'
 for rel,h in plan['upstream_code_sha256'].items():assert sha(upstream/rel)==h
 runtime=ROOT/'crossplatform/round51/runtime';(runtime/'data').mkdir(parents=True,exist_ok=True)
 link=runtime/'data/roberta_base'
 if not link.exists():link.symlink_to(assets,target_is_directory=True)
 assert link.resolve()==assets
 old_cwd=Path.cwd()
 try:
  os.chdir(runtime)
  spec=importlib.util.spec_from_file_location('R51_original_Waymo_factory',factory_path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  model=module.model()
 finally:os.chdir(old_cwd)
 checkpoint=Path(plan['source_checkpoint_path']);assert sha(checkpoint)==SOURCE_SHA
 state=torch.load(checkpoint,map_location='cpu',weights_only=True)
 assert state['epoch']==100 and state['trained_platforms']==['waymo']
 assert not state['source_provenance']['joint_task_checkpoint_initialization']
 assert state['source_provenance']['plan_sha256']==plan['source_training_plan_sha256']
 assert all(torch.isfinite(x).all() for x in state['model'].values() if x.is_floating_point())
 model.load_state_dict(state['model'],strict=True);del state
 model.eval().requires_grad_(False).cuda();strict_runtime()
 assert fingerprint(model)==plan['source_model_state_fingerprint']
 assert len(model.decoder)==6 and model.decoder[5].cross_v.embed_dim==288
 assert sum(p.numel() for p in model.contrastive_align_projection_image.parameters())==184960
 return model


def native_matching(queries,tokens,mask):
 """Preserve batch-1 matmul, ALL-token denominator, native p and exact z."""
 import torch
 assert queries.ndim==tokens.ndim==3 and queries.shape[:2]==(1,256)
 assert queries.shape[-1]==tokens.shape[-1]==64 and mask.shape==(tokens.shape[1],)
 assert mask.dtype==torch.bool and mask.any() and (~mask).any()
 logits=queries@tokens.transpose(-1,-2)/.07
 probability=(logits.softmax(-1)*mask[None,None]).sum(-1)[0]
 logodds=(torch.logsumexp(logits.masked_fill(~mask[None,None],-torch.inf),-1)-
          torch.logsumexp(logits.masked_fill(mask[None,None],-torch.inf),-1))[0]
 assert torch.isfinite(probability).all() and torch.isfinite(logodds).all()
 return probability,logodds


def firstnoun_mask(model,text):
 """Actual existing R7 first noun and explicit all-content fallback, no GT."""
 import torch
 sys.path.insert(0,str(ROOT/'crossplatform/round7'))
 from regression_adapter import first_noun_mask
 tokenized=model.tokenizer([text],padding='longest',return_tensors='pt',return_offsets_mapping=True)
 offsets=tokenized['offset_mapping'][0];assert len(offsets)<=256
 try:mask=first_noun_mask(text,offsets);fallback=False
 except ValueError:mask=offsets[:,1]>offsets[:,0];fallback=True
 assert mask.any() and (~mask).any()
 return torch.as_tensor(mask,dtype=torch.bool),{'readout':'all_content_tokens' if fallback else 'first_lexicon_noun',
   'fallback':fallback,'tokens':len(offsets),'GT_or_annotated_span_used':False}


def source_capture(model,points,indices,phrase,*,retain_prefix=True,on_forward_enter=None):
 """Root-called full source forward; hooks capture immutable decoder5 inputs."""
 import torch
 assert not any(p.requires_grad for p in model.parameters()), 'Capture all G evidence before installing update interfaces'
 assert all(not m.training for m in model.modules())
 points=np.ascontiguousarray(points,np.float32);indices=np.asarray(indices,np.int64)
 assert points.shape==(16384,3) and indices.shape==(16384,) and np.isfinite(points).all()
 text=phrase+SENTINEL;mask,mask_receipt=firstnoun_mask(model,text)
 device=next(model.parameters()).device;mask=mask.to(device);inputs={};features=[]
 def before(_m,args,kwargs):
  assert len(args)==6 and kwargs.get('detected_feats') is None and kwargs.get('detected_mask') is None
  for name,value in zip(PREFIX_KEYS,args):inputs[name]=None if value is None else value.detach().clone()
 def projection(_m,args):features.append(args[0].detach().clone())
 hooks=([model.decoder[5].register_forward_pre_hook(before,with_kwargs=True),
         model.contrastive_align_projection_image.register_forward_pre_hook(projection)] if retain_prefix else [])
 try:
  point_tensor=torch.as_tensor(points,device=device)[None]
  if on_forward_enter is not None:on_forward_enter()
  with torch.no_grad():out=model({'point_clouds':point_tensor,'text':[text]})
 finally:
  for hook in hooks:hook.remove()
 if retain_prefix:
  assert len(features)==7 and inputs['prefix_query'].shape==(1,256,288)
  assert inputs['prefix_pos'].shape==(1,256,6) and inputs['prefix_query_mask'] is None
 assert not out['text_attention_mask'][0].any(), 'Single phrase only; no hidden padded-token convention change'
 if retain_prefix:
  expected_pos=torch.cat([out['4head_center'],out['4head_pred_size']],-1)
  assert torch.equal(inputs['prefix_pos'],expected_pos), 'Final-layer anchor must equal frozen preceding-head anchor'
 probability,logodds=native_matching(out['last_proj_queries'],out['proj_tokens'],mask)
 boxes=torch.cat([out['last_center'],out['last_pred_size']],-1)[0]
 arrays={'boxes':boxes.detach().cpu().numpy(),'scores':probability.detach().cpu().numpy(),
  'logodds':logodds.detach().cpu().numpy(),'query_anchor_xyz':out['query_points_xyz'][0].detach().cpu().numpy(),
  'query_points_sample_inds':out['query_points_sample_inds'][0].detach().cpu().numpy(),
  'point_indices':indices.copy()}
 if retain_prefix:
  arrays.update(h=features[-1][0].cpu().numpy(),query=out['last_proj_queries'][0].detach().cpu().numpy(),
    tokens=out['proj_tokens'][0].detach().cpu().numpy(),token_mask=mask.cpu().numpy(),prefix_query_mask_is_none=np.array(True))
 for key,value in inputs.items():
  if value is not None:arrays[key]=value.cpu().numpy()
 receipt={'actual_model_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
  'phrase_sha256':hashlib.sha256(phrase.encode()).hexdigest(),'point_sample_sha256':array_sha(points),
  'point_indices_sha256':array_sha(indices),'mask':mask_receipt,
  'prefix_arrays_sha256':{k:array_sha(v) for k,v in arrays.items() if k.startswith('prefix_')},
  'full_last_decoder_prefix_retained':retain_prefix,
  'reference_query_indices_are_NOT_main_query_indices':True,'source_checkpoint_sha256':SOURCE_SHA,
  'original256_centers_sizes_preserved':True,'new_visual_or_external_model_calls':0}
 return arrays,receipt


def capture_phrases(model,points,indices,subject,reference_queries):
 """One model, identical sampled points; no phrase shares another's slot IDs."""
 assert isinstance(reference_queries,dict) and all(isinstance(v,str) and v for v in reference_queries.values())
 result={'subject':source_capture(model,points,indices,subject)}
 for entity,phrase in reference_queries.items():
  assert entity!='subject';result[entity]=source_capture(model,points,indices,phrase)
 return result


def replay_matching(model,cache,mode='mlp',geometry_diagnostic=False):
 """Only original cached slots/boxes are selectable; geometry is diagnostic."""
 import torch
 device=next(model.parameters()).device
 tensor=lambda key:torch.as_tensor(cache[key],device=device)
 if mode=='mlp':h=tensor('h')[None]
 else:
  assert mode=='lora_qv'
  args=[tensor(k) if k in cache else None for k in PREFIX_KEYS]
  assert args[0].shape==(1,256,288) and args[3].shape==(1,256,6) and args[4] is None
  h=model.decoder[5](*args,detected_feats=None,detected_mask=None)
 q=torch.nn.functional.normalize(model.contrastive_align_projection_image(h),p=2,dim=-1)
 p,z=native_matching(q,tensor('tokens')[None],tensor('token_mask').bool())
 result={'probability':p,'logodds':z,'h':h[0],
         'output_boxes':tensor('boxes'),'slot_ids':torch.arange(256,device=device),
         'output_geometry_is_cached_original_not_new_prediction':True}
 if geometry_diagnostic:
  diagnostic={}
  with torch.no_grad():
   center,size=model.prediction_heads[5](h.detach().transpose(1,2).contiguous(),
       base_xyz=tensor('query_anchor_xyz')[None],end_points=diagnostic,prefix='diagnostic_')
  result['discarded_geometry_diagnostic']=torch.cat([center,size],-1)[0]
 return result


def bridge_capture(capture,canonical_boxes,canonical_scores):
 """Root executes before G/TTA. No tolerance or new sample fallback."""
 for name,current,old in [('boxes',capture['boxes'],canonical_boxes),('scores',capture['scores'],canonical_scores)]:
  assert current.shape==old.shape and current.dtype==old.dtype and current.tobytes()==old.tobytes(),name
 return {'original_boxes_and_native_firstnoun_probability_bitexact':True}
