"""UNEXECUTED DEV-only source bridge and two real point-sampling captures.

Frozen Waymo source; no optimizer, G calls, label/official class mask, new box
head, or FINAL input. A must reproduce the canonical old336/R8 source exactly.
The image matching MLP hook is the actual installed source interface.
"""
from pathlib import Path
import sys,random,hashlib,time,argparse,json
import numpy as np
from full_common50 import ROOT,EXP,read,sha,dump,npz,seal
from pool_weights50 import exact

def bseed(row,salt):
 return int(hashlib.sha256((salt+'|'+row['id']+'|'+str(row['seed'])).encode()).hexdigest()[:16],16)

def check(a):
 plan=read(a.plan);grant=read(a.authorization)
 assert grant['state']=='ROOT_AUTHORIZED_R50_DEV_EPISODIC_SOURCE_CAPTURE'
 assert grant['plan_sha256']==sha(a.plan)
 assert plan['mode'] in ('engineering2','DEV336') and plan['no_GT_newG_or_FINAL'] is True
 assert plan['readout']=='native_GTfree_first_lexicon_noun_fallback_all_content'
 assert plan['boxes']=='canonical_old336_R8_source256_raw6_no_updates'
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'FINAL384' not in rel and 'round46/final' not in rel and 'full5912_gpu_inputs' not in rel
   assert sha(ROOT/rel)==h,rel
 old=read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
 if plan['mode']=='DEV336':rows=old;assert plan['n']==336
 else:
  rows=read(ROOT/plan['rows_path']);salt=plan['engineering_selection_salt']
  expected=[min([r for r in old if r['platform']==pl],key=lambda r:hashlib.sha256((salt+r['id']).encode()).hexdigest()) for pl in ('drone','quad')]
  assert rows==expected and plan['n']==2
 assert plan['capture_both_views_for_every_registered_row'] is True
 out=ROOT/plan['output_root'];assert out.is_relative_to(ROOT/'crossplatform/round50') and not out.exists()
 return plan,rows,out

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 a=p.parse_args();a.plan=str(Path(a.plan).resolve());a.authorization=str(Path(a.authorization).resolve())
 plan,rows,out=check(a);out.mkdir(parents=True)
 for d in ('samples_PRIVATE','cases_PRIVATE','calls_PRIVATE','raw_views_PRIVATE'):(out/d).mkdir()
 sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
 from run_diagnostic import load_model,mask_for,fingerprint
 from runner_helpers import sample_points
 import torch
 F=torch.nn.functional;torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
 torch.manual_seed(0);np.random.seed(0);random.seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 source_plan=read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json')
 model,_=load_model(source_plan);fp=fingerprint(model);model.eval().requires_grad_(False)
 mlp=model.contrastive_align_projection_image;assert sum(p.numel() for p in mlp.parameters())==184960
 versions=[(n,p._version) for n,p in model.named_parameters()]
 np.savez_compressed(out/'SOURCE_MATCHING_MLP_PRIVATE.npz',**{n:v.detach().cpu().numpy() for n,v in mlp.state_dict().items()})
 canonical=ROOT/'crossplatform/round24/capture336_attempt001';cs=read(canonical/'SEALED.json')
 hooks_h=[]
 hook=mlp.register_forward_pre_hook(lambda m,args:hooks_h.append(args[0].detach().clone()))
 start=time.perf_counter();calls=0;receipts=[]
 def one(row,view):
  nonlocal calls
  points,indices,nraw=sample_points(row);text=row['caption']+' . not mentioned'
  mask,receipt=mask_for(model,text,'all_content_tokens');tm=torch.as_tensor(mask,device='cuda',dtype=torch.bool)
  hooks_h.clear();calls+=1;callpath=out/'calls_PRIVATE'/(row['id']+'_'+view+'.json');assert not callpath.exists()
  callrec={'state':'STARTED','id_PRIVATE':row['id'],'view':view,'model_forward_entered':False,
   'point_sample_sha256':hashlib.sha256(points.tobytes()).hexdigest(),'point_indices_sha256':hashlib.sha256(indices.tobytes()).hexdigest(),
   'actual_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'raw_point_count':int(nraw),'source_checkpoint_sha256':source_plan['source_checkpoint_sha256']};dump(callpath,callrec)
  try:
   pc=torch.tensor(points,device='cuda')[None]
   callrec['model_forward_entered']=True;dump(callpath,callrec)
   with torch.inference_mode():o=model({'point_clouds':pc,'text':[text]})
  except BaseException as error:
   callrec.update(state='ERROR_PRESERVED_NO_RETRY',error_type=type(error).__name__,error=str(error));dump(callpath,callrec);raise
  assert len(hooks_h)==7 and all(not m.training for m in model.modules())
  hh=hooks_h[-1][0].clone();t=o['proj_tokens'][0].clone();q=o['last_proj_queries'][0].clone()
  # Preserve the actual batch-1 model matmul/readout path for the canonical
  # source bridge; a mathematically equal 2D matmul may change FP32 kernels.
  native=((((o['last_proj_queries']@o['proj_tokens'].transpose(-1,-2))/.07).softmax(-1))*tm[None,None]).sum(-1)[0]
  boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0]
  rawpath=out/'raw_views_PRIVATE'/(row['id']+'_'+view+'.npz')
  np.savez_compressed(rawpath,boxes=boxes.cpu().numpy(),scores=native.cpu().numpy(),h=hh.cpu().numpy(),query=q.cpu().numpy(),tokens=t.cpu().numpy(),token_mask=np.asarray(mask,bool),point_indices=indices,raw_point_count=np.array(nraw,np.int64))
  callrec.update(state='RAW_MODEL_OUTPUT_SAVED',raw_file_sha256=sha(rawpath));dump(callpath,callrec)
  with torch.inference_mode():qreplay=F.normalize(mlp(hh[None]),dim=-1);replay=((qreplay@t[None].transpose(-1,-2)/.07).softmax(-1)*tm[None,None]).sum(-1)[0]
  assert torch.isfinite(native).all() and int(native.argmax())==int(replay.argmax())
  err=float((native-replay).abs().max());assert err<=plan['initial_head_native_max_abs_bound']
  callrec.update(state='COMPLETED',head_replay_max_abs_error=err,head_replay_argmax_exact=True);dump(callpath,callrec)
  return ({'boxes':boxes.cpu().numpy(),'scores':native.cpu().numpy(),'h':hh.cpu().numpy(),'query':q.cpu().numpy(),
   'tokens':t.cpu().numpy(),'token_mask':np.asarray(mask,bool),'point_indices':indices,'raw_point_count':np.array(nraw,np.int64),'point_sha256':hashlib.sha256(points.tobytes()).hexdigest()},
   {'view':view,'raw_point_count':int(nraw),'head_replay_max_abs_error':err,'head_replay_argmax_exact':True,'mask':receipt,'point_sample_sha256':hashlib.sha256(points.tobytes()).hexdigest()})
 try:
  for i,row in enumerate(rows):
   # Source projection is never adapted here. Preserve the source full-state
   # before a later head-only episodic stage, whose optimizer is per sample.
   aa,ra=one(row,'A_original_seed');rbrow=dict(row,seed=bseed(row,plan['B_sampling_salt']));bb,rb=one(rbrow,'B_resampled_same_coordinates')
   cp=canonical/'samples_PRIVATE'/(row['id']+'.npz');assert sha(cp)==cs['files'][cp.relative_to(canonical).as_posix()]
   old=npz(cp);exact(aa['boxes'],old['boxes'],'canonical_A_source_boxes');exact(aa['scores'],old['scores'],'canonical_A_native_scores')
   assert aa['point_indices'].shape==bb['point_indices'].shape==(16384,) and aa['point_sha256']!=bb['point_sha256']
   payload={prefix+'_'+k:v for prefix,z in [('A',aa),('B',bb)] for k,v in z.items()}
   np.savez_compressed(out/'samples_PRIVATE'/(row['id']+'.npz'),**payload)
   r={'id_PRIVATE':row['id'],'platform':row['platform'],'source_A_boxes_scores_canonical_bitexact':True,
    'canonical_source_file_sha256':sha(cp),'actual_model_text_sha256':hashlib.sha256((row['caption']+' . not mentioned').encode()).hexdigest(),
    'source_checkpoint_sha256':source_plan['source_checkpoint_sha256'],'A_seed':row['seed'],'B_seed':rbrow['seed'],
    'views':[ra,rb],'geometric_inverse':'identity_sampling_only_no_rotation_flip',
    'new_grounder_forwards':2,'no_tasklabels_or_annotated_class_span':True}
   dump(out/'cases_PRIVATE'/(row['id']+'.json'),r);receipts.append(r)
   if (i+1)%8==0:print('R50_EPISODIC_CAPTURE',i+1,len(rows),calls,flush=True)
  assert calls==2*len(rows) and versions==[(n,p._version) for n,p in model.named_parameters()] and fingerprint(model)==fp
 except BaseException as e:
  dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed':len(receipts),'actual_forward_attempts':calls,'no_automatic_retry':True});raise
 finally:hook.remove()
 dump(out/'RECEIPTS_PRIVATE.json',receipts)
 seal(out,'R50_DEV_EPISODIC_TWO_SOURCE_VIEWS_CAPTURED_NO_OPTIMIZER',n=len(rows),grounder_forwards=calls,
  source_unchanged=True,source_fingerprint=fp,matching_parameters=184960,seconds=time.perf_counter()-start,plan_sha256=sha(a.plan))
if __name__=='__main__':main()
