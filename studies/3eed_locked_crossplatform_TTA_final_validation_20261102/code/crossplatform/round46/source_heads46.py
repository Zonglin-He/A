"""Source capture and actual installed R24/R30 heads; no new adaptation."""
import sys,time,hashlib,random
import numpy as np
from common46 import *
def outside(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():
  if not n.startswith('contrastive_align_projection_image.'):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 a=args(__doc__);plan,rows,out=context(a);folder=out/'source';folder.mkdir();(folder/'samples_PRIVATE').mkdir();(folder/'closure_PRIVATE').mkdir()
 sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
 from run_diagnostic import load_model,mask_for,fingerprint
 from runner_helpers import sample_points
 import torch
 F=torch.nn.functional;torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);outside0=outside(model);original={n:v.detach().clone() for n,v in model.contrastive_align_projection_image.state_dict().items()};h=[];x=[];th=[]
 hooks=[model.contrastive_align_projection_image.register_forward_pre_hook(lambda m,v:h.append(v[0].detach().clone())),model.contrastive_align_projection_image[-1].register_forward_pre_hook(lambda m,v:x.append(v[0].detach().clone())),model.contrastive_align_projection_text.register_forward_pre_hook(lambda m,v:th.append(v[0].detach().clone()))]
 projection=npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz')
 for k,v in original.items():assert np.array_equal(v.cpu().numpy(),projection['image.'+k])
 receipts=[];start=time.perf_counter();maxerr=0.
 for i,r in enumerate(rows):
  points,indices,_=sample_points(r);pc=torch.tensor(points,device='cuda')[None];text=r['caption']+' . not mentioned';mask,maskreceipt=mask_for(model,text,'all_content_tokens');tm=torch.tensor(mask,device='cuda');h.clear();x.clear();th.clear()
  model.contrastive_align_projection_image.load_state_dict(original)
  with torch.inference_mode():o=model({'point_clouds':pc,'text':[text]})
  assert len(h)==len(x)==7 and len(th)==1
  s=((o['last_proj_queries']@o['proj_tokens'].transpose(-1,-2)/.07).softmax(-1)*tm[None,None]).sum(-1)[0];boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0];hh=h[-1][0].clone();tokens=o['proj_tokens'][0].clone();qq=o['last_proj_queries'][0].clone();scores={'source':s.cpu().numpy()};trace=[]
  np.savez_compressed(folder/'samples_PRIVATE'/(r['id']+'.npz'),h=hh.cpu().numpy(),x=x[-1][0].cpu().numpy(),text_h=th[0][0].cpu().numpy(),query=qq.cpu().numpy(),tokens=tokens.cpu().numpy(),token_mask=np.asarray(mask,bool),scores=s.cpu().numpy(),boxes=boxes.cpu().numpy(),point_indices=indices,point_sample_sha256=np.array(hashlib.sha256(points.tobytes()).hexdigest()))
  for name,key in [('R24_query_mlp_lr1_s16','R24_state_template'),('R30_beta1','R30_state_template')]:
   state=npz(ROOT/plan[key].format(platform=r['platform']));model.contrastive_align_projection_image.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()})
   with torch.inference_mode():oo=model({'point_clouds':pc,'text':[text]});ss=((oo['last_proj_queries']@oo['proj_tokens'].transpose(-1,-2)/.07).softmax(-1)*tm[None,None]).sum(-1)[0];bb=torch.cat([oo['last_center'],oo['last_pred_size']],-1)[0];cached=((F.normalize(model.contrastive_align_projection_image(hh),dim=-1)@tokens.T/.07).softmax(-1)*tm).sum(-1)
   er=float((ss-cached).abs().max());maxerr=max(maxerr,er);assert er<2e-5 and int(ss.argmax())==int(cached.argmax()) and torch.equal(bb,boxes)
   scores[name]=ss.cpu().numpy();trace.append({'arm':name,'cached_score_maxerror':er,'argmax_exact':True,'boxes_exact':True})
  model.contrastive_align_projection_image.load_state_dict(original);np.savez_compressed(folder/'closure_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(list(scores)),scores=np.stack(list(scores.values())),queries=np.array([int(v.argmax()) for v in scores.values()]),boxes=boxes.cpu().numpy());receipts.append({'alias':r.get('alias',r['id']),'platform':r['platform'],'mask':maskreceipt,'closure':trace});dump(folder/'PROGRESS.json',{'completed':i+1,'n':len(rows)})
 for hook in hooks:hook.remove()
 afterfp=fingerprint(model);afteroutside=outside(model);assert afterfp==fp and afteroutside==outside0;dump(folder/'CLOSURE_AUDIT.json',{'rows':receipts,'outside_projection_unchanged':True,'outside_projection_sha256_before':outside0,'outside_projection_sha256_after':afteroutside,'full_source_fingerprint_before':fp,'full_source_fingerprint_after':afterfp,'source_restored_exact':True,'cached_score_maxerror':maxerr,'full_model_forwards':len(rows)*3,'readout':'GTfree_first_lexicon_noun_fallback_all_content_tokens_not_official_GTclass_positive_map'})
 seal(folder,'R46_SOURCE_AND_TWO_INSTALLED_HEADS_PREDICTIONS_NO_GT',n=len(rows),source_unchanged=True,boxes_exact=True,full_model_forwards=3*len(rows),seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('R46_SOURCE_HEADS_SEALED',len(rows),flush=True)
if __name__=='__main__':main()
