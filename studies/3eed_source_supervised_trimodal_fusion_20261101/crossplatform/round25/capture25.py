"""Frozen original model inputs for source supervised module fitting and targets.
No task GT is read; existing exact frozen target features are reused.
"""
import sys,time,random,argparse,numpy as np
from common25 import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores

def main(pilot=False):
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out=BASE/('pilot32_source_attempt002' if pilot else 'source_features_attempt002');out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();rr=rows('S3682')[:32] if pilot else rows();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);h=[];x=[];t=[];hooks=[model.contrastive_align_projection_image.register_forward_pre_hook(lambda m,a:h.append(a[0].detach().clone())),model.contrastive_align_projection_image[-1].register_forward_pre_hook(lambda m,a:x.append(a[0].detach().clone())),model.contrastive_align_projection_text.register_forward_pre_hook(lambda m,a:t.append(a[0].detach().clone()))];start=time.perf_counter();calls=0;reuse=0;max_tokens=0
 np.savez_compressed(out/'projection_source_PRIVATE.npz',**{**{'image.'+k:v.detach().cpu().numpy() for k,v in model.contrastive_align_projection_image.state_dict().items()},**{'text.'+k:v.detach().cpu().numpy() for k,v in model.contrastive_align_projection_text.state_dict().items()}})
 for i,r in enumerate(rr):
  prior=ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz')
  pilot_cache=BASE/'pilot32_source_attempt002/samples_PRIVATE'/(r['id']+'.npz')
  if not pilot and (prior.exists() or pilot_cache.exists()):
   z=load_npz(prior if prior.exists() else pilot_cache);reuse+=1
  else:
   pc,_,_=sample_points(r)
   if r['platform']=='waymo':
    from utils.transform_waymo import transform_to_front_view
    pc,_=transform_to_front_view(pc,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pc=np.ascontiguousarray(pc,np.float32)
   text=(' '.join(r['caption'].replace(',', ' ,').split()) if r['platform']=='waymo' else r['caption'])+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');h.clear();x.clear();t.clear()
   with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
   calls+=1;assert len(h)==len(x)==7 and len(t)==1;tokens=o['proj_tokens'][0].cpu().numpy();max_tokens=max(max_tokens,len(tokens));z=dict(h=h[-1][0].cpu().numpy(),x=x[-1][0].cpu().numpy(),text_h=t[0][0].cpu().numpy(),tokens=tokens,query=o['last_proj_queries'][0].cpu().numpy(),token_mask=np.asarray(mask,bool),boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy(),scores=text_bbf_scores(o,mask)[0].cpu().numpy())
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**z)
  if (i+1)%128==0 or pilot and i==31:print('CAPTURE25',i+1,'calls',calls,'reuse',reuse,'seconds',round(time.perf_counter()-start,1),flush=True)
 for hook in hooks:hook.remove()
 assert fingerprint(model)==fp;seal(out,'R25_frozen_source_features_GT_free',n=len(rr),forwards=calls,reused_exact_features=reuse,source_unchanged=True,max_tokens=max_tokens,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('CAPTURE25_SEALED',flush=True)
if __name__=='__main__':main('--pilot' in sys.argv)
