"""Capture or verify shared U128 source cache. GPU inference, no labels/updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,random,numpy as np
from common34 import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores

def main():
 check_plan();out=ROOT/'crossplatform/round33/sourceU128_attempt001'
 if out.exists():
  verify(out);s=read(out/'SEALED.json');assert s['n']==128 and s['source_unchanged'];dump(BASE/'SHARED_U_SOURCE_REUSE.json',{'source_cache':str(out.relative_to(ROOT)),'seal_sha256':sha(out/'SEALED.json'),'verified_existing_cache':True,'new_forwards':0});print('SHARED_U_SOURCE_VERIFIED');return
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);h=[];x=[];t=[];hooks=[model.contrastive_align_projection_image.register_forward_pre_hook(lambda m,a:h.append(a[0].detach().clone())),model.contrastive_align_projection_image[-1].register_forward_pre_hook(lambda m,a:x.append(a[0].detach().clone())),model.contrastive_align_projection_text.register_forward_pre_hook(lambda m,a:t.append(a[0].detach().clone()))]
 state={**{'image.'+k:v.detach().cpu().numpy() for k,v in model.contrastive_align_projection_image.state_dict().items()},**{'text.'+k:v.detach().cpu().numpy() for k,v in model.contrastive_align_projection_text.state_dict().items()}};np.savez_compressed(out/'projection_source_PRIVATE.npz',**state);start=time.perf_counter();shape=None;rr=rows('U128');assert len(rr)==128 and all(r['stage']=='U' for r in rr)
 for i,r in enumerate(rr):
  pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias','split')});shape=list(pc.shape);text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');h.clear();x.clear();t.clear()
  with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
  ss=text_bbf_scores(o,mask)[0].cpu().numpy();bb=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy();assert len(h)==len(x)==7 and len(t)==1
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),h=h[-1][0].cpu().numpy(),x=x[-1][0].cpu().numpy(),text_h=t[0][0].cpu().numpy(),tokens=o['proj_tokens'][0].cpu().numpy(),query=o['last_proj_queries'][0].cpu().numpy(),token_mask=np.asarray(mask,bool),boxes=bb,scores=ss)
  if (i+1)%16==0:print('CAPTURE_U34',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 for hook in hooks:hook.remove()
 assert fingerprint(model)==fp
 seal(out,'shared_R33_R34_U128_frozen_source_cache_before_any_Ulabels',n=128,source_unchanged=True,point_input_shape=shape,source_uses_RGB_point_features=shape[1]==6,plan_R34_sha256=sha(EXP/'PLAN_LOCK.json'),seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('CAPTURE_U34_SEALED')
if __name__=='__main__':main()
