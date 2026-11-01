import sys,time,random,numpy as np
from common32 import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from regression_adapter import text_bbf_scores

def main():
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out=BASE/'sourceT128_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);h=[];x=[];t=[]
 hooks=[model.contrastive_align_projection_image.register_forward_pre_hook(lambda m,a:h.append(a[0].detach().clone())),model.contrastive_align_projection_image[-1].register_forward_pre_hook(lambda m,a:x.append(a[0].detach().clone())),model.contrastive_align_projection_text.register_forward_pre_hook(lambda m,a:t.append(a[0].detach().clone()))]
 state={**{'image.'+k:v.detach().cpu().numpy() for k,v in model.contrastive_align_projection_image.state_dict().items()},**{'text.'+k:v.detach().cpu().numpy() for k,v in model.contrastive_align_projection_text.state_dict().items()}}
 np.savez_compressed(out/'projection_source_PRIVATE.npz',**state);start=time.perf_counter();maxerr=0.;maxbox=0.;inputshape=None
 for i,r in enumerate(rows('T128')):
  pc,_,_=sample_points({k:v for k,v in r.items() if k not in ('stage','alias','frame_alias')});inputshape=list(pc.shape);text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');h.clear();x.clear();t.clear()
  with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
  ss=text_bbf_scores(o,mask).cpu().numpy()[0];bb=torch.cat([o['last_center'],o['last_pred_size']],-1).cpu().numpy()[0];assert len(h)==len(x)==7 and len(t)==1
  if False:
   z=previous_feature(r);err=float(abs(z['scores']-ss).max());be=float(abs(z['boxes']-bb).max());maxerr=max(maxerr,err);maxbox=max(maxbox,be);assert err<2e-5 and be<1e-4,(r['alias'],err,be)
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),h=h[-1][0].cpu().numpy(),x=x[-1][0].cpu().numpy(),text_h=t[0][0].cpu().numpy(),tokens=o['proj_tokens'][0].cpu().numpy(),query=o['last_proj_queries'][0].cpu().numpy(),token_mask=np.asarray(mask,bool),boxes=bb,scores=ss)
  if (i+1)%16==0:print('CAPTURE32',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
 for hook in hooks:hook.remove()
 assert fingerprint(model)==fp
 seal(out,'R32_T128_frozen_source_full_projection_inputs_no_GT',n=128,source_unchanged=True,prior_score_maxerror=maxerr,prior_box_maxerror=maxbox,point_input_shape=inputshape,source_uses_RGB_point_features=inputshape[1]==6,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('CAPTURE32_SEALED',flush=True)
if __name__=='__main__':main()
