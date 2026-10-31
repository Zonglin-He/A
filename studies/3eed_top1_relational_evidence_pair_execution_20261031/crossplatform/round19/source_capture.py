"""Frozen source features for H32 only, no RGB/model changes/task GT."""
import sys,time,random
import numpy as np
from common import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
def main():
 locked('SOURCE_LOCK.json');out=ROOT/'crossplatform/round19/sourceH32_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir()
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);xx=[];hook=model.contrastive_align_projection_image[-1].register_forward_pre_hook(lambda m,a:xx.append(a[0].detach().clone()));start=time.perf_counter()
 for i,r0 in enumerate(read(EXP.parent/'input/H32_PRIVATE.json')):
  r={k:v for k,v in r0.items() if k not in ('stage','alias','frame_alias')};pc,_,_=sample_points(r);text=r['caption']+' . not mentioned';mask,_=mask_for(model,text,'all_content_tokens');xx.clear()
  with torch.inference_mode():o=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
  from regression_adapter import text_bbf_scores
  ss=text_bbf_scores(o,mask).cpu().numpy()[0];bb=torch.cat([o['last_center'],o['last_pred_size']],-1).cpu().numpy()[0];assert len(xx)==7
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),x=xx[-1][0].cpu().numpy(),tokens=o['proj_tokens'][0].cpu().numpy(),query=o['last_proj_queries'][0].cpu().numpy(),token_mask=np.asarray(mask,bool),boxes=bb,scores=ss,sample_id=np.array(r['id']))
  if i in (0,16):
   with torch.inference_mode():rep=model({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
   assert np.array_equal(ss,text_bbf_scores(rep,mask).cpu().numpy()[0]) and np.array_equal(bb,torch.cat([rep['last_center'],rep['last_pred_size']],-1).cpu().numpy()[0])
  print('SOURCE_H',i+1,flush=True)
 hook.remove();assert fingerprint(model)==fp;seal(out,'R19_H32_frozen_source_features_no_GT',n=32,source_unchanged=True,source_repeat_bitexact_two_probes=True,seconds=time.perf_counter()-start)
if __name__=='__main__':main()
