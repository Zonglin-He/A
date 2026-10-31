"""Capture the existing query-projector final-linear inputs and fixed tokens.
All64 original source forwards. B uses point cloud/text only; no RGB/G/GT.
"""
from pathlib import Path
import sys,time,random
import numpy as np
from common import ROOT,EXP,read,dump,sha,arrsha,seal,verify,check_plan
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
def main():
 plan=check_plan();rows=read(EXP.parent/'input/ALL64_PRIVATE.json');old=ROOT/'crossplatform/round15/source96_attempt001';verify(old)
 out=ROOT/'crossplatform/round16/features64_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir()
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 start=time.perf_counter();model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model)
 states={k:v.detach().cpu().numpy().copy() for k,v in model.contrastive_align_projection_image.state_dict().items()};np.savez_compressed(out/'projection_source_PRIVATE.npz',**states)
 features=[]
 hook=model.contrastive_align_projection_image[-1].register_forward_pre_hook(lambda m,args:features.append(args[0].detach().clone()))
 traces=[]
 for i,r in enumerate(rows):
  pc,ix,n=sample_points(r);text=r['caption']+' . not mentioned';mask,info=mask_for(model,text,'all_content_tokens');features.clear();t=time.perf_counter()
  with torch.inference_mode():o=model({'point_clouds':torch.from_numpy(pc)[None].cuda(),'text':[text]})
  from regression_adapter import text_bbf_scores
  scores=text_bbf_scores(o,mask).cpu().numpy();boxes=torch.cat([o['last_center'],o['last_pred_size']],-1).cpu().numpy()
  with np.load(old/'source_cache'/(r['id']+'.npz'),allow_pickle=False) as z:
   assert np.array_equal(scores,z['text_scores']),r['id']+' score drift'
   assert np.array_equal(boxes,z['boxes_raw6']),r['id']+' box drift'
   assert z['point_sample_sha256'].tobytes().hex()==arrsha(pc)
  assert len(features)==7 and features[-1].shape==(1,256,288)
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),x=features[-1][0].cpu().numpy(),tokens=o['proj_tokens'][0].cpu().numpy(),query=o['last_proj_queries'][0].cpu().numpy(),token_mask=np.asarray(mask,bool),boxes=boxes[0],scores=scores[0],sample_id=np.array(r['id']))
  traces.append({'alias':f"{'A' if i<32 else 'B'}_{r['platform']}_{i:03d}",'id_PRIVATE':r['id'],'platform':r['platform'],'source_bitexact':True,'seconds':time.perf_counter()-t,'peak_bytes':torch.cuda.max_memory_allocated()})
  if (i+1)%8==0:print('FEATURES',i+1,flush=True)
 hook.remove();assert fingerprint(model)==fp;assert all(not p.requires_grad and p.grad is None for p in model.parameters())
 dump(out/'trace_PRIVATE.json',traces);dump(out/'registered_plan.json',plan)
 seal(out,'R16_source_features_sealed_no_G_GT_update',n=64,source_unchanged=True,boxes_and_scores_bitexact_R15=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(),programs=plan['programs'])
 print('FEATURES_SEALED',flush=True)
if __name__=='__main__':main()
