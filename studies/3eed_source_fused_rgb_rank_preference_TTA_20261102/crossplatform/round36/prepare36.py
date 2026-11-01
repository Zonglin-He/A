"""Only new DEV64 RGB extraction, then frozen fused rank-feature capture."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time,numpy as np
from common36 import *
def main():
 import torch
 from rank_fusion25 import RankFusion
 from rgb25 import RGBEncoder,project
 from PIL import Image
 check_plan();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;start=time.perf_counter()
 out=BASE/'rgbDEV64_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();enc=RGBEncoder();valid=0
 for r in rows('DEV64'):
  z=feature(r);im=Image.open(r['imagepath']).convert('RGB');rect,geom,v,spec=project(r,z,im.size);rgb,g=enc.encode(im,rect,v);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),rgb=rgb,global_token=g,geom=geom,valid=v,rects=rect,image_sha256=np.array(sha(r['imagepath'])),boxes_sha256=np.array(hashlib.sha256(z['boxes'].tobytes()).hexdigest()),missing_distortion_assumed_zero=np.array(spec.distortion is None));valid+=int(v.sum())
 assert enc.unchanged();seal(out,'R36_only_DEV64_frozen_RGB_features_no_GT',n=64,valid_candidate_ROIs=valid,source_RGB_encoder_unchanged=True,seconds=time.perf_counter()-start);del enc;torch.cuda.empty_cache()
 out=BASE/'capture272_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();records=[];rr=rows()+rows('DEV64')
 for mode in MODES:
  m=RankFusion().cuda().eval().requires_grad_(False);m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in load_npz(source_path(mode)).items()});before=fp_nonrank(m);hooks=[];u=[];rankout=[];hooks.append(m.rank.register_forward_pre_hook(lambda mod,x:u.append(x[0].detach().clone())));hooks.append(m.rank.register_forward_hook(lambda mod,x,y:rankout.append(y.detach().clone())))
  for j in range(0,len(rr),32):
   batch=rr[j:j+32];b=tensorbank(batch);u.clear();rankout.clear()
   with torch.inference_mode():s,d=m(b,mode=mode)
   scores=s.cpu().numpy();gate=d['gate'].cpu().numpy();rankinput=u[-1].cpu().numpy();ro=rankout[-1].squeeze(-1).cpu().numpy()
   for k,r in enumerate(batch):
    z=feature(r);old=None;err=None;rawmatch=None
    if r['stage']!='Q':
     old=load_npz(ROOT/'crossplatform/round25/target400_attempt001/predictions_PRIVATE'/(r['id']+'.npz'));name='fusion_rgb_seed0' if mode=='rgb' else 'noRGB_source_train_seed0';ix=old['arm_names'].tolist().index(name);err=float(abs(scores[k]-old['scores'][ix]).max());rawmatch=int(scores[k].argmax())==int(old['query_indices'][ix]);assert err<2e-5 and rawmatch,(r['alias'],mode,err,rawmatch);assert np.array_equal(z['boxes'][int(scores[k].argmax())],old['boxes'][ix])
    q0=int(scores[k].argmax());qg=grouped_readout(r,scores[k],scores[k]);assert qg==q0
    np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'_'+mode+'.npz'),rankinput=rankinput[k],gate=gate[k],rank0_capture=ro[k],native_scores=scores[k],boxes=z['boxes'],source_q0=np.array(q0));records.append({'alias':r['alias'],'platform':r['platform'],'mode':mode,'split':'DEV64' if r['stage']=='Q' else 'A208','old_R25_score_maxerror':err,'old_R25_raw_argmax_same':rawmatch,'grouped_q0_matches_native_raw':qg==q0})
  for h in hooks:h.remove()
  assert fp_nonrank(m)==before;del m
  print('PREPARE36',mode,'seconds',round(time.perf_counter()-start,1),flush=True)
 dump(out/'SOURCE_REPLAY_RECEIPTS.json',records);seal(out,'R36_source_fused_native_scores_and_rank200_before_adaptation',n=272,modes=MODES,A_R25_replay_observations=208*2,DEV_new_RGB=64,no_GT=True,seconds=time.perf_counter()-start,plan_sha256=sha(EXP/'PLAN_LOCK.json'));print('PREPARE36_SEALED')
if __name__=='__main__':main()
