"""Frozen dense RGB encoder + calibrated candidate ROI tokens, no task GT.
Unlike crop/image-text voting, region features enter a learned matching module.
"""
import sys,time,argparse,numpy as np
from common25 import *
sys.path[:0]=[str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round13')]
from camera_projection import CameraSpec,CAMERA_KEYS,read_camera_allowlist,project_model_points
from association import box_corners
from PIL import Image

def project(r,z,image_size):
 fields=read_camera_allowlist(r['metapath']);spec=CameraSpec.from_camera_fields(r['platform'],fields,waymo_view_suffix=int(r['suffix']) if r['platform']=='waymo' else 0,missing_distortion_policy='assume_zero_declared');boxes=z['boxes'];rect=np.zeros((256,4),np.float32);geom=np.zeros((256,8),np.float32);valid=np.zeros(256,bool);W,H=image_size
 good=np.flatnonzero((boxes[:,3:]>0).all(1))
 if len(good):
  corners=np.concatenate([box_corners(boxes[q]) for q in good]);p=project_model_points(corners,spec,image_size);uv=p['uv'].reshape(-1,8,2);dep=p['depth'].reshape(-1,8)
  for j,q in enumerate(good):
   if not (dep[j]>.1).all() or not np.isfinite(uv[j]).all():continue
   raw=np.r_[uv[j].min(0),uv[j].max(0)];clipped=np.clip(raw,[0,0,0,0],[W,H,W,H]);wh=clipped[2:]-clipped[:2];rawarea=np.prod(np.maximum(raw[2:]-raw[:2],0));area=np.prod(wh)
   if (wh<2).any() or area<9 or rawarea<=0:continue
   valid[q]=True;rect[q]=clipped;geom[q]=[(clipped[0]+clipped[2])/2/W,(clipped[1]+clipped[3])/2/H,wh[0]/W,wh[1]/H,area/rawarea,np.log1p(dep[j].mean())/np.log(201),float(not np.array_equal(raw,clipped)),1.]
 return rect,geom,valid,spec

class RGBEncoder:
 def __init__(self):
  import torch
  from torchvision.models import resnet34
  self.torch=torch;path=BASE/'assets/resnet34-b627a593.pth';receipt=read(path.parent/'RESNET_ASSET_RECEIPT.json');assert sha(path)==receipt['sha256'];self.model=resnet34(weights=None);self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);self.model=self.model.cuda().eval().requires_grad_(False);self.versions=[(n,p._version) for n,p in self.model.named_parameters()]
 @__import__('torch').inference_mode()
 def encode(self,image,rects,valid):
  import torch
  from torchvision.ops import roi_align
  W,H=image.size;scale=768/max(W,H);nw=max(16,int(round(W*scale)));nh=max(16,int(round(H*scale)));im=np.array(image.resize((nw,nh),Image.Resampling.BILINEAR));x=torch.tensor(im,device='cuda',dtype=torch.float32).permute(2,0,1)[None]/255;x=(x-torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None])/torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];m=self.model;x=m.maxpool(m.relu(m.bn1(m.conv1(x))));x=m.layer1(x);l2=m.layer2(x);l3=m.layer3(l2);out=torch.zeros((256,4,384),device='cuda');ix=np.flatnonzero(valid)
  if len(ix):
   b=torch.tensor(rects[ix]*np.array([nw/W,nh/H,nw/W,nh/H]),device='cuda',dtype=torch.float32);f2=roi_align(l2,[b],output_size=2,spatial_scale=1/8,sampling_ratio=2,aligned=True);f3=roi_align(l3,[b],output_size=2,spatial_scale=1/16,sampling_ratio=2,aligned=True);out[ix]=torch.cat([f2,f3],1).flatten(2).transpose(1,2)
  global_token=torch.cat([l2.mean((2,3)),l3.mean((2,3))],-1)[0];assert torch.isfinite(out).all();return out.half().cpu().numpy(),global_token.half().cpu().numpy()
 def unchanged(self):return self.versions==[(n,p._version) for n,p in self.model.named_parameters()] and not self.model.training and all(not p.requires_grad for p in self.model.parameters())

def main(pilot=False):
 import torch
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 out=BASE/('pilot32_RGB_attempt001' if pilot else 'rgb_features_attempt001');out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();encoder=RGBEncoder();rr=rows('S3682')[:32] if pilot else rows();start=time.perf_counter();reused=0;validn=0;cameraassumptions=0
 for i,r in enumerate(rr):
  pcache=BASE/'pilot32_RGB_attempt001/samples_PRIVATE'/(r['id']+'.npz')
  if not pilot and pcache.exists():payload=load_npz(pcache);reused+=1
  else:
   z=load_npz(BASE/'pilot32_source_attempt001/samples_PRIVATE'/(r['id']+'.npz')) if pilot else feature(r);image=Image.open(r['imagepath']).convert('RGB');rect,geom,valid,spec=project(r,z,image.size);rgb,global_token=encoder.encode(image,rect,valid);payload=dict(rgb=rgb,global_token=global_token,geom=geom,valid=valid,rects=rect,image_sha256=np.array(sha(r['imagepath'])),missing_distortion_assumed_zero=np.array(spec.distortion is None))
  validn+=int(payload['valid'].sum());cameraassumptions+=bool(payload['missing_distortion_assumed_zero']);np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),**payload)
  if (i+1)%128==0 or pilot and i==31:print('RGB25',i+1,'seconds',round(time.perf_counter()-start,1),'valid',validn,flush=True)
 assert encoder.unchanged();seal(out,'R25_frozen_ImageNet_RGB_projected_ROI_features_no_GT',n=len(rr),reused_pilot_features=reused,valid_candidate_ROIs=validn,missing_distortion_zero_assumption_observations=cameraassumptions,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('RGB25_SEALED',flush=True)
if __name__=='__main__':main('--pilot' in sys.argv)
