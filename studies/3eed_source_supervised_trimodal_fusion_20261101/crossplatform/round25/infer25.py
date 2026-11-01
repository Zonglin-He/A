"""Cache-free three-modal inference wrapper and full-model closure probes.
The new source-trained residual fusion acts BEFORE the matching projection;
3D proposal boxes and all original feature encoders stay frozen.
"""
import sys,copy,time,numpy as np,torch
from common25 import *
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,fingerprint,mask_for
from runner_helpers import sample_points
from model25 import make_model,state_path
from rgb25 import RGBEncoder,project
from PIL import Image

class FrozenTrimodalGrounder:
 def __init__(self,weights_path,family):
  self.grounder,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));self.original_fingerprint=fingerprint(self.grounder);self.rgb=RGBEncoder();self.fusion=make_model(family,projection=copy.deepcopy(self.grounder.contrastive_align_projection_image)).cuda().eval();self.fusion.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in load_npz(weights_path).items()});self.fusion.requires_grad_(False)
 def predict(self,r,mode='rgb'):
  pc,_,_=sample_points(r)
  if r['platform']=='waymo':
   from utils.transform_waymo import transform_to_front_view
   pc,_=transform_to_front_view(pc,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pc=np.ascontiguousarray(pc,np.float32)
  text=(' '.join(r['caption'].replace(',', ' ,').split()) if r['platform']=='waymo' else r['caption'])+' . not mentioned';mask,_=mask_for(self.grounder,text,'all_content_tokens');h=[];t=[];hooks=[self.grounder.contrastive_align_projection_image.register_forward_pre_hook(lambda m,a:h.append(a[0].detach().clone())),self.grounder.contrastive_align_projection_text.register_forward_pre_hook(lambda m,a:t.append(a[0].detach().clone()))]
  with torch.inference_mode():o=self.grounder({'point_clouds':torch.tensor(pc,device='cuda')[None],'text':[text]})
  for hook in hooks:hook.remove()
  assert len(h)==7 and len(t)==1;boxes=torch.cat([o['last_center'],o['last_pred_size']],-1)[0].cpu().numpy();image=Image.open(r['imagepath']).convert('RGB');rect,geom,valid,spec=project(r,{'boxes':boxes},image.size);rgb,glob=self.rgb.encode(image,rect,valid);b={'h':h[-1],'tokens':o['proj_tokens'],'text_mean':t[0][:,1:-1].mean(1),'mask':torch.tensor(np.asarray(mask,bool),device='cuda')[None],'padding':torch.zeros((1,len(mask)),device='cuda',dtype=torch.bool),'rgb':torch.tensor(rgb,device='cuda')[None],'global':torch.tensor(glob,device='cuda')[None],'geom':torch.tensor(geom,device='cuda')[None],'valid':torch.tensor(valid,device='cuda')[None]}
  with torch.inference_mode():s,d=self.fusion(b,mode=mode)
  return {'scores':s[0].cpu().numpy(),'boxes':boxes,'gate':d['gate'][0].cpu().numpy(),'query':int(s[0].argmax())}
 def unchanged(self):return fingerprint(self.grounder)==self.original_fingerprint and self.rgb.unchanged()

def closure():
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;c=next(x for x in read(EXP/'FINAL_SOURCE_SELECTED_LOCK.json')['selected'] if x['seed']==0 and x['variant']=='rgb');path=state_path(c);model=FrozenTrimodalGrounder(path,c['family']);rr=rows('SV512')[:8]+[r for r in rows('N64') if r['platform']=='drone'][:8]+[r for r in rows('N64') if r['platform']=='quad'][:8];results=[];start=time.perf_counter()
 from fusion25 import bank,part
 for r in rr:
  pred=model.predict(r);z=feature(r);assert np.array_equal(pred['boxes'],z['boxes']);b=part(bank([r]),slice(None))
  with torch.no_grad():s,d=model.fusion(b)
  err=float(abs(s[0].cpu().numpy()-pred['scores']).max());assert err<2e-5 and int(s[0].argmax())==pred['query'];results.append({'alias':r['alias'],'platform':r['platform'],'score_max_error':err,'boxes_bitexact':True})
 assert model.unchanged();dump(EXP/'FULL_MODEL_CLOSURE.json',{'state':'PASSED','n':24,'actual_frozen_point_text_and_RGB_forwards':24,'trained_fusion_source_state':c['state_name'],'nonfusion_source_and_RGB_weights_unchanged':True,'GT_or_source_feature_cache_needed_by_runtime_predict':False,'closure_comparison_uses_cache_only_as_independent_reference':True,'seconds':time.perf_counter()-start,'records':results});print('CLOSURE25_PASSED',flush=True)
if __name__=='__main__':closure()
