"""Independent full-image open-vocabulary subject/reference observations."""
import sys,time,re,hashlib
import numpy as np
from common import *
from observations import load
def rect_iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=np.maximum(hi-lo,0).prod();return float(inter/max((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter,1e-12))
def nms(boxes,scores,threshold=.5):
 out=[]
 for i in np.argsort(-np.asarray(scores),kind='stable'):
  if all(rect_iou(boxes[i],boxes[j])<threshold for j in out):out.append(int(i))
 return out
def main():
 import torch
 from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
 from PIL import Image
 plan=locked('DINO_LOCK.json');asset=read(ROOT/'crossplatform/round20/assets/DINO_ASSET_RECEIPT.json');folder=ROOT/'crossplatform/round20/assets/grounding_dino';assert sha(folder/'model.safetensors')==asset['model_sha256'];out=ROOT/'crossplatform/round20/dino144_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 processor=AutoProcessor.from_pretrained(folder,local_files_only=True);model,info=AutoModelForZeroShotObjectDetection.from_pretrained(folder,local_files_only=True,use_safetensors=True,output_loading_info=True,disable_custom_kernels=True);assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'));model.eval().requires_grad_(False).cuda();version=[(n,p._version) for n,p in model.named_parameters()];calls=0;start=time.perf_counter()
 def detect(rgb,query):
  nonlocal calls
  text=query.casefold().strip(' .')+'.';ins=processor(images=Image.fromarray(rgb),text=text,return_tensors='pt');assert ins.input_ids.shape[1]<=256;ins=ins.to('cuda')
  with torch.inference_mode():p=model(**ins)
  b=p.pred_boxes[0].cpu().numpy();s=p.logits[0].sigmoid().max(-1).values.cpu().numpy();b=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([rgb.shape[1],rgb.shape[0]]*2);take=np.flatnonzero(s>=plan['detector_floor']);keep=nms(b[take],s[take]);take=take[keep[:10]];calls+=1;return {'boxes_PRIVATE':b[take],'scores':s[take],'query_sha256':hashlib.sha256(text.encode()).hexdigest(),'token_count':int(ins.input_ids.shape[1]),'raw_queries':len(s)}
 for i,r in enumerate(read(EXP.parent/'input/ALL144_PRIVATE.json')):
  prep=read(ROOT/'crossplatform/round20/pairs144_attempt001/cases_PRIVATE'/(r['id']+'.json'));g=read(ROOT/'crossplatform/round20/text144_attempt001/cases_PRIVATE'/(r['id']+'.json'));result={'alias':r['alias'],'id_PRIVATE':r['id'],'stage':r['stage'],'platform':r['platform'],'objects':{},'geometry_BINDING':prep['bindings']}
  if prep['pairs'] and g['status']=='parsed':
   frame,scene,binding=load(r);assert binding==prep['bindings'];rgb=frame['rgb'];queries={'subject':g['subject_query'],'category':g['category']};queries.update({ref['entity']:ref['own_query'] for ref in g['references']})
   for key,q in queries.items():
    a=detect(rgb,q);flip_allowed=not bool(re.search(r'\b(left|right|clockwise)\b',q,re.I));b=detect(rgb[:,::-1].copy(),q) if flip_allowed else None
    if b:
     bb=np.array(b['boxes_PRIVATE']);W=rgb.shape[1];b['boxes_PRIVATE']=np.c_[W-bb[:,2],bb[:,1],W-bb[:,0],bb[:,3]]
    stable=[]
    for j,box in enumerate(a['boxes_PRIVATE']):
     overlaps=[rect_iou(box,bb) for bb in b['boxes_PRIVATE']] if b is not None else []
     okay=max(overlaps,default=0)>=.5 if b is not None else True
     if okay:stable.append(j)
    result['objects'][key]={'original':a,'flip':b,'stable_indices':stable,'flip_applicable':flip_allowed,'not_identity_truth':True}
  dump(out/'cases_PRIVATE'/(r['id']+'.json'),result)
  if (i+1)%8==0:dump(out/'PROGRESS.json',{'completed':i+1,'n':144,'calls':calls});print('DINO',i+1,calls,flush=True)
 assert [(n,p._version) for n,p in model.named_parameters()]==version;seal(out,'R20_DINO_subject_reference_boxes_no_GT',n=144,calls=calls,model_unchanged=True,seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated());print('DINO_SEALED',calls)
if __name__=='__main__':main()
