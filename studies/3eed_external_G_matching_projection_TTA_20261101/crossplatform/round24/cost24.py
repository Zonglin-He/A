"""Matched M64 CLIP cost and cosine replay, no labels or method changes."""
import time,sys,numpy as np
from common24 import *
from modality24 import FrozenSemanticEncoder,observe,paths,crops,reference_crop

def main():
 import torch
 from PIL import Image
 torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;model=FrozenSemanticEncoder(device='cuda');start=time.perf_counter();times=[];images=0;texts=0;maxerr=0.
 for r in rows('M64'):
  t=time.perf_counter();p=paths(r);prep=read(p['prep']/'cases_PRIVATE'/(r['id']+'.json'));g=read(p['text']/'cases_PRIVATE'/(r['id']+'.json'));cached=read(BASE/'clip336_attempt001/cases_PRIVATE'/(r['id']+'.json'));ims=[];queries=[];expected=[]
  if prep['mask_pool_PRIVATE']:
   frame,scene,binding=observe(r)
   for m in prep['mask_pool_PRIVATE']:
    for k,image in enumerate(crops(frame['rgb'],scene['objects'][m])):
     x=cached['targets_PRIVATE'][str(m)][k];assert arrsha(np.asarray(image))==x['crop_sha256'];ims += [image,Image.new('RGB',image.size,(127,127,127))];queries += [g['subject_query']]*2;expected += [x['cosine'],x['gray_cosine']]
   for ref in g['references']:
    for x in cached['references'].get(ref['entity'],[]):
     image=reference_crop(frame['rgb'],x['box_PRIVATE']);assert arrsha(np.asarray(image))==x['crop_sha256'];ims += [image,Image.new('RGB',image.size,(127,127,127))];queries += [ref['phrase']]*2;expected += [x['cosine'],x['gray_cosine']]
   unique=list(dict.fromkeys(queries));te,tr=model.encode_texts(unique);im=model.encode_crops(ims);lookup={q:j for j,q in enumerate(unique)};sim=(im*te[[lookup[q] for q in queries]]).sum(-1).numpy();maxerr=max(maxerr,float(abs(sim-expected).max()));assert maxerr<1e-6;images+=len(ims);texts+=len(unique)
  times.append(time.perf_counter()-t)
 out={'n_M64':64,'images_encoded_including_gray':images,'text_queries':texts,'seconds_excluding_model_load':time.perf_counter()-start,'mean_per_observation_seconds':float(np.mean(times)),'p95_seconds':float(np.quantile(times,.95)),'peak_bytes':torch.cuda.max_memory_allocated(),'cosine_replay_max_error':maxerr,'Qwen_same_M64_forward_count':read(BASE/'evidenceM64_attempt001/SEALED.json')['new_calls'],'Qwen_same_M64_seconds_excluding_model_load':read(BASE/'evidenceM64_attempt001/SEALED.json')['seconds'],'interpretation':'matched target/reference images and gray controls; includes RGB/scene read for both, excludes model startup; no evidence rescoring changed predictions'};dump(EXP/'MATCHED_MODALITY_COST.json',out);print(out)
if __name__=='__main__':main()
