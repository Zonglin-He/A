"""Exact-query cache reuse plus official frozen full-frame reference proposals."""
import time,re,hashlib,numpy as np
from common24 import *
from observations import load
from detect import nms,rect_iou
def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    lock('DINO_LOCK.json');out=BASE/'dinoM64_attempt001';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();folder=ROOT/'crossplatform/round20/assets/grounding_dino';receipt=read(folder.parent/'DINO_ASSET_RECEIPT.json');assert sha(folder/'model.safetensors')==receipt['model_sha256'];torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;processor=AutoProcessor.from_pretrained(folder,local_files_only=True);model,info=AutoModelForZeroShotObjectDetection.from_pretrained(folder,local_files_only=True,use_safetensors=True,output_loading_info=True,disable_custom_kernels=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model.eval().requires_grad_(False).cuda();versions=[(n,p._version) for n,p in model.named_parameters()];calls=0;reuse=0;start=time.perf_counter()
    def infer(rgb,query):
        nonlocal calls
        text=query.casefold().strip(' .')+'.';inputs=processor(images=Image.fromarray(rgb),text=text,return_tensors='pt');assert inputs.input_ids.shape[1]<=256;inputs=inputs.to('cuda')
        with torch.inference_mode():p=model(**inputs)
        b=p.pred_boxes[0].cpu().numpy();s=p.logits[0].sigmoid().max(-1).values.cpu().numpy();b=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([rgb.shape[1],rgb.shape[0]]*2);take=np.flatnonzero(s>=.25);take=take[nms(b[take],s[take])[:10]];calls+=1;return {'boxes_PRIVATE':b[take],'scores':s[take],'query_sha256':hashlib.sha256(text.encode()).hexdigest()}
    for i,r in enumerate(read(EXP.parent/'input/M64_PRIVATE.json')):
        g=read(BASE/'textM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));prep=read(BASE/'preparedM64_attempt001/cases_PRIVATE'/(r['id']+'.json'));payload={'alias':r['alias'],'objects':{}}
        if prep['objects']:
            frame,scene,binding=load(r);priorg=read(ROOT/'crossplatform/round21/text208_attempt003/cases_PRIVATE'/(r['id']+'.json')) if r['stage']!='M' else {'references':[]};priord=read(ROOT/'crossplatform/round21/dino208_attempt003/cases_PRIVATE'/(r['id']+'.json')) if r['stage']!='M' else {'objects':{}}
            for ref in g['references']:
                key=next((x['entity'] for x in priorg['references'] if x['own_query']==ref['own_query']),None)
                if key and key in priord['objects']:d=dict(priord['objects'][key],reused_exact_query=True);reuse+=1
                else:
                    rgb=frame['rgb'];a=infer(rgb,ref['own_query']);allowed=not bool(re.search(r'\b(left|right|clockwise)\b',ref['own_query'],re.I));b=infer(rgb[:,::-1].copy(),ref['own_query']) if allowed else None
                    if b:
                        bb=np.array(b['boxes_PRIVATE']);W=rgb.shape[1];b['boxes_PRIVATE']=np.c_[W-bb[:,2],bb[:,1],W-bb[:,0],bb[:,3]]
                    stable=[j for j,box in enumerate(a['boxes_PRIVATE']) if b is None or max([rect_iou(box,bb) for bb in b['boxes_PRIVATE']],default=0)>=.5];d={'original':a,'flip':b,'stable_indices':stable,'reused_exact_query':False}
                payload['objects'][ref['entity']]=d
        dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload)
        if (i+1)%16==0:print('DINO23',i+1,calls,reuse,flush=True)
    assert versions==[(n,p._version) for n,p in model.named_parameters()];seal(out,'R23_frozen_reference_proposals_no_GT',n=64,calls=calls,reused_queries=reuse,seconds=time.perf_counter()-start,model_unchanged=True);print('DINO23_SEALED',calls,reuse)
if __name__=='__main__':main()
