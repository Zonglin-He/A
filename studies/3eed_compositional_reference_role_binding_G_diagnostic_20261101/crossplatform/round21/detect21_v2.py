"""Full compositional query proposals. Reuse only exact phrase-matched old observations."""
import time,re,hashlib
import numpy as np
from common21 import *
from observations import load
from detect import nms,rect_iou
def main():
    import torch
    from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection
    from PIL import Image
    plan=lock('DINO_v2_LOCK.json');verify(BASE/'text208_attempt002');verify(BASE/'dino208_attempt001');out=BASE/'dino208_attempt002';out.mkdir(exist_ok=False);(out/'cases_PRIVATE').mkdir();folder=ROOT/'crossplatform/round20/assets/grounding_dino';asset=read(folder.parent/'DINO_ASSET_RECEIPT.json');assert sha(folder/'model.safetensors')==asset['model_sha256'];torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    processor=AutoProcessor.from_pretrained(folder,local_files_only=True);model,info=AutoModelForZeroShotObjectDetection.from_pretrained(folder,local_files_only=True,use_safetensors=True,output_loading_info=True,disable_custom_kernels=True);assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'));model.eval().requires_grad_(False).cuda();versions=[(n,p._version) for n,p in model.named_parameters()];calls=0;reused=0;start=time.perf_counter()
    def detect(rgb,query):
        nonlocal calls
        text=query.casefold().strip(' .')+'.';ins=processor(images=Image.fromarray(rgb),text=text,return_tensors='pt');assert ins.input_ids.shape[1]<=256;ins=ins.to('cuda')
        with torch.inference_mode():p=model(**ins)
        b=p.pred_boxes[0].cpu().numpy();s=p.logits[0].sigmoid().max(-1).values.cpu().numpy();b=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([rgb.shape[1],rgb.shape[0]]*2);take=np.flatnonzero(s>=.25);take=take[nms(b[take],s[take])[:10]];calls+=1
        return {'boxes_PRIVATE':b[take],'scores':s[take],'query_sha256':hashlib.sha256(text.encode()).hexdigest()}
    for i,r in enumerate(read(ROWS)):
        g=read(BASE/'text208_attempt002/cases_PRIVATE'/(r['id']+'.json'));prep=old(r,'pairs');payload={'alias':r['alias'],'objects':{},'reuse_exact_query_only':True}
        if prep['pairs']:
            frame,scene,binding=load(r);previous=old(r,'dino');oldrefs={x['own_query']:x['entity'] for x in old(r,'text')['references']}
            for ref in g['references']:
                priorg=read(BASE/'text208_attempt001/cases_PRIVATE'/(r['id']+'.json'));priord=read(BASE/'dino208_attempt001/cases_PRIVATE'/(r['id']+'.json'));priorkey=next((x['entity'] for x in priorg['references'] if x['own_query']==ref['own_query']),None);key=oldrefs.get(ref['own_query'])
                if priorkey and priorkey in priord['objects']:d=dict(priord['objects'][priorkey],reused_R21_exact_phrase=True);reused+=1
                elif key and key in previous['objects']:d=previous['objects'][key];reused+=1;d=dict(d,reused_R20_exact_phrase=True)
                else:
                    rgb=frame['rgb'];a=detect(rgb,ref['own_query']);flip_allowed=not bool(re.search(r'\b(left|right|clockwise)\b',ref['own_query'],re.I));b=detect(rgb[:,::-1].copy(),ref['own_query']) if flip_allowed else None
                    if b:
                        bb=np.array(b['boxes_PRIVATE']);W=rgb.shape[1];b['boxes_PRIVATE']=np.c_[W-bb[:,2],bb[:,1],W-bb[:,0],bb[:,3]]
                    stable=[j for j,box in enumerate(a['boxes_PRIVATE']) if b is None or max([rect_iou(box,bb) for bb in b['boxes_PRIVATE']],default=0)>=.5];d={'original':a,'flip':b,'stable_indices':stable,'reused_R20_exact_phrase':False}
                payload['objects'][ref['entity']]=d
        dump(out/'cases_PRIVATE'/(r['id']+'.json'),payload)
        if (i+1)%16==0:print('DINO21',i+1,'new',calls,'reused',reused,flush=True)
    assert versions==[(n,p._version) for n,p in model.named_parameters()];seal(out,'R21_compositional_reference_proposals_no_GT',n=208,new_calls=calls,reused_queries=reused,model_unchanged=True,seconds=time.perf_counter()-start);print('DINO21_SEALED',calls,reused)
if __name__=='__main__':main()
