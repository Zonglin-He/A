"""Local frozen RGB inference. Only image, utterance and opaque ID enter the model."""
import argparse
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from paths import HERE, OUT, PRIV, SPEC, digest, read, write

PROMPTS = {
    '3d': 'Locate the single object referred to by this description in the image and give its 3D bounding box. Return only JSON in this format: [{"bbox_3d":[x_center,y_center,z_center,x_size,y_size,z_size,roll,pitch,yaw],"label":"object"}]. Description: ',
    '2d': 'Locate the single object referred to by this description in the image. Return only JSON in this format: [{"bbox_2d":[x_min,y_min,x_max,y_max],"label":"object"}]. Use normalized coordinates from 0 to 1000. Description: '
}

def parse(text, task):
    field='bbox_'+task
    clean=re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$','',text.strip(),flags=re.I)
    try:
        arr=json.loads(clean)
        if isinstance(arr,dict):arr=[arr]
        if not isinstance(arr,list) or len(arr)!=1:return None,'not_one_object'
        if not isinstance(arr[0],dict) or field not in arr[0]:return None,'missing_field'
        x=np.asarray(arr[0][field],dtype=float)
    except (ValueError,TypeError,OverflowError):
        return None,'parse_error'
    if x.shape!=((9,) if task=='3d' else (4,)) or not np.isfinite(x).all():return None,'invalid_shape_or_number'
    if task=='3d' and (x[2]<=0 or min(x[3:6])<=0 or max(abs(x[6:]))>1):return None,'invalid_3d'
    if task=='2d' and ((x[:2]>=x[2:]).any() or min(x)<0 or max(x)>1000):return None,'invalid_2d'
    return x.tolist(),'ok'

@torch.inference_mode()
def main(model_dir):
    cfg=read(SPEC)
    roster=read(PRIV/'panel.json')
    obs=read(OUT/'panel_seal.json')
    assert digest(SPEC)==obs['protocol'] and digest(PRIV/'panel.json')==obs['panel']
    torch.manual_seed(cfg['inference']['seed'])
    assets=read(HERE/'MODEL.json')
    assert (assets['model'],assets['revision'])==(cfg['model'],cfg['revision'])
    for name,expected in assets['files'].items():
        path=model_dir/name
        assert path.stat().st_size==expected['size'],name
        if 'sha256' in expected:
            assert digest(path)==expected['sha256'],name
        else:
            data=path.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\x00'+data).hexdigest()==expected['git_blob_sha1'],name
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    processor.image_processor.size={'shortest_edge':65536,'longest_edge':cfg['inference']['max_pixels']}
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_dir,dtype=torch.bfloat16,device_map={'':'cuda:0'},
                        attn_implementation='sdpa',local_files_only=True,use_safetensors=True).eval().requires_grad_(False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    stamp=dict(assets=digest(HERE/'MODEL.json'),protocol=obs['protocol'],panel=obs['panel'],code=digest(__file__),model=cfg['revision'])
    folder=PRIV/'replies';folder.mkdir(exist_ok=True)
    begin=time.perf_counter()
    fresh=0
    jobs=[(r,t) for r in roster for t in ('3d','2d')]
    for start in range(0,len(jobs),cfg['inference']['batch_size']):
        batch=jobs[start:start+cfg['inference']['batch_size']]
        existing=[(folder/(r['uid']+'_'+t+'.json')).exists() for r,t in batch]
        for (r,t),exists in zip(batch,existing):
            if exists:assert read(folder/(r['uid']+'_'+t+'.json'))['pins']==stamp
        batch=[item for item,done in zip(batch,existing) if not done]
        if not batch:continue
        fresh+=len(batch)
        images=[];prompts=[]
        for r,t in batch:
            assert digest(r['rgb'])==r['rgb_sha']
            with Image.open(r['rgb']) as img:images.append(img.convert('RGB'))
            msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPTS[t]+r['text']}]}]
            prompts.append(processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True))
        inp=processor(text=prompts,images=images,padding=True,return_tensors='pt').to('cuda')
        out=model.generate(**inp,do_sample=False,max_new_tokens=cfg['inference']['max_new_tokens'])
        suffix=out[:,inp.input_ids.size(1):]
        replies=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        for j,((r,t),reply) in enumerate(zip(batch,replies)):
            val,status=parse(reply,t)
            write(folder/(r['uid']+'_'+t+'.json'),dict(uid=r['uid'],task=t,raw=reply,value=val,status=status,pins=stamp,
                    visual_grid=inp.image_grid_thw[j].tolist(),generated_tokens=int((suffix[j]!=processor.tokenizer.pad_token_id).sum())))
        if start%24==0:print('queries',start+len(batch),'/',len(jobs),flush=True)
    files={r['uid']+'_'+t:digest(folder/(r['uid']+'_'+t+'.json')) for r,t in jobs}
    write(OUT/'model_seal.json',dict(pins=stamp,files=len(files),ordered_reply_hash=__import__('hashlib').sha256(
             json.dumps(files,sort_keys=True).encode()).hexdigest(),elapsed_seconds=time.perf_counter()-begin,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
             gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
             target_labels_read=False, model_dtype='bfloat16',fresh_queries=fresh,previously_sealed_queries=len(files)-fresh))
    write(PRIV/'reply_hashes.json',files)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model_dir',type=Path)
    main(p.parse_args().model_dir)
