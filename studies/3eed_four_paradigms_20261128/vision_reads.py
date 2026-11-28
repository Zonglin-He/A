"""Capture unmerged frozen visual patches and decode unmarked RGB crops."""
from collections import defaultdict
from hashlib import sha256
from time import perf_counter
import argparse
import json
import re

from PIL import Image
import numpy as np
import torch
from torch.nn.functional import normalize

from prepare_blind import HERE, QWEN, RUN, digest, read, sealed_inputs, write


def image(path):
    with Image.open(path) as source:
        return source.convert('RGB')


def patches(processor,model,images):
    request=processor.image_processor(images=images,return_tensors='pt').to('cuda')
    vision=model.model.visual
    captured=[]
    hook=vision.blocks[-1].register_forward_hook(lambda _m,_args,out:captured.append(out))
    try:
        vision(request.pixel_values.to(vision.dtype),grid_thw=request.image_grid_thw)
    finally:
        hook.remove()
    assert len(captured)==1
    merge=vision.spatial_merge_size
    grids=request.image_grid_thw.tolist()
    shapes=[(h,w) for _,h,w in grids]
    chunks=captured[0].split([h*w for h,w in shapes])
    assert all(d==1 and h%merge==w%merge==0 for d,h,w in grids)
    return [x.reshape(h//merge,w//merge,merge,merge,-1).permute(0,2,1,3,4).reshape(h,w,-1)
            for x,(h,w) in zip(chunks,shapes,strict=True)]


def prototype_mask(features, size, window, marks):
    h,w,channels=features.shape
    if not marks:
        return None
    wh=np.asarray(size,float)
    window=np.asarray(window,float)/np.tile(wh,2)
    mark=(np.asarray(marks)/wh*np.array([w,h])).astype(int)
    mark=np.clip(mark,0,np.array([w-1,h-1]))
    ix=np.unique(mark[:,1]*w+mark[:,0])
    yy,xx=np.mgrid[:h,:w]
    grid=np.stack(((xx+.5)/w,(yy+.5)/h),-1).reshape(-1,2)
    center=(window[:2]+window[2:])*.5
    half=(window[2:]-window[:2])*.5
    distance=(np.abs(grid-center)/half).max(1)
    bg=np.flatnonzero((distance>1.)&(distance<=1.55))
    if len(bg)<4:
        bg=np.flatnonzero(distance>1.05)
    if len(bg)<4 or not len(ix):
        return None
    f=features.reshape(-1,channels).float()
    f=normalize(f-f.mean(0),dim=-1)
    positive=normalize(f[ix].mean(0),dim=0)
    negative=normalize(f[bg].mean(0),dim=0)
    field=torch.sigmoid(8*(f@(positive-negative)))
    field=field*field.new_tensor(distance<=1.15)
    return dict(mask=field.reshape(h,w).cpu().numpy().round(5).tolist(),grid=[h,w],
                foreground_seeds=len(ix),background_seeds=len(bg),
                contrast=float(field[ix].mean()-field[bg].mean()))


def masks(processor,model,source,cases):
    grouped=defaultdict(list)
    for case in cases:
        if case['first_window'] and case['marks']:
            grouped[source[case['uid']]['frame']['rgb']].append(case)
    keys=list(grouped)
    result={}
    calls=0
    start=perf_counter()
    grids=[]
    for i in range(0,len(keys),8):
        photos=keys[i:i+8]
        features=patches(processor,model,[image(f) for f in photos])
        calls+=1
        for path,feature in zip(photos,features,strict=True):
            for case in grouped[path]:
                uid=case['uid']
                raw_shape=source[uid]['raw']['2d'].get('visual_grid')
                if raw_shape is not None:
                    grids.append(list(feature.shape[:2])==list(raw_shape[1:]))
                data=prototype_mask(feature,case['size'],case['window'],case['marks'])
                if data is not None:
                    result[uid]=data
        if calls%15==0 or i+8>=len(keys):
            print('ViT images',min(i+8,len(keys)),'/',len(keys),flush=True)
    out=RUN/'vit-masks-private.json'
    write(out,result)
    stats=dict(payload=digest(out),vision_forward_batches=calls,unique_images=len(keys),valid_masks=len(result),
               cached_grid_checks=len(grids),exact_cached_grids=sum(grids),language_decodes=0,
               seconds=perf_counter()-start)
    write(RUN/'vit-seal.json',stats)
    return stats


def crop_rect(window,size,scale):
    b=np.asarray(window,float)
    center=(b[:2]+b[2:])/2
    span=(b[2:]-b[:2])*scale/2
    start=np.maximum(0,np.floor(center-span)).astype(int)
    end=np.minimum(size,np.ceil(center+span)).astype(int)
    return tuple(np.r_[start,end].tolist())


def visual_request(processor, images, prompts):
    chats=[[{'role':'user','content':[{'type':'image'},{'type':'text','text':p}]}] for p in prompts]
    text=[processor.apply_chat_template(chat,tokenize=False,add_generation_prompt=True) for chat in chats]
    return processor(text=text,images=images,return_tensors='pt',padding=True).to('cuda')


def number_field(text,key,n):
    match=re.search(r'"?'+key+r'"?\s*:\s*(\[[^\[\]]+\])',text,flags=re.IGNORECASE)
    if match is None:
        return None
    try:
        values=np.asarray(json.loads(match.group(1)),float)
    except (ValueError,TypeError):
        return None
    return values if values.shape==(n,) and np.isfinite(values).all() else None


def parse_crop(text,crop):
    box=number_field(text,'bbox_2d',4)
    dimensions=number_field(text,'size_lwh',3)
    if box is None or (box[2:]-box[:2]).min()<3 or min(box)<-10 or max(box)>1010:
        box=None
    if box is not None:
        widths=np.asarray(crop[2:])-np.asarray(crop[:2])
        box=(np.clip(box,0,1000)*np.tile(widths,2)/1000.+np.tile(crop[:2],2)).tolist()
    if dimensions is not None:
        dimensions=None if (dimensions<.05).any() or (dimensions>30).any() else dimensions.tolist()
    return dict(window=box,prior=dimensions)


SHAPE=('Find the exact object described in this clean cropped image. Return only JSON with '
       '"bbox_2d": [xmin,ymin,xmax,ymax] on a normalized 0 to 1000 crop grid, and '
       '"size_lwh": [true length, width, height] in meters. Keep the bounding rectangle tight. Description: ')
VIEW=('For the described object, which face dominates its view in the image? Reply only with a digit. '
      '1 front end, 2 rear end, 3 left side, 4 right side, 5 oblique, unclear, or not applicable. Description: ')


def local_read(processor,model,photos,expressions,seed):
    req=visual_request(processor,photos,[SHAPE+t for t in expressions])
    torch.manual_seed(seed)
    tokens=model.generate(**req,do_sample=True,temperature=.65,top_p=.9,top_k=0,
                          num_return_sequences=2,max_new_tokens=94)
    text=processor.batch_decode(tokens[:,req.input_ids.shape[1]:],skip_special_tokens=True)
    choice=visual_request(processor,photos,[VIEW+t for t in expressions])
    ids=[processor.tokenizer.encode(str(i),add_special_tokens=False) for i in range(1,6)]
    assert all(len(x)==1 for x in ids)
    logits=model(**choice,use_cache=False,logits_to_keep=1).logits[:,0,[x[0] for x in ids]].float()
    probabilities=logits.softmax(-1).cpu().tolist()
    assert len(text)==2*len(probabilities)
    return list(zip(text[::2],text[1::2],probabilities,strict=True))


def crops(processor,model,source,cases,*,smoke):
    selected=read(RUN/'fovea-roster-private.json')
    valid={r['uid']:r for r in cases if r['first_window']}
    uids=[u for u in selected if u in valid]
    if smoke:
        uids=uids[:4]
    jobs=[(uid,scale) for uid in uids for scale in (1.3,1.8)]
    parts=RUN/('crop-smoke' if smoke else 'crop-parts')
    parts.mkdir(exist_ok=True)
    output=defaultdict(list)
    calls,reused=0,0
    start=perf_counter()
    for i in range(0,len(jobs),8):
        group=jobs[i:i+8]
        output_path=parts/f'{i:05d}.json'
        if output_path.is_file():
            saved=read(output_path)
            assert saved['jobs']==[list(x) for x in group] and saved['code']==digest(__file__)
            results=saved['results']
            reused+=len(results)
        else:
            images=[]; extents=[]; descriptions=[]
            for uid,scale in group:
                case=valid[uid]
                frame=source[uid]['frame']
                window=crop_rect(case['window'],case['size'],scale)
                extents.append(window)
                images.append(image(frame['rgb']).crop(window))
                descriptions.append(frame['text'])
            seed=int(sha256(str(group).encode()).hexdigest()[:8],16)
            generated=local_read(processor,model,images,descriptions,seed)
            results=[dict(scale=scale,crop=list(window),samples=[parse_crop(a,window),parse_crop(b,window)],
                          view=prob,raw=[a,b]) for (_,scale),window,(a,b,prob) in zip(group,extents,generated,strict=True)]
            write(output_path,dict(jobs=group,code=digest(__file__),results=results))
            calls+=1
        for (uid,_),record in zip(group,results,strict=True):
            output[uid].append(record)
        if (i//8+1)%8==0 or i+8>=len(jobs):
            print('Clean crops',min(i+8,len(jobs)),'/',len(jobs),flush=True)
    if smoke:
        print(output,flush=True)
        return None
    out=RUN/'fovea-private.json'
    write(out,output)
    records=[x for vv in output.values() for x in vv]
    result=dict(payload=digest(out),cases=len(selected),eligible=len(output),image_crops=len(jobs),
                crop_prefills_per_job=2,local_completions=2*len(jobs),view_distributions=len(jobs),
                generation_and_view_batches=2*((len(jobs)+7)//8),completion_with_2d=sum(s['window'] is not None for x in records for s in x['samples']),
                completion_with_sizes=sum(s['prior'] is not None for x in records for s in x['samples']),
                reused_crops=reused,seconds=perf_counter()-start)
    write(RUN/'fovea-seal.json',result)
    return result


@torch.inference_mode()
def run(mode):
    train=read(RUN/'training-seal.json')
    assert not train['used_target_data'] and train['code']['sim_train.py']==digest(HERE/'sim_train.py')
    pin=read(RUN/'input-seal.json')
    assert not pin['labels_read'] and pin['evidence']==digest(RUN/'inputs-private.json')
    assert pin['fovea']==digest(RUN/'fovea-roster-private.json')
    source,*_=sealed_inputs()
    from run_full_qwen import load_model
    cases=read(RUN/'inputs-private.json')
    processor,model,_=load_model(QWEN)
    if mode in ('all','vit') and not (RUN/'vit-seal.json').exists():
        masks(processor,model,source,cases)
    if mode in ('all','crop','smoke') and not (RUN/'fovea-seal.json').exists():
        crops(processor,model,source,cases,smoke=mode=='smoke')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('all','vit','crop','smoke'))
    run(parser.parse_args().mode)
