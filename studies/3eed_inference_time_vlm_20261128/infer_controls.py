"""Frozen Qwen decoding, visual selection and context interventions."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import sys
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from prepare import HERE,RUN,KMAX,PLATFORMS,check,digest,load_blind,publish,read
from shared import MODEL, normalization

PREFIX='[{"bbox_3d":['
PREFIX_MODES=('prefix_format','prefix_native','prefix_metric','prefix_wrong','tree_uniform','tree_physics')
SELECT_MODES=('som','som_permuted','graph','graph_shuffled')
ICL_MODES=('icl_zero','icl_random','icl_verified','icl_cross')
ORDER=(*SELECT_MODES,*PREFIX_MODES,'graph_full',*ICL_MODES)


def start_number(center):
    return ' '+', '.join(f'{x:.2f}' for x in center)+','


class NumberTrie(LogitsProcessor):
    """Locally tilt the next-token distribution by subtree prior mass."""
    def __init__(self, tokenizer, configs, weighted, prompt_width):
        self.width=prompt_width
        self.nodes=[]
        self.done=set()
        for c in configs:
            branches=defaultdict(dict)
            for value,logp in zip(c['grid'], c['log_prior'], strict=True):
                tokens=tokenizer.encode(start_number(value),add_special_tokens=False)
                assert tokenizer.decode(tokens,clean_up_tokenization_spaces=False)==start_number(value)
                for j,t in enumerate(tokens):
                    node=branches[tuple(tokens[:j])]
                    node[t]=node.get(t,0.)+np.exp(logp if weighted else 0.)
            self.nodes.append(branches)

    def __call__(self, input_ids,scores):
        if len(self.done)==len(self.nodes):
            return scores
        for b,table in enumerate(self.nodes):
            if b in self.done:
                continue
            prefix=tuple(input_ids[b,self.width:].tolist())
            children=table.get(prefix)
            if children is None:
                self.done.add(b)
                continue
            ids=list(children)
            log=np.log(list(children.values()))
            adjusted=scores[b,ids]+scores.new_tensor(log-np.log(sum(children.values())))
            scores[b].fill_(-torch.inf)
            scores[b,ids]=adjusted
        return scores


def box_answer(box):
    return json.dumps([dict(bbox_3d=[round(float(v),4) for v in box],label='object')],separators=(',',':'))


def frame_image(path):
    with Image.open(path) as f:
        return f.convert('RGB')


def message(image,text,role='user'):
    if image is None:
        return dict(role=role,content=[dict(type='text',text=text)])
    return dict(role=role,content=[dict(type='image'),dict(type='text',text=text)])


def marked(image, proposals, ids):
    """Draw compact numeric badges; only the selected points' scene is shown."""
    image=image.copy()
    draw=ImageDraw.Draw(image)
    size=max(20,round(min(image.size)*.026))
    try:
        font=ImageFont.truetype('DejaVuSans-Bold.ttf',size)
    except OSError:
        font=ImageFont.load_default(size=size)
    for p,i in zip(proposals,ids,strict=True):
        x,y=p['mark']
        r=size*.69
        x=float(np.clip(x,r,image.width-r))
        y=float(np.clip(y,r,image.height-r))
        draw.ellipse((x-r,y-r,x+r,y+r),fill=(29,33,37),outline=(247,211,57),width=max(2,size//8))
        draw.text((x,y),str(i),font=font,anchor='mm',fill=(255,252,224))
    return image


def describe_graph(proposals, corrupt=False):
    if not proposals:
        return ''
    array=proposals[1:]+proposals[:1] if corrupt and len(proposals)>1 else proposals
    points=np.array([p['center_camera'] for p in array])
    lines=['LiDAR estimates in metres. Camera coordinates: x right, y down, z forward (larger z is farther). '
           'Spans are the visible region, and sensor evidence may be incomplete.']
    for i,p in enumerate(array,1):
        cam=p['center_camera']
        span=p['measured'][3:6]
        height='unknown' if p['height'] is None else f"{p['height']:.1f}"
        lines.append(f"{i}: xyz ({cam[0]:.1f},{cam[1]:.1f},{cam[2]:.1f}); observed span "
                     f"({span[0]:.1f},{span[1]:.1f},{span[2]:.1f}); above ground {height}.")
    if len(array)>1:
        for i,p in enumerate(points):
            d=np.linalg.norm(points-p,axis=1)
            near=[j for j in np.argsort(d) if j!=i][:2]
            for j in near:
                x,z=p[[0,2]]-points[j,[0,2]]
                lines.append(f"{i+1} vs {j+1}: {abs(x):.1f} m {'right' if x>=0 else 'left'}, "
                             f"{abs(z):.1f} m {'farther' if z>=0 else 'nearer'}.")
    return '\n'.join(lines)


def selection_job(mode,unit,frame):
    short=unit['proposals'][:KMAX]
    if not short:
        return None
    labels=unit['permutation'] if mode=='som_permuted' else list(range(1,len(short)+1))
    img=marked(frame_image(frame['rgb']),short,labels)
    prompt=("Select the mark for the single object described below. Each number marks a LiDAR region. "
            "Return one number only, with 0 if no region matches. The available marks are "
            + ', '.join(map(str,sorted(labels)))+".")
    if mode in ('graph','graph_shuffled'):
        prompt+='\n'+describe_graph(short,mode=='graph_shuffled')
    prompt+='\nDescription: '+frame['text']
    return dict(images=[img],messages=[message(img,prompt)],allowed=[0,*labels],
                lookup={int(v):i+1 for i,v in enumerate(labels)})


def generation_job(mode,unit,frame,bank,previous):
    from run_full_qwen import PROMPTS
    prompt=PROMPTS['3d']+frame['text']
    img=frame_image(frame['rgb'])
    content=dict(images=[img],messages=[message(img,prompt)],prefix='')
    if mode=='prefix_format':
        content['prefix']=PREFIX
    elif mode.startswith('prefix_') or mode.startswith('tree_'):
        f=unit['center_forms']
        if f is None:
            return None
        if mode.startswith('tree_'):
            content.update(prefix=PREFIX,center_tree=f)
        else:
            content['prefix']=PREFIX+start_number(f[mode.split('_')[1]])
    elif mode.startswith('icl_') and mode!='icl_zero':
        examples=bank[unit['platform']][mode.split('_')[1]]
        messages=[]
        pictures=[]
        for x in examples:
            image=frame_image(x['rgb'])
            pictures.append(image)
            messages.append(message(image,PROMPTS['3d']+x['text']))
            messages.append(message(None,box_answer(x['raw']),'assistant'))
        content['images']=pictures+content['images']
        content['messages']=messages+content['messages']
    elif mode=='graph_full':
        choice=previous['graph'][unit['uid']]['selected']
        if not choice:
            return None
        short=unit['proposals'][:KMAX]
        proposal=short[choice-1]
        image=marked(img,short,list(range(1,len(short)+1)))
        prompt+=f'\nThe chosen physical region is mark {choice}. Its calibrated center is fixed by the answer prefix. '
        prompt+='Complete its dimensions and orientation.\n'+describe_graph(short)
        center=proposal['box_camera'][:3]
        if center[2]<=.1:
            return None
        content=dict(images=[image],messages=[message(image,prompt)],prefix=PREFIX+start_number(center))
    return content


def tensor(processor,page):
    text=[processor.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True)+r.get('prefix','')
          for r in page]
    images=[i for r in page for i in r['images']]
    return processor(text=text,images=images,padding=True,return_tensors='pt').to('cuda')


def decode_batch(mode,processor,model,maximum,page):
    from run_full_qwen import parse

    pack=tensor(processor,page)
    width=pack.input_ids.shape[1]
    if mode in SELECT_MODES:
        scores=model(**pack,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        answer=[]
        for i,r in enumerate(page):
            labels=r['allowed']
            ids=[processor.tokenizer.encode(str(k),add_special_tokens=False) for k in labels]
            assert all(len(x)==1 for x in ids)
            logits=[float(v) for v in scores[i,[x[0] for x in ids]].cpu()]
            picked=labels[int(np.argmax(logits))]
            answer.append(dict(selected=r['lookup'].get(picked,0),digit=picked,labels=labels,logits=logits))
        return answer,0
    kwargs=dict(max_new_tokens=maximum,do_sample=False)
    if mode.startswith('tree_'):
        trie=NumberTrie(processor.tokenizer,[x['center_tree'] for x in page],mode=='tree_physics',width)
        kwargs['logits_processor']=LogitsProcessorList([trie])
    output=model.generate(**pack,**kwargs)[:,width:]
    answers=processor.batch_decode(output,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    generated=int((output!=processor.tokenizer.pad_token_id).sum())
    values=[]
    for x,r in zip(answers,page,strict=True):
        text=r['prefix']+x
        box,status=parse(text,'3d')
        values.append(dict(raw=text,value=box,status=status))
    return values,generated


@torch.inference_mode()
def main():
    cv_plan=check(RUN/'plan-private.json','input',RUN/'plan-seal.json')
    assert not cv_plan['labels_read'] and cv_plan['protocol']==digest(HERE/'PROTOCOL_ZH.md')
    assert cv_plan['code']==digest(HERE/'prepare.py')
    plan=read(RUN/'plan-private.json')
    source,_,_,_,stamp=load_blind()
    assert plan['source']==stamp
    root=RUN/'gpu-state.json'
    provenance=dict(plan=cv_plan['input'],code=digest(Path(__file__)),weight=MODEL.name)
    if root.exists():
        state=read(root)
        assert state['pin']==provenance
    else:
        state=dict(pin=provenance,answers={},bench={})
    processor,model,limit=normalization()
    print('Loaded frozen checkpoint for',len(plan['units']),'queries',flush=True)
    for mode in ORDER:
        mode_result=state['answers'].setdefault(mode,{})
        measure=state['bench'].setdefault(mode,dict(model_inputs=0,batches=0,tokens=0,elapsed_sec=0.,images=0))
        slots=[u for u in plan['units'] if u['uid'] not in mode_result]
        plan_times=4 if mode.startswith('icl_') and mode!='icl_zero' else 6
        for offset in range(0,len(slots),plan_times):
            jobs=[]
            for u in slots[offset:offset+plan_times]:
                id=u['uid']
                frame=source[id]['frame']
                unit=selection_job(mode,u,frame) if mode in SELECT_MODES else generation_job(mode,u,frame,plan['donors'],state['answers'])
                if unit is None:
                    mode_result[id]=dict(status='inapplicable',selected=0) if mode in SELECT_MODES else dict(status='inapplicable',value=None)
                else:
                    jobs.append((id,unit))
            if jobs:
                batch=[r[1] for r in jobs]
                begin=perf_counter()
                try:
                    output,tokens=decode_batch(mode,processor,model,limit,batch)
                    measure['batches']+=1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    shifted=[decode_batch(mode,processor,model,limit,[x]) for x in batch]
                    output=[q[0][0] for q in shifted]
                    tokens=sum(q[1] for q in shifted)
                    measure['batches']+=len(batch)
                torch.cuda.synchronize()
                for (key,_),value in zip(jobs,output,strict=True):
                    mode_result[key]=value
                measure['elapsed_sec']+=perf_counter()-begin
                measure['model_inputs']+=len(batch)
                measure['tokens']+=tokens
                measure['images']+=sum(len(x['images']) for x in batch)
            publish(root,state)
            if offset%30==0 or offset+plan_times>=len(slots):
                print(mode,len(mode_result),'/',len(plan['units']),round(measure['elapsed_sec'],1),flush=True)
        assert len(mode_result)==len(plan['units'])
    size={mode:dict(Counter(x.get('status','digit') for x in y.values())) for mode,y in state['answers'].items()}
    publish(RUN/'gpu-seal.json',dict(artifact=digest(root),pin=provenance,model_input=sum(v['model_inputs'] for v in state['bench'].values()),
        cuda=torch.cuda.get_device_name(),peak_allocated=int(torch.cuda.max_memory_allocated()),
        checkpoint_training=False,target_labels_read=False,outcomes=size))
    print('Sealed all',len(ORDER),'interventions',flush=True)


if __name__=='__main__':
    main()
