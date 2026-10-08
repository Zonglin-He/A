"""Paper-only K observation control. Uniform4 delegates unchanged to the registered path.
Legacy positions4/single4 serialization field names do not imply K=4 for the budget controls.
"""
import hashlib,time
import numpy as np
import torch
from methods.decota_final_simplified_v1.observations import (observations,uniform_positions,ContextView,probe_from_detection,tensor_hash)

def budget_observations(expert,parses,frames,ids,interval,*,budget=4,audit=False):
    if budget==4:return observations(expert,parses,frames,ids,interval,audit=audit)
    assert budget in [1,2,8]
    context,s1=parses['context'],parses['old']
    eligible=context['eligible'] and len(expert.processor.tokenizer(context['context'])['input_ids'])<=256
    positions=uniform_positions(ids,interval,budget);obs={};anchors=[]
    for pos in positions:
        if not eligible and not s1['phrase']:continue
        rgb=frames[pos];inp={}
        def hook(m,args,kw):
            for k in ('pixel_values','pixel_mask','input_ids','attention_mask'):
                if k in kw:inp[k]=dict(sha256=tensor_hash(kw[k]),shape=list(kw[k].shape),dtype=str(kw[k].dtype))
        h=expert.model.register_forward_pre_hook(hook,with_kwargs=True) if audit else None
        torch.cuda.synchronize();tick=time.perf_counter()
        try:
            with torch.no_grad():
                d=ContextView(expert,context)(rgb,context['context'],context['entity']) if eligible else expert(rgb,s1['phrase'],s1['entity'])
        finally:
            if h is not None:h.remove()
        torch.cuda.synchronize()
        if eligible:z=probe_from_detection(d,pos,ids[pos])
        else:z=dict(position=pos,frame_id=ids[pos],accepted=d['accepted'],reason=d['reason'],margin=d['margin'],
            boxes=torch.tensor([d['box']]) if d['box'] else torch.empty(0,4),target_scores=[d['score']] if d['box'] else [],candidate_ids=[0] if d['box'] else [])
        receipt=dict(position=pos,frame_id=ids[pos],view='original',scale=1.,rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest() if audit else None,
            inputs=inp,seconds=time.perf_counter()-tick,forward_seconds=d['seconds'],text=d['text'],
            target_span=context['span'] if eligible else None,context_active=eligible)
        obs[('original',pos)]=dict(probe=z,receipt=receipt)
        if z['accepted']:
            j=int(np.argmax(z['target_scores']))
            anchors.append(dict(position=pos,frame_id=ids[pos],box=torch.as_tensor(z['boxes'][j]).tolist(),score=float(z['target_scores'][j]),weight=1.))
    return dict(observations=obs,positions4=positions,anchors={'single4':anchors},new_DINO=len(obs),actual_observation_positions=sorted(p for _,p in obs))
