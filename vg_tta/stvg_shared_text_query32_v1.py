"""Frozen shared-memory target-noun offset, fixed native routing, two steps."""
import re
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached
from methods.decota_final_simplified_v1.objectives import generalized_iou
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
ARMS=['E_MBR','Temporal_only','Spatial_only','Joint']

def token_spec(tokenizer,row):
    subject=row['parses']['subject'];caption=row['input']['caption'];text=subject+' '+caption
    tok=tokenizer(text,return_offsets_mapping=True);spans=[]
    if subject.strip():spans.append((0,len(subject)))
    # Only a uniquely occurring referent noun is added in the original query.
    # The original prepended subject remains available for wh/ambiguous cases.
    entity=row['parses']['old']['entity'].strip()
    hits=list(re.finditer(r'(?<!\w)'+re.escape(entity)+r'(?!\w)',caption,re.I))if entity else []
    if len(hits)==1:spans.append((len(subject)+1+hits[0].start(),len(subject)+1+hits[0].end()))
    indices=[i for i,(a,b)in enumerate(tok['offset_mapping'])if b>a and any(a<e and b>s for s,e in spans)]
    assert all(tok['offset_mapping'][i][1]>tok['offset_mapping'][i][0]for i in indices)
    return dict(indices=indices,input_ids=tok['input_ids'],token_count=len(tok['input_ids']),
                selected_count=len(indices),original_unique_noun=len(hits)==1,prefix_subject=bool(subject.strip()),
                scope='prepended subject plus unique original referent noun; no action/relation token insertion')

class SharedTextReplay:
    def __init__(self,base,indices):
        self.base=base;self.indices=list(indices);self.delta=torch.zeros(256,device=base.delta.device,requires_grad=True)
        base.restore(base.initial)
        for _,p in base.named:p.requires_grad_(False)
        base.decoder.requires_grad_(False);base.head.requires_grad_(False);base.norm.requires_grad_(False)
        self.calls=0;self.masks=[]
        for view in base.views:
            n=view['info']['fea_map_size'][0]*view['info']['fea_map_size'][1]
            nt=view['prefix'].shape[0]-2*n
            assert all(0<=i<nt for i in indices)
            mask=torch.zeros(view['prefix'].shape[0],1,1,device=self.delta.device);mask[[n+i for i in indices]]=1
            self.masks.append(mask)
    def values(self):
        calls=[0]
        def insert(module,args,kw):
            k=calls[0];calls[0]+=1
            if k%2==0:return args,kw  # Native first-pass action routing is frozen.
            kw=dict(kw);info=dict(kw['encoded_info']);H=info['encoded_feature']
            info['encoded_feature']=H+self.masks[k//2]*self.delta[None,None,:]
            # Original frame/video global anchors and dynamic initial queries
            # remain fixed. Only the shared cross-attention text memory changes.
            kw['encoded_info']=info;return args,kw
        hook=self.base.decoder.register_forward_pre_hook(insert,with_kwargs=True)
        try:v=self.base.reference_values(capture=False)
        finally:hook.remove()
        assert calls[0]==4;self.calls+=1;return v

def objective(boxes,positions,teacher,delta,regularization):
    if not positions:return delta.square().mean()*0
    b=boxes[positions];target=teacher.detach().to(b)
    return (5*(b-target).abs().sum(-1)+2*(1-generalized_iou(b,target))).mean()+regularization*delta.square().mean()

@torch.enable_grad()
def fit(base,source,E,indices,positions,teacher,lr,regularization,check):
    rp=SharedTextReplay(base,indices);opt=torch.optim.Adam([rp.delta],lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
    path=[];updates=[];teacher=teacher.detach().to(rp.delta);src=source.to(rp.delta);et=E.to(device=rp.delta.device,dtype=torch.float64)
    for step in range(3):
        check();v=rp.values();b=corrected(v['boxes'],src,et)
        if step==0:
            assert torch.equal(b.detach().cpu(),E.double())and torch.equal(v['boxes'].detach().cpu(),source)
            assert all(torch.equal(x.detach(),y)for x,y in zip(v['logits'],base.zero['logits']))
        entry=dict(step=step,delta=rp.delta.detach().cpu().clone(),boxes=b.detach().cpu(),native_boxes=v['boxes'].detach().cpu(),logits=detached(v['logits'],'cpu'))
        loss=objective(b,positions,teacher,rp.delta,regularization);entry['spatial_objective']=float(loss.detach());path.append(entry)
        if step==2:break
        opt.zero_grad(set_to_none=True);old=rp.delta.detach().clone();gradient=None
        if positions:
            gradient=torch.autograd.grad(loss,rp.delta)[0];assert torch.isfinite(gradient).all();rp.delta.grad=gradient;opt.step()
        assert torch.isfinite(rp.delta).all()
        updates.append(dict(step=step,updated=bool(positions),gradient=None if gradient is None else gradient.cpu(),
            delta_step=(rp.delta.detach()-old).cpu(),Adam=detached(opt.state_dict(),'cpu')))
    assert all(p.grad is None and not p.requires_grad for p in base.decoder.parameters())
    if not positions:assert all(torch.count_nonzero(v['delta'])==0 for v in path)
    return dict(path=path,updates=updates,teacher=teacher.cpu(),positions=positions,selected_step=2,
        changed_variables=256,pretrained_weights_frozen=True,native_routing_frozen=True,
        only_second_pass_shared_memory=True,regularization=regularization,lr=lr,GT_read=False)

def properties():
    d=torch.zeros(256,requires_grad=True);h=torch.randn(9,3,256);mask=torch.zeros(9,1,1);mask[[2,4]]=1
    out=h+mask*d[None,None];assert torch.equal(out,h)
    out.sum().backward();assert torch.equal(d.grad,torch.full((256,),6.))
    d2=torch.randn(256,requires_grad=True);out=h+mask*d2[None,None];assert torch.equal(out[[0,1,3,5,6,7,8]],h[[0,1,3,5,6,7,8]])
    g=torch.autograd.grad(objective(torch.ones(4,4),[],torch.empty(0,4),d2,.001),d2)[0];assert torch.count_nonzero(g)==0
    return dict(status='pass',zero_exact=True,only_selected_tokens_change=True,shared_delta_gradient=True,no_support_noop=True)


def input_pool(expert):
    """Original admitted frame set; alternatives from ORIGINAL retained probes."""
    frames = []
    for a in expert['anchors']['single4']:
        pos = int(a['position']); probe = expert['observations'][('original', pos)]['probe']
        b = np.asarray(probe['boxes'], dtype=np.float64).reshape(-1, 4)
        s = np.asarray(probe['target_scores'], dtype=np.float64)
        assert probe['accepted'] and 0 < len(b) <= 3 and np.isfinite(b).all() and (b[:, 2:] > 0).all()
        winner = int(s.argmax()); assert np.array_equal(b[winner], np.asarray(a['box']))
        frames.append(dict(position=pos, boxes=b, scores=s, candidate_ids=list(probe['candidate_ids']),
                           top1=winner, original_frame_accepted=True))
    return frames
