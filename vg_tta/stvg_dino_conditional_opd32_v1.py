"""Frozen DINO reference-conditioned decoder; isolated student queries.

No annotations, dataset selection, network gradients, or text generation.
"""
import time
import numpy as np
import torch
from torchvision.ops import box_iou,box_convert
from vg_tta.stvg_error_state_correction_v1 import encode,decode


def legal(boxes):
    b=np.asarray(boxes)
    return np.isfinite(b).all(-1)&(b>0).all(-1)&(b<1).all(-1)


class ConditionalDecoder:
    def __init__(self,model,inputs,entity_tokens):
        self.model=model;self.entity_tokens=entity_tokens;self.kw=None
        def hook(module,args,kwargs):self.kw=dict(kwargs)
        h=model.model.decoder.register_forward_pre_hook(hook,with_kwargs=True)
        torch.cuda.synchronize();t=time.perf_counter()
        try:
            with torch.no_grad():self.original=model(**inputs)
        finally:h.remove()
        torch.cuda.synchronize();self.encoding_full_forward_seconds=time.perf_counter()-t
        self.textmask=inputs['attention_mask'].bool();assert self.kw is not None
        self.N=self.kw['inputs_embeds'].shape[1];self.decoder_calls=0;self.decoder_seconds=0.
        self.logical_queries=0;self.unique_queries=0

    @torch.no_grad()
    def run(self,boxes,slots=None):
        b=torch.as_tensor(boxes,dtype=torch.float32,device='cuda').reshape(1,-1,4)
        k=b.shape[1];assert k>0 and torch.isfinite(b).all()and ((b>0)&(b<1)).all()
        native=self.kw['reference_points']
        if slots is None:
            slots=box_iou(box_convert(b[0],'cxcywh','xyxy'),box_convert(native[0],'cxcywh','xyxy')).argmax(-1)
        slots=torch.as_tensor(slots,device=b.device,dtype=torch.long)
        kw=dict(self.kw);kw['inputs_embeds']=torch.cat([kw['inputs_embeds'],kw['inputs_embeds'][:,slots]],1)
        kw['reference_points']=torch.cat([native,b],1)
        # Native queries cannot see added queries. Each added query replaces its
        # matched native slot in its own attention support, avoiding double votes.
        n=self.N+k;mask=b.new_full((1,1,n,n),torch.finfo(b.dtype).min)
        mask[:,:,:self.N,:self.N]=0
        for j,slot in enumerate(slots.tolist()):
            mask[:,:,self.N+j,:self.N]=0;mask[:,:,self.N+j,slot]=torch.finfo(b.dtype).min
            mask[:,:,self.N+j,self.N+j]=0
        kw['self_attn_mask']=mask
        old=self.model.model.decoder.config.num_queries
        torch.cuda.synchronize();tick=time.perf_counter()
        try:
            self.model.model.decoder.config.num_queries=n
            out=self.model.model.decoder(**kw)
        finally:self.model.model.decoder.config.num_queries=old
        hidden=out.intermediate_hidden_states[:,-1]
        ref=out.intermediate_reference_points[:,-2]
        pred=(self.model.bbox_embed[-1](hidden)+torch.special.logit(ref,eps=1e-5)).sigmoid()
        logits=self.model.class_embed[-1](hidden,self.kw['text_encoder_hidden_states'],self.textmask)
        torch.cuda.synchronize();self.decoder_seconds+=time.perf_counter()-tick;self.decoder_calls+=1
        assert torch.isfinite(pred).all()
        return pred[0,self.N:].cpu().numpy(),logits[0,self.N:].sigmoid()[:,self.entity_tokens].mean(-1).cpu().numpy(),slots.cpu().numpy(),pred[0,:self.N].cpu()

    @torch.no_grad()
    def qualify(self):
        tick=time.perf_counter();out=self.model.model.decoder(**self.kw)
        hidden=out.intermediate_hidden_states[:,-1];ref=out.intermediate_reference_points[:,-2]
        b=(self.model.bbox_embed[-1](hidden)+torch.special.logit(ref,eps=1e-5)).sigmoid()
        err=float((b-self.original.pred_boxes).abs().max());assert err==0,err
        slots=torch.tensor([0,self.N//2,self.N-1],device='cuda')
        clone,_,_,native=self.run(self.kw['reference_points'][0,slots],slots)
        cloneerr=float(abs(clone-self.original.pred_boxes[0,slots].cpu().numpy()).max())
        nativeerr=float((native-self.original.pred_boxes[0].cpu()).abs().max())
        assert max(cloneerr,nativeerr)<1e-4,(cloneerr,nativeerr)
        return dict(status='pass',native_replay_max_error=err,cloned_native_box_max_error=cloneerr,
                    native_context_max_error=nativeerr,box_tolerance_prelocked=1e-4,seconds=time.perf_counter()-tick,
                    native_queries=self.N,content='nearest original encoder proposal IoU; first argmax',
                    external_queries_isolated=True,native_context_does_not_read_external=True)


def conditional_targets(caches,source,actions,scale,positions):
    """One teacher correction per complete shared4 action; no coordinate clamp.

    Invalid references or no score>=.35 => teacher returns the input action.
    Each accepted anchor gives one shared correction observation. No GT masks.
    """
    actions=np.asarray(actions,np.float64);s=np.asarray(scale);z=encode(source)
    boxes=np.stack([decode(z+v*s)for v in actions]);out=actions.copy();results=[]
    valid=legal(boxes[:,positions]).all(1)
    per=[{}for _ in actions]
    for pos,cache in zip(positions,caches):
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        refs=boxes[ids,pos];unique,inverse=np.unique(refs,axis=0,return_inverse=True)
        pred,score,slot,_=cache.run(unique)
        cache.logical_queries+=len(ids);cache.unique_queries+=len(unique)
        for j,ix in enumerate(ids):
            p=pred[inverse[j]];v=float(score[inverse[j]])
            per[ix][pos]=dict(box=p,score=v,slot=int(slot[inverse[j]]),accepted=v>=.35)
    for i,a in enumerate(actions):
        good=[encode(np.array([v['box']]))[0]-z[pos]for pos,v in per[i].items()if v['accepted']]
        if good:out[i]=np.mean(good,axis=0)/s
        results.append(dict(legal=bool(valid[i]),accepted_anchors=len(good),anchors=per[i],
                            action=a.copy(),teacher_action=out[i].copy()))
    assert np.isfinite(out).all()
    return out,results


def distill(mu,targets):
    # Exact M-projection: argmin_mu mean KL(N(target,I)||N(mu,I)).
    # This is distribution distillation, not an IoU-to-fixed-E reward loop.
    new=np.asarray(targets,np.float64).mean(0)
    before=float(np.square(targets-mu).sum(-1).mean()/2)
    after=float(np.square(targets-new).sum(-1).mean()/2)
    assert after<=before+1e-10
    return new,dict(loss_before=before,loss_after=after,mean_step_l2=float(np.linalg.norm(new-mu)),
                    student_mean_before=np.asarray(mu).copy(),student_mean_after=new.copy())
