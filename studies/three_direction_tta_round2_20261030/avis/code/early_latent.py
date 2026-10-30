"""Per-video early query latent; no annotations or cross-video mutable state."""
import contextlib, math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def pair_matches(logits, masks, teacher_probs, teacher_masks, teacher_ids, min_dice=.5):
    """Detached one-to-one class-compatible clip->frame proposal matching."""
    with torch.no_grad():
        if len(teacher_ids)==0:return []
        p=logits.softmax(-1); classes=teacher_probs[teacher_ids,:-1].argmax(-1)
        q=masks.sigmoid().flatten(1); target=teacher_masks[teacher_ids].sigmoid().flatten(1)
        dice=(2*target@q.T+1)/(target.sum(-1)[:,None]+q.sum(-1)[None,:]+1)
        compatible=p.argmax(-1)[None,:]==classes[:,None]
        cost=(1-dice)-p[:,classes].T+2*(~compatible)
        row,col=linear_sum_assignment(cost.cpu().numpy())
        result=[]
        for r,c in zip(row,col):
            if compatible[r,c] and dice[r,c]>=min_dice and (target[r]>.5).sum()>=4:
                result.append((int(teacher_ids[r]),int(c),float(dice[r,c])))
        return result


class VideoTrace:
    """Capture frozen inputs/outputs; registering hooks does not alter source forward."""
    def __init__(self,model,anchors,full=True):
        self.model=model;self.full=full;self.anchors=set(anchors);self.offset=0;self.cache={};self.frames=[];self.projected=[];self.handles=[];self.clip=None;self.audio_query=[]
    def __enter__(self):
        self.handles=[]
        # Pixel decoder is invoked via forward_features rather than __call__.
        decoder=self.model.sem_seg_head.pixel_decoder
        self.pixel_original=decoder.forward_features
        def pixel(features):
            out=self.pixel_original(features)
            self.pixel_out=out
            return out
        decoder.forward_features=pixel
        self.handles.extend([self.model.sem_seg_head.predictor.register_forward_hook(self._frame),self.model.avism_module.avism_mask_features.register_forward_hook(self._projected),self.model.avism_module.register_forward_hook(self._clip),self.model.sem_seg_head.predictor.av_post_proj.register_forward_hook(self._audio)])
        return self
    def _audio(self,mod,inp,out):self.audio_query.append(out.detach().cpu())
    def _projected(self,mod,inp,out):
        if self.full:self.projected.append(out.detach().cpu())
    def _frame(self,mod,inp,out):
        prediction,q,_=out; n=q.shape[1]
        if not self.full:self.offset+=n;return
        for i in range(n):
            t=self.offset+i
            self.frames.append({'logits':prediction['pred_logits'][i].detach().cpu(),'masks':prediction['pred_masks'][i].detach().cpu()})
            if t in self.anchors:
                mf,cm,_,multi=self.pixel_out
                self.cache[t]={'mask':mf[i:i+1].detach().cpu(),'clip_mask':cm[i:i+1].detach().cpu(),'multi':[m[i:i+1].detach().cpu() for m in multi]}
        self.offset+=n
    def _clip(self,mod,inp,out):
        self.frame_queries=inp[0].detach().clone();self.audio=inp[1].detach().clone();self.clip={k:v.detach().clone() for k,v in out.items() if torch.is_tensor(v)}
    def __exit__(self,*exc):
        self.model.sem_seg_head.pixel_decoder.forward_features=self.pixel_original
        for h in self.handles:h.remove()
        self.projected=torch.cat(self.projected) if self.projected else None
        self.audio_query=torch.cat(self.audio_query,dim=1)
        self.pixel_out=None


@contextlib.contextmanager
def query_latent(model,latent):
    def add(mod,inp,out):return out+latent[:,None,:]
    h=model.sem_seg_head.predictor.av_post_proj.register_forward_hook(add)
    try:yield
    finally:h.remove()


def make_targets(trace,anchors,device):
    probs=trace.clip['pred_logits'][-1,0].softmax(-1)
    candidates=((probs.argmax(-1)<probs.shape[-1]-1)&(probs[:,:-1].amax(-1)>=.3)).nonzero().flatten()
    embed=trace.clip['pred_mask_embed'][-1,0]
    pairs={};teacher={};binary=[]
    for t in anchors:
        m=torch.einsum('qc,chw->qhw',embed,trace.projected[t].to(device))
        teacher[t]=m.detach();binary.append(m[candidates]>0)
        x=trace.frames[t]
        pairs[t]=pair_matches(x['logits'].to(device),x['masks'].to(device),probs,m,candidates)
    # Exclude ambiguous different-proposal negatives. Low overlap is not proof
    # of physical identity; this is an explicit pseudo-trajectory assumption.
    disjoint=torch.ones(len(candidates),len(candidates),dtype=torch.bool,device=device)
    for masks in binary:
        a=masks.flatten(1).float();inter=a@a.T;union=a.sum(1)[:,None]+a.sum(1)[None,:]-inter
        disjoint &= (inter/(union+1e-6)<.1)
    negative={(int(a),int(b)) for i,a in enumerate(candidates) for j,b in enumerate(candidates) if i!=j and bool(disjoint[i,j])}
    return probs,teacher,pairs,negative


def adapt_latent(model,trace,anchors,identity=True,steps=3,lr=.03):
    device=model.device;predictor=model.sem_seg_head.predictor
    latent=torch.zeros((100,256),device=device,requires_grad=True)
    probs,teacher,pairs,negative=make_targets(trace,anchors,device)
    total=sum(map(len,pairs.values()));rec={'parameterization':'per-video100x256 early query latent; pretrained weights frozen','pair_count':total,'pairs_by_anchor':{str(t):p for t,p in pairs.items()},'identity_negative_track_pairs':len(negative),'identity_enabled':identity,'losses':[],'latent_initial_max_abs':float(latent.abs().max())}
    if not total:return latent.detach(),dict(rec,skip='no_pseudo_correspondences',latent_delta_l2=0.)
    opt=torch.optim.Adam([latent],lr=lr)
    fixed_clip=trace.clip['pred_cq_embed'][-1,0].detach()
    with torch.enable_grad():
      for step in range(steps):
        opt.zero_grad(set_to_none=True);pred_queries={};cls_loss=latent.sum()*0;mask_loss=latent.sum()*0;dice_loss=latent.sum()*0
        # Build all eight local forward graphs; no backbone/pixel decoder grads.
        with query_latent(model,latent):
          for t in anchors:
            c=trace.cache[t]
            pred,q,_=predictor([m.to(device) for m in c['multi']],c['mask'].to(device),c['clip_mask'].to(device),trace.audio[t:t+1])
            pred_queries[t]=q[-1,0]
            for cq,fq,quality in pairs[t]:
              logit=pred['pred_logits'][0,fq];pm=pred['pred_masks'][0,fq];target=teacher[t][cq].sigmoid()
              cls_loss+=-(probs[cq]*logit.log_softmax(-1)).sum()/total
              mask_loss+=F.binary_cross_entropy_with_logits(pm,target)/total
              soft=pm.sigmoid();dice_loss+=(1-(2*(soft*target).sum()+1)/(soft.sum()+target.sum()+1))/total
        # Replace only anchor queries in the whole frozen context; tracker
        # returns embeddings at the location used by original training.
        identity_loss=latent.sum()*0
        if identity:
          q_all=torch.stack([pred_queries.get(t,trace.frame_queries[0,t]) for t in range(trace.frame_queries.shape[1])])[None]
          clip=model.avism_module(q_all,trace.audio);nodes=[];ids=[]
          for t in anchors:
            for cq,fq,_ in pairs[t]:nodes.append(clip['pred_fq_embed'][0,0,t,fq]);ids.append(cq)
          selected=sorted(set(ids))
          nodes.extend([fixed_clip[q] for q in selected]);ids.extend(selected)
          emb=torch.stack(nodes);labels=torch.tensor(ids,device=device);equal=labels[:,None]==labels[None,:]
          valid=equal.clone()
          for a,b in negative:valid|=(labels[:,None]==a)&(labels[None,:]==b)
          # Teacher-teacher entries are constant and excluded from denominator.
          nframe=total;valid[nframe:,nframe:]=False
          identity_loss=F.binary_cross_entropy_with_logits(emb@emb.T,equal.float(),reduction='none')[valid].mean()
        anchor=.01*latent.square().mean()
        loss=2*cls_loss+5*mask_loss+5*dice_loss+.5*identity_loss+anchor
        if not torch.isfinite(loss):raise RuntimeError('nonfinite latent objective')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([latent],1.,error_if_nonfinite=True)
        if not latent.grad.isfinite().all():raise RuntimeError('nonfinite latent gradient')
        opt.step()
        with torch.no_grad():latent.clamp_(-.25,.25)
        rec['losses'].append({'total':float(loss.detach()),'class':float(cls_loss.detach()),'mask_bce':float(mask_loss.detach()),'dice':float(dice_loss.detach()),'identity':float(identity_loss.detach()),'anchor':float(anchor.detach()),'gradient_norm_preclip':float(grad),'latent_l2':float(latent.detach().norm()),'latent_max_abs':float(latent.detach().abs().max())})
        del pred_queries
    rec['latent_delta_l2']=float(latent.detach().norm());rec['pretrained_trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert rec['pretrained_trainable_parameters']==0
    return latent.detach(),rec
