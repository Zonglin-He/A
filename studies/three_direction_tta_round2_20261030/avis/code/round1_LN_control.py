"""Episodic clip-head TTA; input tensors contain no ground-truth annotations.
This module is a diagnostic baseline, not a claim that consistency proves grounding.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class ClipHeadTTA:
    def __init__(self, module, method='entropy', steps=1, lr=1e-4):
        self.module=module; self.original=module.forward; self.method=method;self.steps=steps;self.lr=lr;self.records=[]
        self.names=[];self.params=[]
        # Restrict to audio-fusion LayerNorm affine parameters; leave visual path fixed.
        for name,p in module.named_parameters():
            p.requires_grad_(False)
            if name.startswith(('enc_av_cross_attn.','enc_av_ffn.')) and '.norm.' in name:
                self.names.append(name);self.params.append(p)
        if not self.params:raise RuntimeError('No intended audio-fusion LayerNorm parameters found')
        self.initial=[p.detach().clone() for p in self.params]
        module.forward=self.forward

    def reset(self):
        with torch.no_grad():
            for p,v in zip(self.params,self.initial):p.copy_(v);p.requires_grad_(False)

    @staticmethod
    def align(reference, other):
        # Query slots are not object identities. Match detached predicted class
        # distributions and mask vectors in the common frozen pixel-feature basis.
        # Clip mask embeddings have no temporal axis, so inverse-time restoration
        # applies to per-frame features (which remain in original order), not vectors.
        with torch.no_grad():
            p=reference['pred_logits'][-1,0].softmax(-1)
            q=other['pred_logits'][-1,0].softmax(-1)
            m=F.normalize(reference['pred_mask_embed'][-1,0],dim=-1)
            n=F.normalize(other['pred_mask_embed'][-1,0],dim=-1)
            cls=torch.cdist(p,q,p=1)/2
            cosine=m@n.T
            cost=cls+(1-cosine)
            row,col=linear_sum_assignment(cost.detach().cpu().numpy())
            assert (row==list(range(len(row)))).all()
            index=torch.tensor(col,device=p.device)
        out=dict(other)
        for key in ('pred_logits','pred_mask_embed'):
            out[key]=other[key].index_select(-2,index)
        return out, {'matching_cost_mean':float(cost[row,col].mean()),
                     'matched_mask_cosine':cosine[row,col].detach(),
                     'query_permutation_count':int((index!=torch.arange(len(index),device=index.device)).sum())}

    @staticmethod
    def average(a,b):
        out=dict(a)
        out['pred_logits']=((a['pred_logits'].softmax(-1)+b['pred_logits'].softmax(-1))/2).clamp_min(1e-8).log()
        out['pred_mask_embed']=(a['pred_mask_embed']+b['pred_mask_embed'])/2
        return out

    def forward(self, frame_query, audio_features):
        self.reset(); q=frame_query.detach();a=audio_features.detach();r={'method':self.method,'steps':self.steps,'parameter_count':sum(p.numel() for p in self.params),'losses':[]}
        with torch.no_grad():
            base=self.original(q,a)
            if self.method=='source':return base
            repeat=self.original(q,a)
            r['repeat_max_abs_error']=max(float((base[k]-repeat[k]).abs().max()) for k in ('pred_logits','pred_mask_embed'))
            if r['repeat_max_abs_error']>1e-5:raise RuntimeError('same-input deterministic control failed')
            reverse,match=self.align(base,self.original(q.flip(1),a.flip(0)))
            r.update({k:v for k,v in match.items() if k!='matched_mask_cosine'})
            teacher=self.average(base,reverse)
            target=teacher['pred_logits'][-1,0].softmax(-1).detach()
            mask_target=F.normalize(teacher['pred_mask_embed'][-1,0].detach(),dim=-1)
            # Report every query as well as the fixed confidence gate; never treat as accuracy.
            fg=target[:,:-1].max(-1).values
            agreement=(base['pred_logits'][-1,0].argmax(-1)==reverse['pred_logits'][-1,0].argmax(-1))
            gate=(fg>=0.3)&agreement&(match['matched_mask_cosine']>=0.9)
            r.update({'foreground_queries':int((fg>=0.3).sum()),'gated_queries':int(gate.sum()),'view_disagreement':float((~agreement).float().mean())})
            if self.method=='views' or self.steps==0:
                self.records.append(r);return teacher
        for p in self.params:p.requires_grad_(True)
        opt=torch.optim.Adam(self.params,lr=self.lr)
        with torch.enable_grad():
            for step in range(self.steps):
                opt.zero_grad(set_to_none=True)
                x,_=self.align(base,self.original(q,a));logp=x['pred_logits'][-1,0].log_softmax(-1);prob=logp.exp()
                if self.method=='entropy':
                    # Include no-object; explicitly diagnose trivial background collapse in scoring.
                    loss=-(prob*logp).sum(-1).mean()
                elif self.method=='consistency':
                    y,_=self.align(base,self.original(q.flip(1),a.flip(0)));logr=y['pred_logits'][-1,0].log_softmax(-1)
                    per=(F.kl_div(logp,target,reduction='none').sum(-1)+F.kl_div(logr,target,reduction='none').sum(-1))/2
                    if not gate.any():break
                    class_loss=per[gate].mean()
                    mask_per=(1-F.cosine_similarity(x['pred_mask_embed'][-1,0],mask_target,dim=-1)+1-F.cosine_similarity(y['pred_mask_embed'][-1,0],mask_target,dim=-1))/2
                    mask_loss=mask_per[gate].mean()
                    anchor=0.01*sum((p-v).square().mean() for p,v in zip(self.params,self.initial))
                    loss=class_loss+mask_loss+anchor
                    r.setdefault('class_losses',[]).append(float(class_loss.detach()));r.setdefault('mask_losses',[]).append(float(mask_loss.detach()));r.setdefault('anchor_losses',[]).append(float(anchor.detach()))
                else:raise ValueError(self.method)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite TTA loss')
                loss.backward();grad_norm=torch.nn.utils.clip_grad_norm_(self.params,1.0,error_if_nonfinite=True);opt.step()
                if not all(torch.isfinite(p).all() for p in self.params):raise RuntimeError('nonfinite adapted parameter')
                r.setdefault('gradient_norms',[]).append(float(grad_norm));r['losses'].append(float(loss.detach()))
        with torch.no_grad():
            final=self.original(q,a);final_reverse,_=self.align(final,self.original(q.flip(1),a.flip(0)));out=self.average(final,final_reverse)
            r['parameter_delta_l2']=float(sum((p-v).square().sum() for p,v in zip(self.params,self.initial)).sqrt())
        self.records.append(r);self.reset();return out
