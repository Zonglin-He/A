"""Persistent original start/end last layer, on a restricted paired-span policy.

No extra ranker. Both native offsets remain. The eight retained physical
envelopes carry their actual generating per-offset legal spans. This is not the
full span likelihood: it is the product of native legal span probabilities,
renormalized on the retained pairs. No GT is available to this module.
"""
import numpy as np
import torch
from methods.decota_final_simplified_v1.tensors import detached,state_hash
from methods.decota_final_simplified_v1.objectives import legal_logp,native_view_indices
from vg_tta.tastvg_corruption_c0c1_v1 import ranked_spans,envelope
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
from vg_tta.tastvg_proximal_method_v1 import OnlineMethod,norm,project_arrival

def support_pairs(layers,support,records,ids):
    ranked=[ranked_spans(z) for z in layers[-1]['logits']]
    pairs=[]
    for c in support:
        origin=c['origin']
        if origin.startswith('layer'):
            layer=layers[int(origin[5:])-1]
            pair=[list(native_view_indices(z)) for z in layer['logits']]
        else:
            assert origin.startswith('final_top_pair:')
            indices=[int(x) for x in origin.split(':')[1].split(',')]
            pair=[list(ranked[i][indices[i]][1:]) for i in [0,1]]
        assert all(s<e for s,e in pair)
        assert envelope(pair[0],pair[1],records,ids)==c['indices']
        pairs.append(pair)
    return pairs

def restricted_logp(logits,pairs):
    values=[]
    for offset,z in enumerate(logits):
        lp,ij=legal_logp(z)
        mapping={(int(s),int(e)):j for j,(s,e) in enumerate(ij.T.cpu().tolist())}
        index=torch.tensor([mapping[tuple(pair[offset])] for pair in pairs],device=z.device)
        values.append(lp[index])
    score=values[0]+values[1]
    return score-torch.logsumexp(score,0)

class HeadActor:
    def __init__(self,model):
        self.named=[('temporal.last.'+n,p) for n,p in model.temp_embed.layers[-1].named_parameters()]
        assert sum(p.numel() for _,p in self.named)==514
        self.initial=self.state()
        for _,p in self.named:p.requires_grad_(True)
    def state(self):return {n:p.detach().clone() for n,p in self.named}
    def restore(self,state):
        with torch.no_grad():
            for n,p in self.named:p.copy_(state[n]);p.grad=None
    def close(self):
        self.restore(self.initial)
        for _,p in self.named:p.requires_grad_(False)

class NativeOnlineMethod(OnlineMethod):
    def __init__(self,model,deltas,*,head_lr,head_s_ref,head_teacher_temperature,
                 head_cap_fraction=.005,**kwargs):
        super().__init__(model,deltas,**kwargs)
        self.head=HeadActor(model);self.head_lr=float(head_lr);self.head_s_ref=float(head_s_ref)
        self.head_teacher_temperature=float(head_teacher_temperature)
        self.head_radius=head_cap_fraction*norm(self.head.initial)
    def close(self):
        self.head.close();super().close()
    def arrive(self,data,scheduled,temporal_provider=None,spatial_provider=None):
        hidden=[];before=self.head.state()
        def observe(module,args,out):
            if len(hidden)<2:hidden.append(args[0].detach())
        h=self.model.temp_embed.register_forward_hook(observe)
        try:result,ev=super().arrive(data,scheduled,temporal_provider,spatial_provider)
        finally:h.remove()
        assert len(hidden)==2
        result['temporal_pre_state']=detached(before,'cpu');result['temporal_pre_sha']=state_hash(before)
        result['native_temporal_update']=None
        if scheduled:
            # Current output was already sealed by the spatial/Fast method.
            pairs=support_pairs(result['temporal_layers'],result['temporal']['candidates'],data['records'],data['frame_ids'])
            logits=[self.model.temp_embed(v)[-1] for v in hidden]
            for z,old in zip(logits,result['prediction']['logits']):assert torch.equal(z.detach().cpu(),old)
            lp=restricted_logp(logits,pairs);p_ref=lp.detach().exp()
            scores=np.asarray(result['temporal']['scores']);spread=float(np.ptp(scores))
            strength=spread/(spread+self.head_s_ref);rank=average_ranks(scores)
            lrq=(-torch.tensor(rank,device=lp.device,dtype=lp.dtype)/self.head_teacher_temperature).log_softmax(0)
            if strength==0.:target=lp.detach();loss=sum(z.sum() for z in logits)*0.
            else:
                target=torch.logaddexp(lp.detach()+np.log1p(-strength),lrq.detach()+np.log(strength))
                loss=(lp.exp()*(lp-target)).sum()
            grad=torch.autograd.grad(loss,[p for _,p in self.head.named])
            assert torch.isfinite(loss) and all(torch.isfinite(g).all() for g in grad)
            with torch.no_grad():
                for (_,p),g in zip(self.head.named,grad):p.add_(g,alpha=-self.head_lr)
                projection=project_arrival(self.head,before,self.head_radius)
                after_logits=[self.model.temp_embed(v)[-1] for v in hidden]
                after_lp=restricted_logp(after_logits,pairs)
                after_loss=(after_lp.exp()*(after_lp-target)).sum()
            result['native_temporal_update']=dict(probability_scope='restricted_native_legal_span_pair_support',
                pairs=pairs,scores=scores.tolist(),rank=rank.tolist(),p=p_ref.cpu(),q_rank=lrq.detach().exp().cpu(),
                target=target.exp().cpu(),frozen_log_target=target.cpu(),strength=strength,spread=spread,
                s_ref=self.head_s_ref,lr=self.head_lr,teacher_temperature=self.head_teacher_temperature,
                loss_before=float(loss.detach()),loss_after=float(after_loss),
                gradients={n:g.detach().cpu() for (n,_),g in zip(self.head.named,grad)},
                logits_before=detached(logits,'cpu'),logits_after=detached(after_logits,'cpu'),
                global_gradient_norm=float(torch.sqrt(sum(g.double().square().sum() for g in grad))),
                updated=any(torch.count_nonzero(g)>0 for g in grad),**projection)
        after=self.head.state();result['temporal_post_state']=detached(after,'cpu');result['temporal_post_sha']=state_hash(after)
        result['compute']['temporal_head_backward_calls']=int(scheduled)
        result['compute']['temporal_head_forward_calls']=4*int(scheduled)
        return result,ev
