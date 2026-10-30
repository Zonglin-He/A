"""Round4 engineering primitives only. No adaptation loss or teacher selected.
Inputs are frozen cached tensors, not a loaded model. Production choice waits for
parent development error decomposition. This module does not read data or files.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

class Rank4QueryResidual(nn.Module):
    """512 trainable parameters at64D, exactzero output init and per-episode reset.
    A starts orthonormal, B startszero: bothzero would kill the first gradient.
    Relative normalization preserves each sourcevector's original norm, so source
    floatingpoint query projections are not renormalized at step0.
    """
    def __init__(self,dim=64,rank=4,seed=20261030,dtype=torch.float32):
        super().__init__()
        if not 0<rank<=dim:raise ValueError('invalid rank')
        generator=torch.Generator(device='cpu').manual_seed(seed)
        initial=torch.linalg.qr(torch.randn(dim,rank,generator=generator,dtype=dtype),mode='reduced')[0].T.contiguous()
        self.register_buffer('_down_initial',initial.clone());self.down=nn.Parameter(initial.clone());self.up=nn.Parameter(torch.zeros(dim,rank,dtype=dtype));self.dim=dim;self.rank=rank
    def forward(self,source_queries):
        q=source_queries.detach()
        if q.shape[-1]!=self.dim:raise ValueError('query projection dimension mismatch')
        delta=F.linear(F.linear(q,self.down),self.up)
        shifted=q+delta;original_norm=torch.linalg.vector_norm(q,dim=-1,keepdim=True);new_norm=torch.linalg.vector_norm(shifted,dim=-1,keepdim=True)
        # Zero sourcevectors remainzero, ratherthan inventing a semantic direction.
        ratio=original_norm/new_norm.clamp_min(torch.finfo(q.dtype).tiny)
        return shifted*ratio
    @torch.no_grad()
    def reset_episode(self):
        self.down.copy_(self._down_initial);self.up.zero_()
        for p in self.parameters():p.grad=None
    def parameter_delta_l2(self):
        return torch.sqrt((self.down-self._down_initial).square().sum()+self.up.square().sum())


def delta_anchored_logits(adapted_queries,source_queries,frozen_tokens,exact_source_logits,temperature=.07):
    """Cache-kernel bridge: exact source logits plus a differentiable delta.
    Token projection is modelpoint-conditioned, NOT independent language evidence.
    No token deletion or padding/special/sentinel change insoftmax denominator.
    """
    if temperature<=0:raise ValueError('temperature mustbepositive')
    q=source_queries.detach();tokens=frozen_tokens.detach();logits=exact_source_logits.detach()
    if adapted_queries.shape!=q.shape or q.shape[-1]!=tokens.shape[-1]:raise ValueError('projection shapes mismatch')
    if logits.shape!=(q.shape[-2],tokens.shape[-2]):raise ValueError('source logit shape mismatch')
    difference=F.normalize(adapted_queries,p=2,dim=-1)-F.normalize(q,p=2,dim=-1)
    return logits+difference@tokens.T/temperature


def anchored_span_scores(adapted_queries,source_queries,frozen_tokens,exact_source_logits,span_mask,exact_source_span_scores,temperature=.07):
    """Exactly recovers cached source readout atzero delta, including acrossCPU/GPU.
    The anchor corrects only source numerical reconstruction; it does not label
    context tokens or select a teacher. Output is a ranking score, not a newly
    calibrated probability. rawCPU re-softmax can otherwise move tiedargmaxes.
    """
    logits=delta_anchored_logits(adapted_queries,source_queries,frozen_tokens,exact_source_logits,temperature)
    base=exact_source_logits.detach();mask=span_mask.detach().to(dtype=logits.dtype)
    if mask.shape!=(base.shape[-1],):raise ValueError('span maskshape mismatch')
    difference=((logits.softmax(-1)-base.softmax(-1))*mask[None]).sum(-1)
    return exact_source_span_scores.detach()+difference


def direct_span_scores(queries,tokens,span_mask,temperature=.07):
    """Sameinformation no-update numericalreadout; no adapteror groundingclaim."""
    logits=(queries.detach()@tokens.detach().T)/temperature
    return (logits.softmax(-1)*span_mask.to(logits.dtype)[None]).sum(-1)


class SizePriorPoseLatent(nn.Module):
    """Optional future geometryinterface, not a selected geometryteacher/objective.
    Source sizes are frozen byte-for-byte. CenterL2 and yaw budgets mustbe supplied
    only after decomposition/protocol selection; there are no researchdefaults.
    A hard eligible mask prevents sharedgeometry updates leaking to otherqueries.
    """
    def __init__(self,source_boxes,eligible,center_budget_m,yaw_budget_radians):
        super().__init__();s=source_boxes.detach().clone()
        if s.ndim!=2 or s.shape[1] not in [6,7] or not s.is_floating_point():raise ValueError('need Nx6/7 floating sourceboxes')
        if s.shape[1]==6:s=torch.cat([s,s.new_zeros((len(s),1))],-1)
        if not torch.isfinite(s).all():raise ValueError('nonfinite source')
        if center_budget_m<=0 or yaw_budget_radians<=0:raise ValueError('explicitpositive physicalbudgets required')
        e=eligible.detach().bool()&(s[:,3:6]>0).all(-1)
        if e.shape!=(len(s),):raise ValueError('eligiblemask mismatch')
        self.register_buffer('source',s);self.register_buffer('eligible',e);self.raw=nn.Parameter(s.new_zeros((len(s),4)));self.center_budget_m=float(center_budget_m);self.yaw_budget_radians=float(yaw_budget_radians)
    def forward(self):
        center=self.source[:,:3]+self.center_budget_m/math.sqrt(3)*self.raw[:,:3].tanh()
        yaw=self.source[:,6]+self.yaw_budget_radians*self.raw[:,3].tanh()
        center=torch.where(self.eligible[:,None],center,self.source[:,:3]);yaw=torch.where(self.eligible,yaw,self.source[:,6])
        return torch.cat([center,self.source[:,3:6],yaw[:,None]],-1)
    @torch.no_grad()
    def reset_episode(self):
        self.raw.zero_();self.raw.grad=None


def validate_cache_arrays(arrays,expected_queries=256,expected_dim=64):
    """Shape/finite contract only; caller independently verifies cache file seals."""
    import numpy as np
    required=['boxes','semantic_scores','proj_queries','proj_tokens','exact_logits','token_scores','input_ids','attention_mask','offset_mapping','noun_mask','caption_mask','context_mask']
    for name in required:
        if name not in arrays:raise ValueError('missingcachefield '+name)
    q=np.asarray(arrays['proj_queries']);t=np.asarray(arrays['proj_tokens']);n=expected_queries;L=len(t)
    if q.shape!=(n,expected_dim) or t.ndim!=2 or t.shape[1]!=expected_dim:raise ValueError('embeddingcache shape mismatch')
    shapes={'boxes':(n,6),'semantic_scores':(n,),'exact_logits':(n,L),'token_scores':(n,L),'offset_mapping':(L,2)}
    for key,shape in shapes.items():
        if np.asarray(arrays[key]).shape!=shape:raise ValueError(key+'shape mismatch')
    for key in ['input_ids','attention_mask','noun_mask','caption_mask','context_mask']:
        if np.asarray(arrays[key]).reshape(-1).shape!=(L,):raise ValueError(key+'shape mismatch')
    for key in ['boxes','semantic_scores','proj_queries','proj_tokens','exact_logits','token_scores']:
        if not np.isfinite(np.asarray(arrays[key])).all():raise ValueError(key+'nonfinite')
    return {'queries':n,'tokens':L,'embedding_dim':expected_dim,'source_zero_query_vectors':int((np.linalg.norm(q,axis=-1)==0).sum()),'token_embeddings_are_point_conditioned':True,'query_slot_alignment_is_not_identity':True,'allactualtokens_preserved_for_denominator':True}
