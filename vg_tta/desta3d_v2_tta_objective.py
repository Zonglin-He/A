"""Explicit pre-gate v2 TTA objective, not an efficacy result.

Caller must provide corresponding physical cells from two video views and a
detached source-fit teacher. Gates are intentionally frozen: every objective
here precedes the residual gates. No occupancy-to-presence inequality is used.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibrationWeights:
    latent: float = 1.0
    referent: float = 1.0
    event: float = 1.0
    alignment: float = 0.0
    parameter_anchor: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in self.__dict__.values()):
            raise ValueError('loss weights must be finite and nonnegative')


def configure_pre_gate_calibration(adapter):
    adapter.set_train_stage('tta')
    for p in adapter.parameter_groups()['gates']:
        p.requires_grad_(False)
    initial = {n:p.detach().clone() for n,p in adapter.named_parameters() if p.requires_grad}
    return {'initial_parameters':initial,
            'parameter_count':sum(x.numel() for x in initial.values()),
            'updated_groups':['branch_film','norm_affine'],
            'excluded_groups':{'gates':'all current objectives are pre-gate; no correction gradient'},
            'joint_weight':0.0}


def bernoulli_kl(student_logits, teacher_logits):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError('corresponding physical support required')
    target=teacher_logits.detach().float().sigmoid()
    log_t=F.logsigmoid(teacher_logits.detach().float())
    log_not_t=F.logsigmoid(-teacher_logits.detach().float())
    s=student_logits.float()
    return (target*(log_t-F.logsigmoid(s))+(1-target)*(log_not_t-F.logsigmoid(-s))).mean()


def calibration_objective(adapter, student, teacher, initial_parameters,
                          *, weights=CalibrationWeights(), source_moments=None):
    terms={}
    latent=[];alignment=[]
    if weights.alignment:
        if source_moments is None or source_moments.get('feature_definition')!='v2_query_conditioned_readers':
            raise ValueError('v2 query-conditioned source moments required; v1 moments are invalid')
    for branch in ('spatial','event'):
        key='branch_features_'+branch
        current=student[key].float();target=teacher[key].detach().float()
        if current.shape!=target.shape or current.ndim!=5:
            raise ValueError('views must correspond on the same physical THW grid')
        latent.append(F.mse_loss(F.normalize(current,dim=-1),F.normalize(target,dim=-1)))
        if weights.alignment:
            pooled=current.reshape(-1,current.shape[-1])
            mean=pooled.mean(0);std=pooled.var(0,unbiased=False).clamp_min(1e-12).sqrt()
            record=source_moments[branch]
            m=torch.as_tensor(record['mean'],device=mean.device,dtype=mean.dtype).detach()
            s=torch.as_tensor(record['std'],device=std.device,dtype=std.dtype).detach()
            if mean.shape!=m.shape or std.shape!=s.shape or not torch.isfinite(m).all() or not torch.isfinite(s).all() or (s<0).any():
                raise ValueError('invalid source channel moments')
            alignment.append(F.mse_loss(mean,m)+F.mse_loss(std,s))
    terms['latent']=torch.stack(latent).mean()
    terms['referent']=bernoulli_kl(student['referent_logits'],teacher['referent_logits'])
    terms['event']=bernoulli_kl(student['event_logits'],teacher['event_logits'])
    terms['alignment']=torch.stack(alignment).mean() if alignment else terms['latent'].new_zeros(())
    active={n:p for n,p in adapter.named_parameters() if p.requires_grad}
    if set(active)!=set(initial_parameters):
        raise ValueError('anchor must cover exactly the declared updated parameters')
    squared=[(p-initial_parameters[n].detach().to(p)).square().sum() for n,p in active.items()]
    terms['parameter_anchor']=torch.stack(squared).sum()/sum(p.numel() for p in active.values())
    total=sum(getattr(weights,n)*value for n,value in terms.items())
    if not torch.isfinite(total):raise RuntimeError('nonfinite TTA objective')
    return total,terms


def gradient_groups(adapter):
    result={}
    for name,params in adapter.parameter_groups().items():
        row={'parameters':sum(p.numel() for p in params),
             'trainable_parameters':sum(p.numel() for p in params if p.requires_grad),
             'none_tensors':0,'zero_tensors':0,'nonzero_tensors':0,'gradient_l2':0.0}
        squared=0.
        for p in params:
            if p.grad is None:row['none_tensors']+=1
            else:
                g=p.grad.detach().float()
                if not torch.isfinite(g).all():raise RuntimeError('nonfinite group gradient: '+name)
                row['nonzero_tensors' if g.count_nonzero() else 'zero_tensors']+=1
                squared+=float(g.square().sum())
        row['gradient_l2']=math.sqrt(squared);result[name]=row
    return result
