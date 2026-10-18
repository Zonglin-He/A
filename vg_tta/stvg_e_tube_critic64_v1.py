"""Frozen event-critic interface around E. No labels or dataset access."""
from types import SimpleNamespace
import numpy as np
import torch
from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table
from vg_tta.stvg_error_state_correction_v1 import encode,decode
from vg_tta.stvg_joint_event_critic_v1 import extract_visual,candidate_tensor,collate


def intervals(source,mbr):
    ids,q=mass_table(probabilities(source['records'],source['logits']))
    other=sorted([(float(q[i,j]),int(ids[i]),int(ids[j])+1)for i,j in zip(*np.where(q>0))],key=lambda x:(-x[0],x[1],x[2]))
    iv=[list(map(int,mbr))]
    for prob,a,b in other:
        if [a,b]not in iv:iv.append([a,b])
        if len(iv)==4:break
    return iv,ids,q


def states(source,E,delta):
    s=np.asarray(source,np.float64);e=np.asarray(E,np.float64);d=np.asarray(delta,np.float64)
    assert np.max(abs(decode(encode(s)+d)-e))<1e-12
    return [s.copy(),decode(encode(s)+.5*d),e.copy()]


@torch.no_grad()
def normalized_maps(base):
    return [dict(prefix=base.norm(v['prefix']).detach().float(),info=dict(fea_map_size=v['info']['fea_map_size']))for v in base.views]


def from_maps(maps):return SimpleNamespace(norm=lambda x:x,views=maps)


@torch.no_grad()
def score_grid(critic,features,iv,fps):
    examples=[candidate_tensor(f,I,fps)for f in features for I in iv]
    y=critic(*collate(examples)).reshape(len(features),len(iv))
    assert torch.isfinite(y).all();return y.detach().cpu()


def counterfactual_advantage(scores):
    """Independent samples; omit own sample from each conditional baseline."""
    assert scores.shape==(8,4)
    return scores-(scores.sum(0,keepdim=True)-scores)/7,scores-(scores.sum(1,keepdim=True)-scores)/3


def contract():
    c=torch.arange(32,dtype=torch.float64).reshape(8,4);a,b=counterfactual_advantage(c)
    assert torch.allclose(a.sum(0),torch.zeros(4,dtype=c.dtype))and torch.allclose(b.sum(1),torch.zeros(8,dtype=c.dtype))
    # A component-only reward cannot provide advantage to the other component.
    assert torch.count_nonzero(counterfactual_advantage(c[:,0,None].repeat(1,4))[1])==0
    assert torch.count_nonzero(counterfactual_advantage(c[0,None,:].repeat(8,1))[0])==0
    return dict(status='pass',LOO_not_self_baseline=True,no_GT_or_model=True)
