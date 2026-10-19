"""Native temporal query factory and six existing shared time-head outputs.
No new inputs, learned mappings, layer selection, model or spatial updates.
"""
import numpy as np
import torch


def indices(mask):
    return torch.nonzero(mask).reshape(-1).tolist()


@torch.no_grad()
def native_stages(model,kw,spatial_kw,g):
    l=g['h']*g['w'];motion=kw['encoded_feature'];appearance=spatial_kw['encoded_feature'];T=motion.shape[1]
    assert torch.equal(appearance[l:],motion[:-l])
    ft=motion[:-l].mean(1).unsqueeze(0);H=motion[-l:].permute(1,0,2)
    fm=H.permute(0,2,1).reshape(T,256,g['h'],g['w']);fa=appearance[:l].permute(1,2,0).reshape_as(fm)
    lm=model.t_temporal_clas(fm,ft);la=model.s_temporal_clas(fa,ft);p=(lm.sigmoid()+la.sigmoid())/2
    fallback=indices(p>0);first=indices(p>model.theta)or fallback
    _,am1=model.t_spatial_clas(fm[first],ft[:,:1]);q1=(H[first]*am1[...,None]).mean((0,1))
    k1=dict(kw,query_tgt=q1[None,None,:].expand_as(kw['query_tgt']));h1=model.ground_decoder.time_decoder(**k1);logits1=model.temp_embed(h1);action_logits=model.action_embed(h1)[-1].squeeze();ap=action_logits.sigmoid();second=indices(ap>.5)or fallback
    _,am2=model.t_spatial_clas(fm[second],ft[:,:1]);q2=(H[second]*am2[...,None]).mean((0,1))
    assert torch.equal(q2,kw['query_tgt'][0,0]),'native second query must be bit-exact before measuring stage differences'
    k2=dict(kw,query_tgt=q2[None,None,:].expand_as(kw['query_tgt']));h2=model.ground_decoder.time_decoder(**k2);logits2=model.temp_embed(h2)
    assert logits1.shape==logits2.shape==(6,1,T,2)
    return logits1.cpu(),logits2.cpu(),dict(first=first,second=second,fallback=fallback,theta=float(model.theta),initial_motion_logits=lm.cpu(),initial_appearance_logits=la.cpu(),initial_gate_prob=p.cpu(),action_logits1=action_logits.cpu(),action1=ap.cpu(),attention1=am1.cpu(),attention2=am2.cpu(),query1=q1.cpu(),query2=q2.cpu(),native_query2=kw['query_tgt'][0,0].cpu())


def ensemble(q0,pass1,pass2,hires):
    assert len(pass1)==len(pass2)==6 and np.array_equal(pass2[-1],q0)
    u=(q0>0).astype(float);u/=u.sum();m1=np.mean(pass1,axis=0);m2=np.mean(pass2,axis=0)
    out=dict(FirstPassMix=.5*q0+.5*pass1[-1],Pass1Mean6=m1,Pass2Mean6=m2,BothMean12=.5*m1+.5*m2,HiResAuxMix=.5*hires+.5*m2,NativeUniformMix=.5*q0+.5*u,HiResUniformMix=.5*hires+.5*u)
    assert all(abs(q.sum()-1)<1e-12 and np.isfinite(q).all()and(q>=0).all()for q in out.values())
    return out


def properties():
    rng=np.random.default_rng(41);q=np.triu(rng.random((8,8)),1);q/=q.sum();out=ensemble(q,[q]*6,[q]*6,q)
    assert all(np.max(abs(out[k]-q))<1e-15 for k in ['FirstPassMix','Pass1Mean6','Pass2Mean6','BothMean12','HiResAuxMix'])
    return dict(status='pass',identical_heads_ensemble_identity_to_roundoff=True,uniform_controls_same_positive_support=True,native_query_factory_not_norm_matched=True)
