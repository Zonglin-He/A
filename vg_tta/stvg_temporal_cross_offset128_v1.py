"""Frozen two-stream temporal decoder: only self-attention key/value access changes."""
import math
import numpy as np
import torch


def binding(content,position,wrong):
    if wrong:content=torch.roll(content,len(content)//2,0)
    return content+position,content


@torch.no_grad()
def decode(decoder,kws,cross=True,wrong=False,capture=False):
    assert len(kws)==2 and all(not m.training for m in decoder.modules())
    if not cross:
        return [decoder(**kw)for kw in kws],[]
    states=[kw['query_tgt']for kw in kws];inter=[[],[]];trace=[]
    for level,layer in enumerate(decoder.layers):
        old=list(states);sa=[]
        for s in (0,1):
            other=1-s;own=old[s]+kws[s]['query_time'];otherk,otherv=binding(old[other],kws[other]['query_time'],wrong)
            key=torch.cat([own,otherk],0);value=torch.cat([old[s],otherv],0);mask=torch.cat([kws[s]['query_mask'],kws[other]['query_mask']],1)
            out,weights=layer.self_attn(own,key,value=value,key_padding_mask=mask)
            n=len(old[s]);mass=weights[...,n:].sum(-1)
            item=dict(layer=level,offset=s,cross_mass_mean=float(mass.mean()),cross_mass_min=float(mass.min()),cross_mass_max=float(mass.max()),own_length=n,other_length=len(old[other]),wrong_binding=wrong,other_content_temporal_variation=float((old[other]-old[other][:1]).abs().max()))
            if capture:
                item['audit']={k:v.detach().cpu().clone()for k,v in dict(q=own,k=key,v=value,mask=mask,out=out,weights=weights,in_weight=layer.self_attn.in_proj_weight,in_bias=layer.self_attn.in_proj_bias,out_weight=layer.self_attn.out_proj.weight,out_bias=layer.self_attn.out_proj.bias).items()};item['heads']=layer.self_attn.num_heads
            trace.append(item);sa.append(out)
        # Both SA reads above use pre-layer states. CA/FFN retain original stream shapes.
        for s in (0,1):
            kw=kws[s];x=layer.norm1(old[s]+layer.dropout1(sa[s]))
            y,_=layer.cross_attn_image(query=x.permute(1,0,2),key=kw['encoded_feature']+kw['encoded_pos'],value=kw['encoded_feature'],key_padding_mask=kw['encoded_mask'])
            x=layer.norm3(x+layer.dropout3(y.transpose(0,1)))
            y=layer.linear2(layer.dropout(layer.activation(layer.linear1(x))));states[s]=layer.norm4(x+layer.dropout4(y));inter[s].append(decoder.norm(states[s]))
    return [torch.stack(v).transpose(1,2)for v in inter],trace


def audit_attention(item):
    """Independent FP64 projection, head softmax and output on actual saved SA inputs."""
    z={k:v.numpy().astype(np.float64)for k,v in item['audit'].items()};q,k,v=z['q'][:,0],z['k'][:,0],z['v'][:,0];W=z['in_weight'];b=z['in_bias'];d=q.shape[-1];heads=item['heads'];dh=d//heads
    qp=(q@W[:d].T+b[:d]).reshape(len(q),heads,dh).transpose(1,0,2)
    kp=(k@W[d:2*d].T+b[d:2*d]).reshape(len(k),heads,dh).transpose(1,0,2)
    vp=(v@W[2*d:].T+b[2*d:]).reshape(len(v),heads,dh).transpose(1,0,2)
    score=np.einsum('hqd,hkd->hqk',qp,kp)/math.sqrt(dh);score[:,:,z['mask'][0].astype(bool)]=-np.inf;score-=np.max(score,axis=-1,keepdims=True);w=np.exp(score);w/=w.sum(-1,keepdims=True)
    context=np.einsum('hqk,hkd->hqd',w,vp).transpose(1,0,2).reshape(len(q),d);out=context@z['out_weight'].T+z['out_bias']
    errors={name:float(np.max(abs(expected-actual))/max(1.,float(np.max(abs(actual)))))for name,expected,actual in [('weights',w.mean(0),z['weights'][0]),('out',out,z['out'][:,0])]}
    assert max(errors.values())<2e-5,errors
    return errors


def properties():
    x=torch.arange(7*3,dtype=torch.float64).reshape(7,1,3);p=torch.sin(x*.17);key,value=binding(x,p,True);assert torch.equal(value,torch.roll(x,3,0))and torch.equal(key,value+p);assert not torch.equal(key,torch.roll(x+p,3,0))
    q=torch.tensor([[[.2,.4,.1]]],dtype=torch.float64);a=torch.softmax((q@(x+p).permute(1,2,0))/math.sqrt(3),-1)@x.transpose(0,1);b=torch.softmax((q@key.permute(1,2,0))/math.sqrt(3),-1)@value.transpose(0,1);assert not torch.allclose(a,b)
    return dict(status='pass',wrong_binding_rolls_content_and_value_but_not_position=True,not_trivial_joint_permutation=True,synthetic_attention_response_differs=True)


def decoder_properties():
    """Small CPU unit test of original class, no checkpoint/data/optimizer/GT."""
    from types import SimpleNamespace as NS
    from methods.decota_final_simplified_v1._tastvg_load import official_imports
    official_imports()
    from models.grounding_model.query_decoder import TimeDecoder
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4127)
        cfg=NS(MODEL=NS(TASTVG=NS(HIDDEN=8,HEADS=2,FFN_DIM=16,DROPOUT=.1)))
        d=TimeDecoder(cfg,3,return_intermediate=True,d_model=8).eval().requires_grad_(False)
        kws=[]
        for n in [5,4]:
            kws.append(dict(query_tgt=torch.randn(1,1,8).expand(n,1,8),query_content=torch.randn(n,1,8),query_time=torch.randn(n,1,8),query_mask=torch.zeros(1,n,dtype=torch.bool),encoded_feature=torch.randn(7,n,8),encoded_pos=torch.randn(7,n,8),encoded_mask=torch.zeros(n,7,dtype=torch.bool)))
        native=[d(**kw)for kw in kws];closed,_=decode(d,kws,cross=False);assert all(torch.equal(x,y)for x,y in zip(native,closed))
        joint={k:torch.cat([v[k]for v in kws],dim=1 if k in ['query_mask','encoded_feature','encoded_pos']else 0)for k in kws[0]}
        full=d(**joint);actual,trace=decode(d,kws,capture=True);want=[full[:,:,:5],full[:,:,5:]];err=max(float((x-y).abs().max())for x,y in zip(actual,want));assert err<2e-6,err
        audit=max(max(audit_attention(v).values())for v in trace);wrong,_=decode(d,kws,wrong=True);assert any(not torch.equal(x,y)for x,y in zip(actual,wrong));assert all(torch.equal(d(**kw),v)for kw,v in zip(kws,native))
    return dict(status='pass',closed_bit_exact=True,only_SA_joint_oracle_max_error=err,independent_attention_scaled_error=audit,wrong_binding_changes_response=True,native_after_exact=True,no_checkpoint_or_data_or_GPU=True)
