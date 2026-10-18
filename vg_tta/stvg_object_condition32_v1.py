"""Rebuild frozen native initial queries from region-supported fused features.

No optimizer, new learned mapping, memory deletion, or temporal foreground labels.
"""
import torch
from methods.decota_final_simplified_v1.tensors import detached
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
from vg_tta.stvg_dino_query_closedloop32_v1 import corrected
ARMS=['E_MBR','E_region_spatial','DINO_region_spatial','Object_temporal','Spatial_then_temporal']


def region_weights(att, boxes, valid, h, w):
    """Fractional cells in original normalized frame coordinates, padded cells masked.

    The resized image occupies the valid rectangle; original aspect ratio may
    change during the already fixed preprocessing. Boxes stay normalized to it.
    Each nonempty region preserves that frame's native valid-cell attention mass.
    A zero-overlap region leaves that frame's native weights unchanged.
    """
    assert att.shape==valid.shape and att.shape[1]==h*w
    outputs=[];coverage=[];fallback=[]
    for a,b,v in zip(att,boxes,valid):
        vv=v.reshape(h,w);vh=int(vv.any(1).sum());vw=int(vv.any(0).sum())
        assert vh>0 and vw>0 and torch.equal(vv[:vh,:vw],torch.ones_like(vv[:vh,:vw]))
        c=torch.zeros(h,w,device=a.device,dtype=torch.float64)
        c[:vh,:vw]=cell_overlap(b[None],vh,vw)[0]
        c=c.flatten().to(a);original=a*v;raw=original*c;empty=bool(raw.sum()<=0)
        weighted=original if empty else raw*(original.sum()/raw.sum())
        outputs.append(weighted);coverage.append(c);fallback.append(empty)
    return torch.stack(outputs),torch.stack(coverage),fallback


def conditioned_query(H, att, boxes, valid, native_weights, native_q, h, w):
    weights,coverage,fallback=region_weights(att,boxes,valid,h,w)
    # Native query = mean(native features * native weights). Preserve exactly
    # the total coefficient mass despite a different number of support frames.
    mass=native_weights.mean();den=weights.sum()
    if not bool(den>0) or not bool(native_q.norm()>0):
        return native_q,dict(noop=True,reason='zero_weight_or_native_norm')
    coefficients=weights*(mass/den)
    raw=(H*coefficients[...,None]).sum((0,1))
    if not bool(raw.norm()>0):return native_q,dict(noop=True,reason='zero_region_query')
    q=raw*(native_q.norm()/raw.norm())
    assert torch.isfinite(q).all()
    assert torch.allclose(q.norm(),native_q.norm(),rtol=2e-6,atol=2e-7)
    assert torch.allclose(coefficients.sum(),mass,rtol=2e-6,atol=2e-7)
    return q,dict(noop=False,weights=weights,coefficients=coefficients,coverage=coverage,
        fallback=fallback,native_mass=mass,conditioned_mass=coefficients.sum(),
        native_norm=native_q.norm(),raw_norm=raw.norm(),conditioned_norm=q.norm(),
        delta_norm=(q-native_q).norm(),native_query=native_q,conditioned_query=q)


@torch.no_grad()
def readout(base, E, source, positions, dino, spatial='none', motion=None):
    assert spatial in ['none','E','DINO'];calls=[0];traces=[]
    def insert(module,args,kw):
        k=calls[0];calls[0]+=1
        if k%2==0:return args,kw  # Exact native first pass and actionness routing.
        view=k//2;info=kw['encoded_info'];H=info['encoded_feature'];h,w=info['fea_map_size'];l=h*w;n=H.shape[1]
        chosen=base.zero['gates'][view]['second'];ft=H[l:-l].mean(1).unsqueeze(0).detach()
        fm=H[-l:].permute(1,2,0).reshape(n,256,h,w);fa=H[:l].permute(1,2,0).reshape(n,256,h,w)
        _,native_s=base.model.s_spatial_clas(fa[chosen],ft[:,:1]);_,native_t=base.model.t_spatial_clas(fm[chosen],ft[:,:1])
        ns=(H[:l].permute(1,0,2)[chosen]*native_s[...,None]).mean((0,1))
        nt=(H[-l:].permute(1,0,2)[chosen]*native_t[...,None]).mean((0,1))
        assert torch.equal(ns,kw['isq']) and torch.equal(nt,kw['itq'])
        kw=dict(kw);tr=dict(view=view,native_chosen=list(chosen),spatial=None,temporal=None)
        local=[j for j,i in enumerate(positions)if i%2==view];indices=[positions[j]//2 for j in local]
        valid_s=~info['encoded_mask'][:,:l];valid_t=~info['encoded_mask'][:,-l:]
        if spatial!='none'and indices:
            _,a=base.model.s_spatial_clas(fa[indices],ft[:,:1])
            b=E[positions][local]if spatial=='E'else dino[local]
            kw['isq'],tr['spatial']=conditioned_query(H[:l].permute(1,0,2)[indices],a,b.to(H),valid_s[indices],native_s,ns,h,w)
            tr['spatial']['positions']=indices
        if motion is not None and positions:
            # Every actual frame is supported; no current interval or actionness
            # mask is used to remove evidence from object-conditioned motion.
            _,a=base.model.t_spatial_clas(fm,ft[:,:1])
            kw['itq'],tr['temporal']=conditioned_query(H[-l:].permute(1,0,2),a,motion[view::2].to(H),valid_t,native_t,nt,h,w)
            tr['temporal']['positions']=list(range(n))
        traces.append(detached(tr,'cpu'));return args,kw
    hook=base.decoder.register_forward_pre_hook(insert,with_kwargs=True)
    try:v=base.reference_values(capture=False)
    finally:hook.remove()
    assert calls[0]==4 and v['gates']==base.zero['gates']
    b=corrected(v['boxes'],source.to(v['boxes']),E.to(v['boxes'].device))
    return dict(boxes=b.cpu(),native_boxes=v['boxes'].cpu(),logits=detached(v['logits'],'cpu'),traces=traces,gates=v['gates'])


def properties():
    torch.manual_seed(20261018)
    H=torch.randn(2,12,256);a=torch.rand(2,12);v=torch.ones_like(a,dtype=torch.bool);b=torch.tensor([[.5,.5,1.,1.],[.5,.5,1.,1.]])
    q=(H*a[...,None]).mean((0,1));qq,t=conditioned_query(H,a,b,v,a,q,3,4)
    assert torch.allclose(q,qq,atol=1e-7,rtol=2e-6)
    b[0]=torch.tensor([.2,.3,.3,.4]);qq,t=conditioned_query(H,a,b,v,a,q,3,4)
    assert torch.allclose(t['weights'].sum(1),(a*v).sum(1),atol=1e-6)
    assert torch.allclose(q.norm(),qq.norm(),rtol=2e-6)
    s=torch.rand(5,4);s[:,2:]+=.1;e=s.double()+.02
    assert torch.equal(corrected(s,s,e),e)
    return dict(status='pass',full_region_identity=True,native_coefficient_mass_preserved=True,
        per_frame_attention_mass_preserved=True,query_norm_preserved=True,zero_E_retaining_exact=True)
