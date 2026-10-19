"""Registered frozen RGB-motion observations on the existing valid image grid.

Coordinates describe uniform normalized feature cells, not exact CNN receptive
fields. Registration cannot undo the video encoder's crop-coordinate mixing.
"""
import numpy as np
import torch
import torch.nn.functional as F
from methods.stvg_referdino_opd_v1._capture import transfer
from vg_tta.stvg_temporal_early_fusion128_v1 import decode_prefix


def geometry(rectangles,width,height,valid):
    """Per native cell: area fraction and intersection-centroid crop coordinate."""
    t,h,w=valid.shape;assert len(rectangles)==t and valid.dtype==torch.bool
    alpha=torch.zeros((t,h,w),dtype=torch.float64);grid=torch.zeros((t,h,w,2),dtype=torch.float64);cropgrid=[]
    for n,rr in enumerate(rectangles):
        vh=int(valid[n].any(1).sum());vw=int(valid[n].any(0).sum());assert valid[n,:vh,:vw].all()and int(valid[n].sum())==vh*vw
        l,b,r,u=np.asarray(rr,float)/np.array([width,height,width,height]);assert 0<=l<r<=1 and 0<=b<u<=1
        yy,xx=torch.meshgrid(torch.arange(vh,dtype=torch.float64),torch.arange(vw,dtype=torch.float64),indexing='ij');x0=torch.maximum(xx/vw,torch.tensor(l));x1=torch.minimum((xx+1)/vw,torch.tensor(r));y0=torch.maximum(yy/vh,torch.tensor(b));y1=torch.minimum((yy+1)/vh,torch.tensor(u));a=(x1-x0).clamp_min(0)*(y1-y0).clamp_min(0)*vh*vw
        alpha[n,:vh,:vw]=a;grid[n,:vh,:vw,0]=2*((x0+x1)/2-l)/(r-l)-1;grid[n,:vh,:vw,1]=2*((y0+y1)/2-b)/(u-b)-1;cropgrid.append([l,b,r,u,vh,vw])
    assert torch.isfinite(grid).all()and alpha.min()>=0 and alpha.max()<=1+1e-12
    return alpha.clamp(0,1).float(),grid.float(),cropgrid


def independent_geometry(rectangles,width,height,valid):
    v=valid.cpu().numpy();t,h,w=v.shape;alpha=np.zeros((t,h,w));grid=np.zeros((t,h,w,2))
    for n,rr in enumerate(rectangles):
        vh=int(v[n].any(1).sum());vw=int(v[n].any(0).sum());l,b,r,u=[float(rr[0])/width,float(rr[1])/height,float(rr[2])/width,float(rr[3])/height]
        for y in range(vh):
            for x in range(vw):
                a=max(x/vw,l);c=min((x+1)/vw,r);d=max(y/vh,b);e=min((y+1)/vh,u);alpha[n,y,x]=max(c-a,0)*max(e-d,0)*vh*vw;grid[n,y,x]=[2*((a+c)/2-l)/(r-l)-1,2*((d+e)/2-b)/(u-b)-1]
    return np.clip(alpha,0,1).astype('float32'),grid.astype('float32')


def feature_crop(native,cropgrid,crop_hw=(7,7)):
    t,c,h,w=native.shape;ch,cw=crop_hw;out=[]
    for n,(l,b,r,u,vh,vw) in enumerate(cropgrid):
        yy,xx=torch.meshgrid((torch.arange(ch,dtype=torch.float64)+.5)/ch,(torch.arange(cw,dtype=torch.float64)+.5)/cw,indexing='ij');g=torch.stack((2*(l+xx*(r-l))-1,2*(b+yy*(u-b))-1),-1).to(device=native.device,dtype=torch.float32)
        # Border belongs to the valid image, not the padded full tensor.
        out.append(F.grid_sample(native[n:n+1,:,:vh,:vw].float(),g[None],mode='bilinear',padding_mode='border',align_corners=False))
    return torch.cat(out).to(native.dtype)


def register(native,local,alpha,grid):
    assert native.shape[:2]==local.shape[:2]and native.shape[0]==len(alpha)
    a=alpha.to(native.device)[:,None];g=grid.to(native.device);patch=F.grid_sample(local.float(),g,mode='bilinear',padding_mode='border',align_corners=False);mix=((1-a)*native.float()+a*patch).to(native.dtype)
    out=torch.where(a==0,native,mix);assert torch.equal(out.expand_as(native)[(a==0).expand_as(native)],native[(a==0).expand_as(native)])and torch.isfinite(out).all();return out


@torch.no_grad()
def fuse_motion(model,captured,view,motion=None,progress=None):
    x=transfer(captured['src'],'cuda');kw=transfer(captured['kwargs'],'cuda');h,w=view['info']['fea_map_size'];l=h*w;old=x.clone()
    if motion is not None:
        assert motion.shape==(x.shape[1],x.shape[2],h,w);x[-l:]=motion.to(x).flatten(2).permute(2,0,1)
    assert torch.equal(x[:-l],old[:-l]);pref=[];actual=[]
    def entered(m,args,kwargs):
        assert torch.equal(args[0],x)and torch.equal(kwargs['src_key_padding_mask'],kw['src_key_padding_mask'])and torch.equal(kwargs['pos'],kw['pos']);actual.append(True)
    def emit(m,a):
        value=a[0].detach().cpu();pref.append(value)
        if progress is not None:progress['returned_prefix']=value
    hooks=[model.ground_encoder.encoder.register_forward_pre_hook(entered,with_kwargs=True),model.ground_encoder.encoder.norm.register_forward_pre_hook(emit)]
    try:
        with torch.autocast('cuda',dtype=torch.float16):model.ground_encoder.encoder(x,**kw)
    finally:
        for hook in hooks:hook.remove()
    assert len(pref)==len(actual)==1 and torch.isfinite(pref[0]).all();v=dict(view,prefix=pref[0],info=dict(view['info']));return v,dict(actual_input_sha256=tensor_sha(x),prefix_sha256=tensor_sha(pref[0]),appearance_text_input_exact=True,position_padding_exact=True,actual_encoder_entry_verified=True)


def tensor_sha(x):
    import hashlib
    x=x.detach().cpu().contiguous();return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def properties():
    torch.manual_seed(29128);native=torch.randn(3,4,4,6);valid=torch.ones(3,4,6,dtype=torch.bool);valid[1,:,5:]=False;valid[2,3:]=False;rect=[[0,0,100,80],[20,10,50,65],[95,70,100,80]];a,g,c=geometry(rect,100,80,valid);aa,gg=independent_geometry(rect,100,80,valid);assert np.array_equal(a.numpy(),aa)and np.array_equal(g.numpy(),gg)
    constant=register(native,torch.ones(3,4,7,7),a,g);assert torch.allclose(constant[0],torch.ones_like(constant[0]),atol=2e-6);local=torch.randn(3,4,7,7);out=register(native,local,a,g);assert torch.equal(out[(a==0)[:,None].expand_as(out)],native[(a==0)[:,None].expand_as(native)]);f=feature_crop(native,c);fw=register(native,f,a,g);assert not torch.equal(out,fw)
    # Padding values cannot contaminate valid image FeatureWarp evidence.
    altered=native.clone();altered[(~valid)[:,None].expand_as(native)]=1000.;fc0=feature_crop(native,c);fc1=feature_crop(altered,c);assert torch.equal(fc0,fc1);w0=register(native,fc0,a,g);w1=register(altered,fc1,a,g);assert torch.equal(w0[valid[:,None].expand_as(native)],w1[valid[:,None].expand_as(native)])
    # Same grid/whole image is a numerical identity (zero branch bypasses warp).
    v=torch.ones(1,4,6,dtype=torch.bool);a0,g0,c0=geometry([[0,0,100,80]],100,80,v);fc=feature_crop(native[:1],c0,(4,6));rr=register(native[:1],fc,a0,g0);err=float((rr-native[:1]).abs().max());assert err<2e-6
    # Independent scalar bilinear border sampler checks registration, including padding.
    ref=np.zeros_like(out.numpy());z=local.numpy();old=native.numpy();mx=0.
    for n in range(3):
      for y in range(4):
       for x in range(6):
        px=np.clip((float(g[n,y,x,0])+1)*7/2-.5,0,6);py=np.clip((float(g[n,y,x,1])+1)*7/2-.5,0,6);ix,iy=int(px),int(py);jx,jy=min(ix+1,6),min(iy+1,6);sx,sy=px-ix,py-iy;vv=(1-sy)*((1-sx)*z[n,:,iy,ix]+sx*z[n,:,iy,jx])+sy*((1-sx)*z[n,:,jy,ix]+sx*z[n,:,jy,jx]);ref[n,:,y,x]=(1-aa[n,y,x])*old[n,:,y,x]+aa[n,y,x]*vv
    mx=float(abs(ref-out.numpy()).max());assert mx<2e-6
    return dict(status='pass',independent_geometry_exact=True,independent_bilinear_max_error=mx,whole_grid_numeric_identity_max_error=err,padding_and_outside_exact=True,FeatureWarp_padding_change_invariant=True,FeatureWarp_not_assumed_Native=True,no_model_or_GT=True)
