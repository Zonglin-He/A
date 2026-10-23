"""One fixed lighter spatial proposal; no cycle, mask-transport links or tuning."""
import torch
import torch.nn.functional as F
from stvg.spatial import SpatialGeometry
from stvg._spatial_math import affinity,fit_axis_affine,transform_native_box,occupancy,extent_kernel,EPS

def known_grid(spatial,frame_ids,height,width):
    """Explicit native grid shape replaces position-grid factor search.

    In a deployment adapter H,W must come from the original encoder's actual
    projected grid, never inferred from GT. This study uses exact cached geometry
    dimensions and checks every output against original grid extraction.
    """
    n=height*width;T=len(frame_ids);features=[];valid=[]
    for t in range(T):
        view=spatial[t%2];j=t//2;f=view['encoded_feature'][:n,j].float();v=~view['encoded_mask'][j,:n]
        assert f.shape==(n,256) and v.any() and torch.isfinite(f[v]).all()
        features.append(torch.where(v[:,None],f,torch.zeros_like(f)));valid.append(v)
    return torch.stack(features),torch.stack(valid)

def box_links(boxes):
    lo=boxes[...,:2]-boxes[...,2:]/2;hi=boxes[...,:2]+boxes[...,2:]/2
    a0,a1=lo[:-1,:,None],hi[:-1,:,None];b0,b1=lo[1:,None],hi[1:,None]
    inter=(torch.minimum(a1,b1)-torch.maximum(a0,b0)).clamp_min(0).prod(-1)
    area=boxes[...,2:].prod(-1);return inter/(area[:-1,:,None]+area[1:,None]-inter).clamp_min(EPS)

def build(features,valid,ids,positions,seeds,E,native_boxes,*,nearest):
    """Exact original two-way affine fit, one or two physical nearest anchors.

    Retain all affine validity tests and fixed score coefficients. Uniform
    available-side support, complete-box extent+appearance score and E guard.
    Adjacent box IoU replaces feature/mask-transport link. No original geometry
    or candidate generation is computed and discarded inside this implementation.
    """
    raw=features.float();cells=valid.bool();E=torch.as_tensor(E,dtype=torch.float64);native=torch.as_tensor(native_boxes,dtype=torch.float64);ids=torch.as_tensor(ids,dtype=torch.long)
    seeds=seeds.double()*cells[positions].reshape_as(seeds);T,N,D=raw.shape;H,W=seeds.shape[-2:]
    g=SpatialGeometry.__new__(SpatialGeometry);g.E=E;g.frames=T;g.frame_ids=ids;g.positions=list(positions);g.positions8=g.positions;g.height=H;g.width=W
    g.native_boxes=native;g.observed=torch.zeros(T,dtype=torch.bool);g.observed[positions]=True;g.source_anchors=torch.full((T,2),-1,dtype=torch.long);g.side_available=torch.zeros(T,2,dtype=torch.bool)
    g.side_reference_boxes=E[:,None].expand(-1,2,-1).clone();g.side_prototypes=torch.zeros(T,2,D,dtype=torch.float64)
    pooled=(raw[positions].double()*seeds.flatten(1)[...,None]).sum(1)/seeds.flatten(1).sum(1).clamp_min(EPS)[:,None];g.seed_features=F.normalize(pooled,dim=-1,eps=1e-12);g.seed_valid=(seeds.flatten(1).sum(-1)>EPS)&(g.seed_features.norm(dim=-1)>EPS)
    normalized=F.normalize(raw,dim=-1,eps=1e-12);yy,xx=torch.meshgrid((torch.arange(H,dtype=torch.float64)+.5)/H,(torch.arange(W,dtype=torch.float64)+.5)/W,indexing='ij');xy=torch.stack((xx,yy),-1).reshape(N,2)
    order=sorted(range(8),key=lambda j:positions[j]);calls=0;fits=0
    for t in range(T):
        if g.observed[t]:neighbors=[positions.index(t),positions.index(t)]
        elif nearest:neighbors=[min(order,key=lambda j:(abs(int(ids[t])-int(ids[positions[j]])),positions[j]))]
        else:
            past=[j for j in order if positions[j]<t];future=[j for j in order if positions[j]>t];neighbors=[past[-1]if past else None,future[0]if future else None]
        for side,j in enumerate(neighbors):
            if j is None:continue
            a=positions[j];g.source_anchors[t,side]=j;g.side_prototypes[t,side]=g.seed_features[j]
            if a==t:g.side_available[t,side]=True;g.side_reference_boxes[t,side]=native[j];continue
            ta=affinity(normalized,cells,t,a);at=affinity(normalized,cells,a,t);calls+=2
            fitted,info=fit_axis_affine(at*ta.T*seeds[j].flatten()[:,None],xy,xy);fits+=1
            if fitted is None:continue
            box=transform_native_box(native[j],*fitted)
            if box is not None:g.side_available[t,side]=True;g.side_reference_boxes[t,side]=box
    weights=g.side_available.double();total=weights.sum(-1);g.confidence=total/total.clamp_min(1);g.reliable=g.side_available.any(-1)
    g.boxes=E[:,None].expand(-1,3,-1).clone();g.valid=torch.zeros(T,3,dtype=torch.bool);g.valid[:,0]=True;g.candidate_origin=torch.full((T,3),-1,dtype=torch.long);g.candidate_quality=torch.zeros(T,3,dtype=torch.float64)
    for t in range(T):
        for side in (0,1):
            if g.side_available[t,side]:
                g.boxes[t,side+1]=g.side_reference_boxes[t,side];g.valid[t,side+1]=True;g.candidate_origin[t,side+1]=-2 if g.observed[t]else side;g.candidate_quality[t,side+1]=1.
        if g.valid[t,1]and g.valid[t,2]and torch.equal(g.boxes[t,1],g.boxes[t,2]):g.valid[t,2]=False
    occ,_,_=occupancy(g.boxes,H,W);occ*=cells.reshape(T,1,H,W);f=torch.bmm(occ.flatten(-2),raw.double())/occ.flatten(-2).sum(-1).clamp_min(EPS)[...,None];g.features=F.normalize(f,dim=-1,eps=1e-12)
    g.extent_scores=extent_kernel(g.boxes[:,:,None],g.side_reference_boxes[:,None]);g.association_scores=torch.einsum('tkd,tsd->tks',g.features,g.side_prototypes).clamp(0,1)
    g.projection=((.5*g.extent_scores+.5*g.association_scores)*weights[:,None]).sum(-1)/total.clamp_min(EPS)[:,None];g.u=g.projection
    dz=torch.cat(((g.boxes[...,:2]-E[:,None,:2])/E[:,None,2:],g.boxes[...,2:].log()-E[:,None,2:].log()),-1);g.guard=-torch.expm1(-(torch.linalg.vector_norm(dz,dim=-1)/2).square())
    g.link=box_links(g.boxes);g.metadata=dict(schema='light_spatial_research_v1',GT_read=False,nearest_anchor=nearest,cycle=False,link='unclipped complete-box adjacent IoU',affinity_calls=calls,affine_fits=fits)
    return g
