"""Ray-canonical completion and exact upright IoU for source-only simulation."""
from dataclasses import dataclass
from math import pi

import torch
from torch import nn


def axial(yaw):
    return (yaw + pi / 2).remainder(pi) - pi / 2


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1.e-9)


def yaw_frame(yaw):
    c, s = yaw.cos(), yaw.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*c.shape, 3, 3)


def ray_frame(view):
    forward = unit(view)
    up = torch.zeros_like(forward)
    up[:, 2] = 1
    right = torch.linalg.cross(up, forward)
    fallback = torch.zeros_like(right)
    fallback[:, 0] = 1
    right = unit(torch.where(right.norm(dim=-1, keepdim=True) > 1.e-8, right, fallback))
    above = unit(torch.linalg.cross(forward, right))
    return torch.stack((right, above, forward), -1)


def slab(center, span, axes, eye, rays):
    start = ((eye - center)[:, None, :] @ axes)
    direction = rays @ axes
    half = span[:, None, :] / 2
    parallel = direction.abs() < 1.e-8
    safe = torch.where(parallel, 1., direction)
    a, b = (-half - start) / safe, (half - start) / safe
    near = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(-1)
    far = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(-1)
    hit = (near > 0) & (near <= far) & ~(parallel & (start.abs() > half)).any(-1)
    return torch.where(hit, near, 0.), hit


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def exact_iou(first, second):
    """Batch-paired 3D IoU via intersection of two convex upright rectangles."""
    signs = first.new_tensor([[-1,-1],[1,-1],[1,1],[-1,1]])
    def vertices(box):
        frame = yaw_frame(box[:,6])[:, :2, :2]
        return (signs * box[:,None,3:5]/2) @ frame.transpose(-1,-2) + box[:, None, :2], frame
    a, ar = vertices(first)
    b, br = vertices(second)
    inside_a = (((a-second[:,None,:2])@br).abs() <= second[:,None,3:5]/2+1.e-7).all(-1)
    inside_b = (((b-first[:,None,:2])@ar).abs() <= first[:,None,3:5]/2+1.e-7).all(-1)
    u = (a.roll(-1,1)-a)[:,:,None]
    v = (b.roll(-1,1)-b)[:,None,:]
    ab = b[:,None] - a[:,:,None]
    det = cross(u,v)
    valid = det.abs()>1.e-9
    d = torch.where(valid,det,1.)
    t = cross(ab,v)/d
    s = cross(ab,u)/d
    good = valid & (t>=0) & (t<=1) & (s>=0) & (s<=1)
    points = torch.cat((a,b,(a[:,:,None]+u*t[...,None]).flatten(1,2)),1)
    mask = torch.cat((inside_a,inside_b,good.flatten(1)),1)
    n = mask.sum(1)
    mean = torch.where(mask[...,None],points,0.).sum(1)/n.clamp(min=1)[:,None]
    p = points-mean[:,None]
    order = torch.where(mask,torch.atan2(p[...,1],p[...,0]),10.).argsort(1)
    pts = p.gather(1,order[...,None].expand(-1,-1,2))
    ids = torch.arange(pts.shape[1],device=pts.device)[None,:]
    next_ids = (ids+1).remainder(n[:,None].clamp(min=1))
    paired = pts.gather(1,next_ids[...,None].expand(-1,-1,2))
    area = (torch.where(ids<n[:,None],cross(pts,paired),0.).sum(1)/2).abs()
    za,zb=first[:,2],second[:,2]
    ha,hb=first[:,5]/2,second[:,5]/2
    height=(torch.minimum(za+ha,zb+hb)-torch.maximum(za-ha,zb-hb)).clamp(min=0)
    intersect = area*height
    union = first[:,3:6].prod(1)+second[:,3:6].prod(1)-intersect
    return (intersect/union.clamp(min=1.e-9)).clamp(0,1)


@dataclass
class Scene:
    points: torch.Tensor  # Coordinates relative to the observed support centroid.
    directions: torch.Tensor
    mask: torch.Tensor
    condition: torch.Tensor
    origin: torch.Tensor
    frame: torch.Tensor
    prior: torch.Tensor
    scale: torch.Tensor
    yaw0: torch.Tensor
    eye: torch.Tensor

    def repeat(self, count):
        return Scene(**{k:v.repeat_interleave(count,0) for k,v in vars(self).items()})

    def encode(self, box):
        center = ((box[:,:3]-self.origin)[:,None,:]@self.frame)[:,0]/self.scale
        return torch.cat((center,(box[:,3:6]/self.prior).clamp(min=1.e-5).log(),
                          (axial(box[:,6]-self.yaw0)/pi)[:,None]),1)

    def decode(self, x):
        center=((x[:,:3].clamp(-4,4)*self.scale)[:,None]@self.frame.transpose(-1,-2))[:,0]+self.origin
        return torch.cat((center,self.prior*x[:,3:6].clamp(-1.1,1.1).exp(),
                          axial(self.yaw0+x[:,6]*pi)[:,None]),1)


def scene_for(points, directions, mask, eye, prior, tangents, seed):
    count = mask.sum(1,keepdim=True)
    location = torch.where(mask[...,None],points,0.).sum(1)/count.clamp(min=1)
    location = torch.where(count>0,location,seed)
    direction = unit(location-eye)
    frame = ray_frame(direction)
    scale = prior.norm(dim=-1,keepdim=True).clamp(min=.1)
    local = (points-location[:,None])@frame/scale[:,None]
    rays = unit(directions)@frame
    condition = torch.cat((prior.log(),tangents,frame[:,2],count.float().log1p()/5),-1)
    return Scene(local,rays,mask,condition,location,frame,prior,scale,
                 torch.atan2(direction[:,1],direction[:,0]),((eye-location)[:,None,:]@frame)[:,0]/scale)


class Flow(nn.Module):
    def __init__(self):
        super().__init__()
        self.points = nn.Sequential(nn.Linear(6,64),nn.SiLU(),nn.Linear(64,96),nn.SiLU())
        self.velocity = nn.Sequential(nn.Linear(211,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),
                                      nn.Linear(192,128),nn.SiLU(),nn.Linear(128,7))

    def embed(self, scene):
        h=self.points(torch.cat((scene.points,scene.directions),-1))
        mask=scene.mask[...,None]
        mean=torch.where(mask,h,0.).sum(1)/mask.sum(1).clamp(min=1)
        maximum=torch.where(mask,h,-1.e4).amax(1)
        maximum=torch.where(mask.any(1),maximum,0.)
        return torch.cat((mean,maximum,scene.condition),-1)

    def forward(self, x, time, embedding):
        return self.velocity(torch.cat((x,time,embedding),-1))


def critic_features(scene, box):
    # Signed offsets to candidate faces, normalized ray residuals, span ratios,
    # and relative orientation. The camera distance is used by the slab only.
    pr = scene.encode(box)
    center = pr[:,:3]
    axes = scene.frame.transpose(-1,-2) @ yaw_frame(box[:,6])
    size = box[:,3:6]/scene.scale
    local = ((scene.points-center[:,None])@axes)/size[:,None,:]
    signed = torch.cat((.5-local,.5+local),-1)
    mask=scene.mask
    count=mask.sum(1).clamp(min=1)
    def avg(x):
        return torch.where(mask[...,None],x,0.).sum(1)/count[:,None]
    def fractions(signal, cuts):
        return torch.cat([avg((signal<cut).float()) for cut in cuts],-1)
    hist = fractions(signed,(-.15,0.,.08,.25,.5))
    surface = signed.abs().amin(-1)
    outside = (-signed).clamp(min=0.).amax(-1)
    first,hit=slab(center,size,axes,scene.eye,scene.directions)
    actual=((scene.points-scene.eye[:,None])*scene.directions).sum(-1)
    delta=torch.where(hit,(actual-first).abs(),1.)
    contact = fractions(torch.stack((surface,outside,delta),-1),(.03,.07,.15,.3))
    relative = pr[:,3:6].clamp(-2,2)
    facing = torch.stack((torch.cos(2*pi*pr[:,6]),torch.sin(2*pi*pr[:,6])),1)
    return torch.cat((hist,contact,relative,facing),-1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(48,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self, geometry, silhouette):
        return self.net(torch.cat((geometry,silhouette.reshape(-1,1)),1))[:,0].sigmoid()
