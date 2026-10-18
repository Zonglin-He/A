"""D-FINE-inspired two-stage E-relative boundary distributions, not full D-FINE."""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def edges(b):return torch.cat((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
def boxes(x):return torch.cat(((x[...,:2]+x[...,2:])/2,(x[...,2:]-x[...,:2]).clamp_min(1e-5)),-1)
def bins_for(radius):
    # Nonuniform signed construction following the official weighting-function
    # principle, expressed directly in E-relative edge-offset coordinates.
    u=radius/2;step=(u+1)**(2/30)
    return torch.tensor([-radius]+[-step**i+1 for i in range(15,0,-1)]+[0.]+[step**i-1 for i in range(1,16)]+[radius],dtype=torch.float32)

def roi(maps,valid,hw,b,size=7,context=1.5):
    B,C,H,W=maps.shape;t=(torch.arange(size,device=b.device,dtype=b.dtype)+.5)/size-.5
    x=b[:,0,None]+context*b[:,2,None]*t;y=b[:,1,None]+context*b[:,3,None]*t
    gx=(2*x*hw[:,1,None]/W-1)[:,None,:].expand(B,size,size);gy=(2*y*hw[:,0,None]/H-1)[:,:,None].expand(B,size,size)
    grid=torch.stack((gx,gy),-1);return F.grid_sample(maps*valid[:,None],grid,align_corners=False,padding_mode='zeros')

class Refiner(nn.Module):
    def __init__(self,radius):
        super().__init__();self.register_buffer('bins',bins_for(radius));prior=torch.full((4,33),-4.);prior[:,16]=0;self.register_buffer('initial_logits',prior)
        self.encoder=nn.Sequential(nn.Conv2d(256,32,1),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU(),nn.Flatten(),nn.Linear(32*49,128),nn.LayerNorm(128),nn.SiLU())
        self.heads=nn.ModuleList([nn.Sequential(nn.Linear(265,128),nn.SiLU(),nn.Linear(128,132))for _ in range(2)])
        for h in self.heads:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
    def decode(self,q,E):
        offset=(q*self.bins).sum(-1);xx=edges(E)+offset*torch.cat((E[:,2:],E[:,2:]),-1);ok=(xx[:,2:]>xx[:,:2]+1e-5).all(-1)&torch.isfinite(xx).all(-1)
        out=boxes(xx);return torch.where(ok[:,None],out,E),ok
    def forward(self,data,current):
        E=data['E'];ap=data['anchor_patches'];mask=data['anchor_mask'];B=len(E);ae=self.encoder(ap.reshape(-1,256,7,7)).reshape(B,4,128);z=self.initial_logits[None].expand(B,-1,-1);pos=current;out=[]
        for j in range(2):
            patch=roi(data['maps'],data['valid'],data['hw'],pos);local=self.encoder(patch);score=(local[:,None]*ae).sum(-1)/math.sqrt(128);safe=mask.clone();safe[~safe.any(-1),0]=True;score=score.masked_fill(~safe,-1e4);weight=score.softmax(-1)*mask;ctx=(weight[...,None]*ae).sum(1);relative=(edges(pos)-edges(E))/torch.cat((E[:,2:],E[:,2:]),-1).clamp_min(1e-5)
            inp=torch.cat((local,ctx,relative,E,mask.sum(-1,keepdim=True).float()/4),-1);z=z+self.heads[j](inp).reshape(B,4,33);q=z.softmax(-1);refined,ok=self.decode(q,E);out.append(dict(logits=z,prob=q,boxes=refined,valid=ok,roi_boxes=pos));pos=refined
        return out

def distribution_loss(stages,E,target,bins):
    from methods.decota_final_simplified_v1.objectives import generalized_iou
    y=(edges(target)-edges(E))/torch.cat((E[:,2:],E[:,2:]),-1).clamp_min(1e-5);clipped=y.clamp(float(bins[0]),float(bins[-1]));right=torch.searchsorted(bins,clipped.contiguous()).clamp(1,32);left=right-1;wr=(clipped-bins[left])/(bins[right]-bins[left]);losses=[]
    for s in stages:
        logp=s['logits'].log_softmax(-1);ce=-((1-wr)*logp.gather(-1,left[...,None]).squeeze(-1)+wr*logp.gather(-1,right[...,None]).squeeze(-1)).mean(-1)
        reg=5*(s['boxes']-target).abs().sum(-1)+2*(1-generalized_iou(s['boxes'],target));losses.append((ce+reg).mean())
    return sum(losses)/2,dict(outside_bins=float((y!=clipped).float().mean().detach()),invalid_stage_fraction=float(torch.stack([~s['valid']for s in stages]).float().mean().detach()))

def prepare_query(z,device='cpu'):
    maps=z['maps'].float();valid=z['valid'];N,C,H,W=maps.shape;E=z['E'].double();hw=torch.tensor([[H,W]]*N,dtype=torch.float32);anchor=[]
    for frame,k in zip(z['frames'],z['selected']):
        if k>=0:
            p=frame['position'];b=torch.tensor(np.asarray(frame['boxes'])[k],dtype=torch.float32)[None];anchor.append(roi(maps[p:p+1],valid[p:p+1],hw[p:p+1],b,context=1.)[0])
    patches=torch.zeros(4,256,7,7);mask=torch.zeros(4,dtype=torch.bool)
    for i,patch in enumerate(anchor):patches[i]=patch;mask[i]=True
    return dict(maps=maps,valid=valid,hw=hw,E=E,anchor_patches=patches,anchor_mask=mask,active=bool(anchor))

def pack(queries,positions,device):
    H=max(q['maps'].shape[-2]for q in queries);W=max(q['maps'].shape[-1]for q in queries);ms=[];vs=[]
    for q,i in zip(queries,positions):
        m=q['maps'][i];v=q['valid'][i];ms.append(F.pad(m,(0,W-m.shape[-1],0,H-m.shape[-2])));vs.append(F.pad(v,(0,W-v.shape[-1],0,H-v.shape[-2])))
    data=dict(maps=torch.stack(ms),valid=torch.stack(vs),hw=torch.stack([q['hw'][i]for q,i in zip(queries,positions)]),E=torch.stack([q['E'][i]for q,i in zip(queries,positions)]),anchor_patches=torch.stack([q['anchor_patches']for q in queries]),anchor_mask=torch.stack([q['anchor_mask']for q in queries]))
    return {k:(v.float()if k=='E'else v).to(device)for k,v in data.items()}

def crn(alias,n,samples=32):
    seed=int(hashlib.sha256(('dfine-crn-v1|'+alias).encode()).hexdigest()[:15],16);g=torch.Generator().manual_seed(seed);return torch.rand(n,4,samples,generator=g).clamp(1e-7,1-1e-7)

def sample_teacher(prob,E,bins,u):
    cdf=prob.cumsum(-1).contiguous();cdf[...,-1]=1;idx=torch.searchsorted(cdf,u.contiguous()).clamp_max(32);offset=bins[idx].permute(0,2,1);xx=edges(E)[:,None]+offset*torch.cat((E[:,2:],E[:,2:]),-1)[:,None];ok=(xx[...,2:]>xx[...,:2]+1e-5).all(-1)&torch.isfinite(xx).all(-1);b=boxes(xx)
    return dict(boxes=b,valid=ok,invalid_fraction=float((~ok).float().mean()),outside_image_fraction=float(((xx<0)|(xx>1)).any(-1).float().mean()))

def expected_reward(candidates,teacher):
    a=edges(candidates)[:,:,None];b=edges(teacher['boxes'])[:,None];inter=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp_min(0).prod(-1);union=(a[...,2:]-a[...,:2]).clamp_min(0).prod(-1)+(b[...,2:]-b[...,:2]).clamp_min(0).prod(-1)-inter;v=torch.where(union>0,inter/union.clamp_min(1e-10),torch.zeros_like(inter));m=teacher['valid'][:,None];return (v*m).sum(-1)/m.sum(-1).clamp_min(1)

@torch.no_grad()
def query_teacher(model,z,current,u,device='cuda'):
    n=len(z['E']);d=pack([z]*n,list(range(n)),device);st=model(d,current.float().to(device));s=sample_teacher(st[-1]['prob'],d['E'],model.bins,u.to(device));return dict(**{k:v.cpu()if torch.is_tensor(v)else v for k,v in s.items()},prob=st[-1]['prob'].cpu(),stage_boxes=[v['boxes'].cpu()for v in st],stage_roi_boxes=[v['roi_boxes'].cpu()for v in st],stage_invalid=[int((~v['valid']).sum())for v in st])

@torch.no_grad()
def outputs(model,z,alias,eps,cfg,planned,device='cuda'):
    from vg_tta.stvg_discriminative_dense32_v1 import candidates,student_logits
    from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
    E=z['E'].double();n=len(E);zero=torch.zeros(n,2,dtype=torch.float64);allowed=torch.ones(n,dtype=torch.bool);allowed[planned]=False;u=crn(alias,n,cfg['teacher_samples']);q0=query_teacher(model,z,E,u,device);basevalid=z['valid'];active=z['active'];paths={};results={'E_MBR':E.numpy()};calls=1
    direct=E.clone();teacher=q0;directpath=[]
    for j in range(2):
        if j:teacher=query_teacher(model,z,direct,u,device);calls+=1
        center=(direct[:,:2]-E[:,:2])/(cfg['center_scale']*E[:,2:]);cand=candidates(E,center,eps[j],cfg['center_scale']);reward=expected_reward(cand,teacher);valid=(cell_overlap(cand,*basevalid.shape[-2:])*basevalid[:,None]).sum((-1,-2))>1e-12;valid[:,0]=True;reward[~valid]=-torch.inf;best=reward.argmax(-1);new=cand[torch.arange(n),best];new[~allowed]=E[~allowed]
        if not active:new=E.clone()
        directpath.append(dict(boxes=new.clone(),teacher=teacher,rewards=reward,best=best));direct=new
    results['Direct_refine2']=direct.numpy();paths['Direct_refine2']=directpath
    for arm in ['Fixed_distill','Resample_fixed_teacher','Requery_OPD']:
        eta=zero.clone();path=[]
        for j in range(2):
            center=zero if arm=='Fixed_distill'else eta;teacher=q0
            if j and arm=='Requery_OPD':teacher=query_teacher(model,z,candidates(E,eta,torch.zeros(n,1,2),cfg['center_scale'])[:,0],u,device);calls+=1
            cand=candidates(E,center,eps[j],cfg['center_scale']);reward=expected_reward(cand,teacher).double();visible=(cell_overlap(cand,*basevalid.shape[-2:])*basevalid[:,None]).sum((-1,-2))>1e-12;safe=visible.clone();safe[~safe.any(-1)]=True;prior=-.5*eps[j].square().sum(-1);prior[~safe]=-torch.inf;pt=(prior+reward/cfg['temperature']).softmax(-1);d=center[:,None]+eps[j];sp=student_logits(d,eta,visible).softmax(-1);g=((sp-pt)[...,None]*d).sum(1);use=allowed&visible.any(-1)&(reward.masked_fill(~visible,-torch.inf).amax(-1)-reward.masked_fill(~visible,torch.inf).amin(-1)>1e-12)&teacher['valid'].any(-1)
            if not active:use[:]=False
            g[~use]=0;new=eta-cfg['step_size']*g;bb=candidates(E,new,torch.zeros(n,1,2),cfg['center_scale'])[:,0];bb[~allowed]=E[~allowed];path.append(dict(eta_before=eta.clone(),eta_after=new.clone(),sampling_center=center.clone(),candidates=cand,teacher=teacher,rewards=reward,prior=prior,prob=pt,student_prob=sp,gradient=g,boxes=bb,used=use));eta=new
        results[arm]=path[-1]['boxes'].numpy();paths[arm]=path
    assert torch.equal(paths['Fixed_distill'][0]['boxes'],paths['Resample_fixed_teacher'][0]['boxes'])and torch.equal(paths['Fixed_distill'][0]['boxes'],paths['Requery_OPD'][0]['boxes'])
    # Source/target E is fp64. Preserve its exact bits in all fixed components.
    return dict(outputs=results,paths=paths,actual_shared_teacher_calls=calls,logical_teacher_calls=dict(Direct_refine2=2,Fixed_distill=1,Resample_fixed_teacher=1,Requery_OPD=2),u=u,active=active)
