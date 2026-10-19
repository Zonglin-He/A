"""Four independently supervised bounded edge distributions, frozen Fine ROI."""
import hashlib
import numpy as np
import torch
from vg_tta.stvg_roi_layout_opd32_v1 import build,feedback
from vg_tta.stvg_fine_roi_scoped_opd32_v1 import scope
ARMS=['E_MBR','Center_live','Edge_fixed','Edge_live','Edge_shuffle']
EDGES=['L','T','R','B']
GROUPS=torch.tensor([[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16]])

def noise(alias,n):
 from scipy.special import ndtri
 seed=int.from_bytes(hashlib.sha256(('fine-edge32-v1|'+alias).encode()).digest()[:8],'little');rng=np.random.default_rng(seed);out=[]
 for _ in range(2):
  u=(rng.integers(1,2**53,size=(n,4,2)).astype(np.float64)/2**53+np.arange(2)[None,None])/4
  out.append(torch.from_numpy(ndtri(np.concatenate([u,1-u[...,::-1]],-1))))
 return torch.stack(out)

def edgebox(E,a):
 span=E[:,[2,3,2,3]];d=.5*span[:,None]*torch.tanh(.5*a)
 b=E[:,None].expand(-1,a.shape[1],-1).clone();b[...,:2]+=(d[...,:2]+d[...,2:])/2;b[...,2:]+=d[...,2:]-d[...,:2]
 return b

def propose(E,center,eps):
 a=center[:,None].expand(-1,17,-1).clone()
 for e in range(4):a[:,1+4*e:5+4*e,e]+=eps[:,e]
 assert torch.isfinite(a).all(),'nonfinite latent proposal: engineering stop'
 b=edgebox(E,a);assert torch.isfinite(b).all(),'nonfinite physical proposal: engineering stop';valid=torch.isfinite(b).all(-1)&(b[...,2:]>0).all(-1);assert valid[:,0].all()
 return a,b,valid

def edgeprob(raw,eps,legal,active,tau):
 # Same score aggregation across anchors; only per-edge candidate support differs.
 rr=raw['rewards'][:,GROUPS];ok=raw['valid'][:,GROUPS]&legal[:,GROUPS];safe=ok.clone();safe[~safe.any(-1)]=True
 zz=torch.cat((torch.zeros(*eps.shape[:2],1,dtype=torch.float64),eps),-1);lp=(-.5*zz.square()).masked_fill(~safe,-torch.inf);prior=lp.softmax(-1);q=(lp+rr/tau).softmax(-1)
 info=ok.any(-1)&((rr.masked_fill(~ok,-torch.inf).amax(-1)-rr.masked_fill(~ok,torch.inf).amin(-1))>1e-6)&bool(active);q=torch.where(info[...,None],q,prior)
 return dict(prob=q,prior=prior,rewards=rr,valid=ok,informative=info,noise_with_zero=zz)

def fit(E,context,eps,cfg,allowed,mode,permutation):
 mu=torch.zeros(len(E),4,dtype=torch.float64);trace=[];path=[dict(state=mu.clone(),boxes=E.clone())]
 for st in range(2):
  center=torch.zeros_like(mu)if mode=='Edge_fixed'else mu.clone();a,b,legal=propose(E,center,eps[st]);f=feedback(context,b,torch.zeros(len(E),17,2,dtype=torch.float64),cfg['temperature'],permutation if mode=='Edge_shuffle'else None);ef=edgeprob(f,eps[st],legal,context['positions'],cfg['temperature']);actions=torch.stack([a[:,GROUPS[e],e]for e in range(4)],1);safe=ef['valid'].clone();safe[~safe.any(-1)]=True;lp=(-.5*(actions-mu[:,:,None]).square()).masked_fill(~safe,-torch.inf);sp=lp.softmax(-1);g=((sp-ef['prob'])*actions).sum(-1);g[~(allowed[:,None]&ef['informative'])]=0;new=mu-cfg['step_size']*g;assert torch.isfinite(new).all(),'nonfinite mean: engineering stop';out=edgebox(E,new[:,None])[:,0];assert torch.isfinite(out).all()and(out[:,2:]>0).all();out[~allowed]=E[~allowed]
  trace.append(dict(step=st,sampling_center=center,latent_actions=a,candidates=b,legal=legal,visual_feedback={k:v for k,v in f.items()if k not in ['prob','prior']},edge_feedback=ef,student_prob=sp,gradient=g,before=mu.clone(),after=new.clone(),output_step_L2=float((out-path[-1]['boxes']).norm())));mu=new;path.append(dict(state=mu.clone(),boxes=out))
 return dict(mode=mode,rounds=trace,path=path,final=path[-1]['boxes'],selected_step=2,GT_read=False)

def properties():
 E=torch.tensor([[.5,.5,.4,.3]]*4,dtype=torch.float64);ep=noise('edge-properties',4);assert ep.shape==(2,4,4,4)and(ep.sum(-1).abs()<1e-12).all();assert torch.equal(edgebox(E,torch.zeros(4,1,4))[:,0],E);assert torch.equal(edgebox(E,torch.zeros(4,1,4))[:,0,2:],E[:,2:]);a,b,legal=propose(E,torch.zeros(4,4),ep[0]);assert legal.all()
 for e in range(4):
  delta=a[:,GROUPS[e]];other=[j for j in range(4)if j!=e];assert not delta[:,:,other].any()
 maps=torch.ones(4,8,6,8);v=torch.ones(4,6,8,dtype=torch.bool);ctx=build(maps,v,[0],E[:1]);cfg=dict(temperature=.1,step_size=.5);allowed=torch.tensor([False,True,True,True])
 for mode in ['Edge_fixed','Edge_live','Edge_shuffle']:assert torch.equal(fit(E,ctx,ep,cfg,allowed,mode,[0,2,3,1])['final'],E)
 ctx=build(torch.randn_like(maps),v,[0],E[:1]);fx=fit(E,ctx,ep,cfg,allowed,'Edge_fixed',[0,2,3,1]);lv=fit(E,ctx,ep,cfg,allowed,'Edge_live',[0,2,3,1]);assert torch.equal(fx['path'][1]['boxes'],lv['path'][1]['boxes']);assert torch.equal(lv['rounds'][1]['sampling_center'],lv['path'][1]['state']);assert torch.equal(fx['rounds'][1]['sampling_center'],torch.zeros(4,4,dtype=torch.float64));assert not lv['path'][-1]['state'][~allowed].any()
 return dict(status='pass',zero_exact_E=True,legal_disjoint_edge_ranges=True,stratified_four_quartiles_antithetic=True,one_edge_changes_per_candidate=True,shared_zero=True,constant_noop=True,Fixed_Live_first_exact=True,predicted_scope=True)
