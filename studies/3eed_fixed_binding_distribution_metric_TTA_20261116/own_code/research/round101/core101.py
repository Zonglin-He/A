"""Exact categorical KL with its own Fisher/Gauss-Newton metric; same frozen targets."""
from pathlib import Path
import sys
R101=Path(__file__).resolve().parents[2];H101=R101/'research/round101';B101=R101/'crossplatform/round101'
sys.path.insert(0,str(R101/'research/round99'));from core99 import *
FITS101=['kl3','kl3_shuffle'];REFS101=ARMS99;ARMS101=REFS101+[n+'_'+v for n in FITS101 for v in ['F','C','FC']]
def probabilities101(m,views,r0):return torch.stack([read94(m,f,r0) for f in views])
def loss101(p,target):return (target*(target.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum(-1).mean()
def fisher_gram101(a,g,p):
 nv,k=p.shape;n=nv*k;G=((g@g.T)*(a@a.T+1)).reshape(nv,k,nv,k);row=(G*p[:,:,None,None]).sum(1,keepdim=True);col=(G*p[None,None,:,:]).sum(3,keepdim=True);both=(G*p[:,:,None,None]*p[None,None,:,:]).sum((1,3),keepdim=True);G=(G-row-col+both).reshape(n,n);sq=p.sqrt().flatten();return G*sq[:,None]*sq[None,:]
def proposal101(a,g,p,target,damping):
 nv,k=p.shape;G=fisher_gram101(a,g,p);sq=p.sqrt();e=((target-p)/sq).flatten();v=torch.linalg.solve(G+damping*torch.eye(nv*k,device=p.device,dtype=p.dtype),e);coeff=sq*v.reshape(nv,k);coeff=coeff-p*coeff.sum(1,keepdim=True);coeff=coeff.flatten();return (coeff[:,None]*g).T@a,(coeff[:,None]*g).sum(0)
def fit101(m,views,r0,target,S,maxiter=10):
 put94(m,S);m.requires_grad_(False);initial=damping=None;records=[];reason='max_iterations';nv=len(views)
 for it in range(maxiter):
  with torch.no_grad():
   aa,gg,_=zip(*(components_solve(m,f,r0) for f in views));a=torch.cat(aa);g=torch.cat(gg);p=probabilities101(m,views,r0);before=float(loss101(p,target))
   if initial is None:initial=before
   if before<=max(1e-12,initial*1e-3):reason='relative_error';break
   if damping is None:damping=max(1e-12,.01*float(fisher_gram101(a,g,p).trace())/(nv*256))
   old=patch94(m);accepted=False
   for tr in range(6):
    used=damping;dW,db=proposal101(a,g,p,target,used);delta={'weight':dW.float().cpu().numpy(),'bias':db.float().cpu().numpy()};st={k:old[k]+delta[k] for k in old};put94(m,st);dw=torch.as_tensor(st['weight']-old['weight'],device='cuda').double();dbb=torch.as_tensor(st['bias']-old['bias'],device='cuda').double();pred=(g*F.linear(a,dw,dbb)).sum(-1).reshape(nv,256);pred=pred-(p*pred).sum(1,keepdim=True);predred=float((((target-p)*pred).sum(1)-.5*(p*pred.square()).sum(1)).mean());after=float(loss101(probabilities101(m,views,r0),target));rho=(before-after)/max(predred,1e-30);ok=bool(predred>0 and np.isfinite(after) and after<before and rho>.1);records.append({'iteration':it+1,'trial':tr+1,'before':before,'after':after,'predicted_reduction':predred,'rho':rho,'damping':used,'accepted':ok,'step_L2':float(np.sqrt(sum(np.square((st[k]-old[k]).astype(np.float64)).sum() for k in st)))})
    if ok:
     accepted=True
     if rho>.75:damping=max(1e-15,damping/3)
     elif rho<.25:damping*=2
     break
    put94(m,old);damping*=10
   if not accepted:reason='no_accepted_step';break
 with torch.no_grad():after=float(loss101(probabilities101(m,views,r0),target))
 st=patch94(m);return st,records,{'initial_error':initial if initial is not None else after,'final_error':after,'relative_error':after/max(initial or 0.,1e-30),'accepted_steps':sum(t['accepted'] for t in records),'attempted_steps':len(records),'stop_reason':reason,'parameter_L2':float(np.sqrt(sum(np.square((st[k]-S[k]).astype(np.float64)).sum() for k in st))),'objective':'categorical_KL','fit_views':nv}
def check101(m,views,r0,target,S):
 put94(m,S);aa,gg,_=zip(*(components_solve(m,f,r0) for f in views));a=torch.cat(aa);g=torch.cat(gg);nv=len(views);J=torch.cat([(g[:,:,None]*a[:,None,:]).reshape(nv*256,-1),g],1).reshape(nv,256,-1);P=probabilities101(m,views,r0);Jw=(J-(P[:,:,None]*J).sum(1,keepdim=True))*P.sqrt()[:,:,None];Jw=Jw.reshape(nv*256,-1);err=float(abs(Jw@Jw.T-fisher_gram101(a,g,P)).max());assert err<1e-7
 W=m.match[-1].weight.detach().double().requires_grad_(True);b=m.match[-1].bias.detach().double().requires_grad_(True);loss=0.;pdouble=[]
 for av,f in zip(aa,views):
  raw=F.linear(av,W,b)+f['late'].double();qq=F.normalize(raw,dim=-1);s=native_scores(qq,f['u'].double(),f['mask'].bool())+r0;pdouble.append(s.softmax(0))
 pd=torch.stack(pdouble);L=loss101(pd,target);dw,db=torch.autograd.grad(L,(W,b));e=(pd.detach()-target).flatten()/nv;gw=(e[:,None]*g).T@a;gb=(e[:,None]*g).sum(0);graderr=max(float(abs(dw-gw).max()),float(abs(db-gb).max()));assert graderr<1e-8;return {'explicit_Fisher_Gram_max_error':err,'autograd_KL_gradient_max_error':graderr}
