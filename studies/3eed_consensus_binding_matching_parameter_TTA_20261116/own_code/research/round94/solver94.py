"""One disclosed post-DEV numerical repair: damped score-Jacobian edit, no new evidence."""
from core94 import *
B_SOLVE=B94/'solver_repair336_attempt001';METHOD_SOLVE=['gn','gradient','gn_mixture','gn_shuffle']
def components_solve(m,f,r0):
 # Existing nonlinear matching prefix fixed; the trainable last linear layer is exact.
 a=m.match[:-1](f['h'][None])[0].double();W=m.match[-1].weight.double();bias=m.match[-1].bias.double();raw=F.linear(a,W,bias)+f['late'].double();norm=raw.norm(dim=-1,keepdim=True).clamp_min(1e-12);q=raw/norm;u=f['u'].double();mask=f['mask'].bool();v=q@u.T/.07;pos=v[:,mask].softmax(-1)@u[mask];neg=v[:,~mask].softmax(-1)@u[~mask];g=(pos-neg)/.07;g=(g-q*(g*q).sum(-1,keepdim=True))/norm;s=torch.logsumexp(v[:,mask],-1)-torch.logsumexp(v[:,~mask],-1)+r0;return a,g,s

def proposal_solve(m,f,r0,target,kind):
 with torch.no_grad():
  a,g,s=components_solve(m,f,r0);error=(target-s);error-=error.mean();G=(g@g.T)*(a@a.T+1);G=G-G.mean(0,keepdim=True)-G.mean(1,keepdim=True)+G.mean();damping=float(G.trace()/256)*.01
  if kind=='gradient':v=error/256
  else:v=torch.linalg.solve(G+damping*torch.eye(256,device='cuda',dtype=torch.float64),error) if damping>1e-16 else error*0
  v=v-v.mean();dW=(v[:,None]*g).T@a;db=(v[:,None]*g).sum(0);ds={'weight':dW.float().cpu().numpy(),'bias':db.float().cpu().numpy()};norm=float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in ds.values())));base_norm=float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in patch94(m).values())));limit=.05*base_norm
  if norm>limit:ds={k:x*(limit/norm) for k,x in ds.items()};norm=limit
 return ds,{'damping':damping,'unconstrained_L2':norm,'source_relative_radius_limit':limit,'gram_trace':float(G.trace())}

def score_loss_solve(m,f,r0,target):
 p=read94(m,f,r0);s=p.log();err=s-target;err=err-err.mean();return err.square().mean()/2,p
