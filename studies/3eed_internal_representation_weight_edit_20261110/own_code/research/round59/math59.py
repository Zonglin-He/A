"""Internal target, weighted ridge edit and native grounding readout; no task GT."""
import numpy as np,torch,scipy.linalg

def head(initial):
 m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64));m.load_state_dict({k:torch.tensor(v.copy()) for k,v in initial.items()});return m.eval().requires_grad_(False)
def fingerprint(m):
 import hashlib
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode()+v.detach().numpy().tobytes())
 return h.hexdigest()
def unit(x,fallback=None):
 norm=np.linalg.norm(x,axis=-1,keepdims=True);out=np.divide(x,norm,out=np.zeros_like(x),where=norm>1e-12)
 if fallback is not None:out=np.where(norm>1e-12,out,fallback)
 return out

def features(model,v,c):
 out={}
 with torch.no_grad():
  for name,z,pre in [('A',v,'A_'),('B',v,'B_'),('C',c,'')]:
   hh=torch.as_tensor(z[pre+'h'])[None];x=model[3](model[2](model[1](model[0](hh))))[0].numpy().copy();out[name+'_X']=x;out[name+'_H']=np.column_stack([x.astype(float),np.ones(256)]);out[name+'_tokens']=z[pre+'tokens'];out[name+'_mask']=z[pre+'token_mask']
 return out

def target_and_system(f,g,M):
 raw={v:f[v+'_H']@M.T for v in 'ABC'};q={v:unit(raw[v]) for v in 'ABC'};WB=g['B_W'];WC=g['C_W'];rb=g['B_affinity'].max(1);rowmass=WB.sum(1);qb=np.divide(WB@q['B'],rowmass[:,None],out=np.zeros_like(q['A']),where=rowmass[:,None]>0);ua=unit(q['A']+rb[:,None]*qb,q['A']);targets={'A':np.linalg.norm(raw['A'],axis=1)[:,None]*ua};confidence={'A':rb}
 for v,W in [('B',WB),('C',WC)]:
  mass=W.sum(0);uu=np.divide(W.T@ua,mass[:,None],out=np.zeros_like(q[v]),where=mass[:,None]>0);valid=np.linalg.norm(uu,axis=1)>1e-12;u=unit(uu,q[v]);conf=g[v+'_affinity'].max(0)*valid;confidence[v]=conf;targets[v]=np.linalg.norm(raw[v],axis=1)[:,None]*u
 weight=confidence['C'];den=max(float(weight.sum()),1e-12);Hw=f['C_H']*np.sqrt(weight/den)[:,None];E=(targets['C']-raw['C'])*np.sqrt(weight/den)[:,None];G=Hw.T@Hw;B=E.T@Hw;return raw,targets,confidence,Hw,E,G,B

def solve(G,B,M):
 ev,V=scipy.linalg.eigh(G,check_finite=True,driver='evr');ev=np.maximum(ev,0);base=max(float(np.trace(G))/289,1e-12);cap=.05*float(np.linalg.norm(M));R=B@V
 def result(lam):return (R/(ev+lam)[None])@V.T
 lam=base;delta=result(lam);constrained=float(np.linalg.norm(delta))>cap
 if constrained:
  lo=base;hi=base
  while np.linalg.norm(R/(ev+hi)[None])>cap:hi*=2
  for _ in range(50):
   mid=(lo+hi)/2
   if np.linalg.norm(R/(ev+mid)[None])>cap:lo=mid
   else:hi=mid
  lam=hi;delta=result(lam)
 tol=max(float(ev[-1]),1.)*289*np.finfo(float).eps;null=V[:,ev<=tol];nullaction=float(np.linalg.norm(delta@null)) if null.shape[1] else 0.;normal=float(np.linalg.norm(delta@(G+lam*np.eye(289))-B));return delta,{'lambda_base':base,'lambda_effective':lam,'cap_active':constrained,'edit_cap_L2':cap,'edit_L2':float(np.linalg.norm(delta)),'relative_edit_L2':float(np.linalg.norm(delta)/max(np.linalg.norm(M),1e-300)),'feature_rank':int((ev>tol).sum()),'normal_equation_residual_L2':normal,'null_action_L2':nullaction,'step_size':1/(2*(float(ev[-1])+lam)),'max_eigenvalue':float(ev[-1])}

def objective(delta,Hw,E,lam):return float(np.square(Hw@delta.T-E).sum()+lam*np.square(delta).sum())
def gradient_trajectory(M,Hw,E,lam,eta):
 W=torch.nn.Parameter(torch.tensor(M.copy(),dtype=torch.float64));base=torch.tensor(M,dtype=torch.float64);h=torch.tensor(Hw,dtype=torch.float64);e=torch.tensor(E,dtype=torch.float64);opt=torch.optim.SGD([W],lr=eta);states={};logs=[]
 for k in range(1,257):
  opt.zero_grad(set_to_none=True);d=W-base;L=((h@d.T-e).square().sum()+lam*d.square().sum());L.backward();gn=float(W.grad.norm());opt.step()
  if k in [1,3,16,256]:
   delta=W.detach().numpy()-M;states[k]=W.detach().numpy().copy();logs.append({'step':k,'objective_before':float(L.detach()),'objective_after':objective(delta,Hw,E,lam),'gradient_norm_before':gn,'delta_L2':float(np.linalg.norm(delta))})
 return states,logs

def score_from_raw(raw,f,g):
 vv={}
 with torch.no_grad():
  for v in 'ABC':
   y=raw[v];yy=y if isinstance(y,torch.Tensor) else torch.tensor(np.asarray(y,np.float32))[None];yy=yy[None] if yy.ndim==2 else yy;q=torch.nn.functional.normalize(yy,dim=-1);t=torch.as_tensor(f[v+'_tokens'])[None];mask=torch.as_tensor(f[v+'_mask']).bool()[None,None];l=q@t.transpose(-1,-2)/.07;z=(torch.logsumexp(l.masked_fill(~mask,-torch.inf),-1)-torch.logsumexp(l.masked_fill(mask,-torch.inf),-1))[0];p=z.double().softmax(-1).numpy()
   if v!='A':
    pp=g[v+'_W']@p;mass=float(g[v+'_fixed_mass']);p=mass*pp/max(float(pp.sum()),1e-300)+(1-mass)*g['base']
   vv[v]=p
 return sum(vv.values())/3

def readout(model,f,g):
 with torch.no_grad():raw={v:model[4](torch.as_tensor(f[v+'_X'])[None]) for v in 'ABC'}
 return score_from_raw(raw,f,g)
def virtual_readout(M,f,g):
 w=torch.tensor(M[:,:288],dtype=torch.float32);b=torch.tensor(M[:,288],dtype=torch.float32)
 with torch.no_grad():raw={v:torch.nn.functional.linear(torch.as_tensor(f[v+'_X'])[None],w,b) for v in 'ABC'}
 return score_from_raw(raw,f,g)
def install(model,M):
 with torch.no_grad():model[4].weight.copy_(torch.tensor(M[:,:288],dtype=torch.float32));model[4].bias.copy_(torch.tensor(M[:,288],dtype=torch.float32))
