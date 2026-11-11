"""Fixed-reference bidirectional token KL; no task labels or held mask access."""
from common75 import *
import torch,copy
sys.path.insert(0,str(ROOT/'research/round58'));from numerics58 import head,state,optimizer,phash,ohash

def rep(m,h):return torch.nn.functional.normalize(m(h),dim=-1)
def matrix(q,u):return q.double()@u.double().T/.07
def cand(S,mask):return (torch.logsumexp(S[:,mask],1)-torch.logsumexp(S[:,~mask],1)).softmax(0)
def KL(logp,logq,dim):return (logp.exp()*(logp-logq)).sum(dim)
def data(initial,z,masked,perm):
 m=head(initial);d={'h0':torch.tensor(z['h']),'u':torch.tensor(z['tokens']),'mask':torch.tensor(z['token_mask']).bool(),'perm':torch.tensor(perm).long()}
 for v in ['M1','M2']:d[v]=torch.tensor(masked[v+'_h'])
 with torch.no_grad():
  d['q0']=rep(m,d['h0']);d['S0']=matrix(d['q0'],d['u']);d['p0']=cand(d['S0'],d['mask']);d['R0']=d['S0'].log_softmax(1);d['C0']=d['S0'].log_softmax(0)
 return d

def objective(m,d,kind):
 dense=[];coarse=[]
 for v in ['M1','M2']:
  S=matrix(rep(m,d[v]),d['u'])
  if kind=='shuffled':R=d['R0'][d['perm']];C=d['C0'][d['perm']]
  else:R=d['R0'];C=d['C0']
  dense.append(KL(R,S.log_softmax(1),1).mean()+KL(C,S.log_softmax(0),0).mean());pv=cand(S,d['mask']);coarse.append((d['p0']*(d['p0'].clamp_min(1e-300).log()-pv.clamp_min(1e-300).log())).sum())
 keep=(rep(m,d['h0'])-d['q0']).double().square().sum(1).mean()/2;aux=torch.stack(coarse if kind=='output' else dense).mean();return aux+keep,aux,keep

def one(initial,d,kind):
 m=head(initial);opt=optimizer(m);old={n:v.detach().clone() for n,v in m.named_parameters()};oldopt=copy.deepcopy(opt.state_dict());before_hash=phash(m);opt_hash=ohash(opt);loss,aux,keep=objective(m,d,kind);before=float(loss.detach());ab=float(aux.detach());assert np.isfinite(before);opt.zero_grad();loss.backward();gn=float(torch.sqrt(sum(p.grad.square().sum() for p in m.parameters())));assert np.isfinite(gn);torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();proposal={n:v.detach().clone() for n,v in m.named_parameters()};trials=[];accepted=False;scale=0
 with torch.no_grad():
  for j in range(9):
   alpha=2.**(-j)
   for n,v in m.named_parameters():v.copy_(old[n]+alpha*(proposal[n]-old[n]))
   l,a,k=objective(m,d,kind);after=float(l);trials.append({'scale':alpha,'loss':after,'auxiliary':float(a),'keep':float(k)})
   if np.isfinite(after) and after<=before-max(1e-12,1e-6*abs(before)):accepted=True;scale=alpha;break
  if not accepted:
   for n,v in m.named_parameters():v.copy_(old[n])
   opt.load_state_dict(oldopt)
  l,a,k=objective(m,d,kind);delta=float(torch.sqrt(sum((v-old[n]).square().sum() for n,v in m.named_parameters())))
 assert float(l)<before and delta>0 if accepted else (phash(m)==before_hash and ohash(opt)==opt_hash and delta==0)
 opt.zero_grad(set_to_none=True);return state(m),{'kind':kind,'accepted':accepted,'scale':scale,'loss_before':before,'loss_after':float(l),'auxiliary_before':ab,'auxiliary_after':float(a),'keep_after':float(k),'gradient_norm':gn,'parameter_step_L2':delta,'trials':trials,'before_sha256':before_hash,'after_sha256':phash(m),'optimizer_before_sha256':opt_hash,'optimizer_after_sha256':ohash(opt),'reference_text_fixed':True,'M3_used':False,'GT_or_expert_used':False}
