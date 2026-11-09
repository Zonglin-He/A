"""Cached original matching head and R57 objective; no label or media access."""
import torch,copy,hashlib
import numpy as np

def head(initial):
 m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64));m.load_state_dict({k:torch.as_tensor(v) for k,v in initial.items()});return m.eval()
def state(m):return {k:v.detach().numpy().copy() for k,v in m.state_dict().items()}
def phash(m):
 h=hashlib.sha256()
 for n,v in m.state_dict().items():h.update(n.encode()+v.detach().numpy().tobytes())
 return h.hexdigest()
def ohash(opt):
 h=hashlib.sha256();s=opt.state_dict()
 for i,v in sorted(s['state'].items()):
  h.update(str(i).encode())
  for name,value in sorted(v.items()):h.update(name.encode()+value.detach().numpy().tobytes())
 h.update(str(s['param_groups']).encode());return h.hexdigest()
def optimizer(s):return torch.optim.Adam(s.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0)
def prepared(v,c,g):
 T=lambda a:torch.as_tensor(a)
 d={view:{'h':T(z[pre+'h'])[None],'tokens':T(z[pre+'tokens'])[None],'mask':T(z[pre+'token_mask']).bool()} for view,z,pre in [('A',v,'A_'),('B',v,'B_'),('C',c,'')]}
 d['base']=T(g['base']).double()
 for view in ['B','C']:d[view].update(W=T(g[view+'_W']).double(),mass=T(g[view+'_fixed_mass']).double())
 return d
def distribution(m,d,view):
 x=d[view];q=torch.nn.functional.normalize(m(x['h']),dim=-1);l=q@x['tokens'].transpose(-1,-2)/.07;mask=x['mask'][None,None];z=(torch.logsumexp(l.masked_fill(~mask,-torch.inf),-1)-torch.logsumexp(l.masked_fill(mask,-torch.inf),-1))[0];p=z.double().softmax(-1)
 if view=='A':return p
 pp=x['W']@p;return x['mass']*pp/pp.sum().clamp_min(1e-300)+(1-x['mass'])*d['base']
def ensemble(m,d,views='ABC'):return sum(distribution(m,d,v) for v in views)/len(views)
def objective(s,d,target):
 pp=distribution(s,d,'C');return (target*(target.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()
def permitted(before,after):return np.isfinite(after) and after<=before-max(1e-12,1e-6*abs(before))
def step(s,t,opt,d,kind,rule):
 with torch.no_grad():target=ensemble(t,d,'AB').detach().clone();told={n:v.detach().clone() for n,v in t.named_parameters()};old={n:v.detach().clone() for n,v in s.named_parameters()}
 teacher_before=phash(t);old_opt=copy.deepcopy(opt.state_dict()) if rule=='backtrack' else None;opt_before=ohash(opt);before_hash=phash(s);opt.zero_grad(set_to_none=True);L=objective(s,d,target);before=float(L.detach());assert np.isfinite(before);L.backward();gn=float(torch.sqrt(sum(v.grad.square().sum() for v in s.parameters())));assert np.isfinite(gn);torch.nn.utils.clip_grad_norm_(s.parameters(),1)
 opt.step();proposed={n:v.detach().clone() for n,v in s.named_parameters()};delta={n:proposed[n]-old[n] for n in old};proposed_opt=ohash(opt);trials=[];accepted=False;scale=0.
 with torch.no_grad():
  for j in range(1 if rule=='original' else 9):
   alpha=2.**(-j)
   for n,v in s.named_parameters():v.copy_(proposed[n] if j==0 else old[n]+alpha*delta[n])
   after=float(objective(s,d,target));trials.append({'scale':alpha,'KL':after if np.isfinite(after) else None,'finite':bool(np.isfinite(after))})
   if rule=='original' or permitted(before,after):accepted=True;scale=alpha;break
  if not accepted:
   for n,v in s.named_parameters():v.copy_(old[n])
   opt.load_state_dict(old_opt)
  if kind=='ema' and accepted:
   for n,v in t.named_parameters():v.copy_(.99*told[n]+.01*dict(s.named_parameters())[n])
  final=float(objective(s,d,target));sd=float(torch.sqrt(sum((v-old[n]).square().sum() for n,v in s.named_parameters())));td=float(torch.sqrt(sum((v-told[n]).square().sum() for n,v in t.named_parameters())));assert np.isfinite(final) and all(v.grad is None and not v.requires_grad for v in t.parameters())
 after_hash=phash(s);opt_after=ohash(opt);assert opt_after==(proposed_opt if accepted else opt_before)
 if rule=='backtrack':assert permitted(before,final) if accepted else (final==before and before_hash==after_hash and td==0)
 opt.zero_grad(set_to_none=True)
 return {'KL_before':before,'KL_after_same_detached_target':final,'gradient_norm':gn,'teacher_step_L2':td,'student_step_L2':sd,'accepted':accepted,'scale':scale,'trial_count':len(trials),'trials':trials,'full_step_loss_increased':bool(trials[0]['KL'] is not None and trials[0]['KL']>before+1e-12),'student_before_hash':before_hash,'student_after_hash':after_hash,'optimizer_before_hash':opt_before,'optimizer_proposed_hash':proposed_opt,'optimizer_after_hash':opt_after,'teacher_before_hash':teacher_before,'teacher_after_hash':phash(t),'target_sha256':hashlib.sha256(target.numpy().tobytes()).hexdigest(),'target_requires_grad':False,'optimizer_step_after':int(next(iter(opt.state.values()))['step']) if opt.state else 0}
