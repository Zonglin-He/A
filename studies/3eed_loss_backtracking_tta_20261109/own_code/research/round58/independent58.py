"""Independent functional head and transactional Adam replay; producer not imported."""
import torch,copy,hashlib,math

def params(arr,learn=False):return {k:torch.tensor(v.copy(),requires_grad=learn) for k,v in arr.items()}
def digest(p):
 h=hashlib.sha256()
 for k,v in p.items():h.update(k.encode()+v.detach().numpy().tobytes())
 return h.hexdigest()
def optdigest(o):
 h=hashlib.sha256();z=o.state_dict()
 for i,st in sorted(z['state'].items()):
  h.update(str(i).encode())
  for k,v in sorted(st.items()):h.update(k.encode()+v.detach().numpy().tobytes())
 h.update(str(z['param_groups']).encode());return h.hexdigest()
def inputs(v,c,g):
 d={}
 for view,cache,prefix in [('A',v,'A_'),('B',v,'B_'),('C',c,'')]:d[view]=(torch.tensor(cache[prefix+'h'])[None],torch.tensor(cache[prefix+'tokens'])[None],torch.tensor(cache[prefix+'token_mask']).bool())
 d['base']=torch.tensor(g['base']).double()
 for view in ['B','C']:d[view+'_W']=torch.tensor(g[view+'_W']).double();d[view+'_mass']=float(g[view+'_fixed_mass'])
 return d
def prediction(p,d,view):
 h,t,m=d[view];x=torch.nn.functional.linear(h,p['0.weight'],p['0.bias']).relu();x=torch.nn.functional.linear(x,p['2.weight'],p['2.bias']).relu();x=torch.nn.functional.linear(x,p['4.weight'],p['4.bias']);x=torch.nn.functional.normalize(x,dim=-1);l=x@t.transpose(-1,-2)/.07;z=torch.logsumexp(l.masked_fill(~m[None,None],-torch.inf),dim=-1)-torch.logsumexp(l.masked_fill(m[None,None],-torch.inf),dim=-1);pp=z[0].double().softmax(-1)
 if view=='A':return pp
 mapped=d[view+'_W']@pp;return d[view+'_mass']*mapped/mapped.sum().clamp_min(1e-300)+(1-d[view+'_mass'])*d['base']
def mean_prediction(p,d,views='ABC'):return sum(prediction(p,d,v) for v in views)/len(views)
def divergence(p,d,target):
 pp=prediction(p,d,'C');return (target*(target.clamp_min(1e-300).log()-pp.clamp_min(1e-300).log())).sum()
def create(initial):
 s=params(initial,True);t=params(initial);o=torch.optim.Adam(list(s.values()),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0);return s,t,o

def independently_step(s,t,o,d,kind,rule):
 old={n:v.detach().clone() for n,v in s.items()};oldt={n:v.detach().clone() for n,v in t.items()};oldopt=copy.deepcopy(o.state_dict());oldhash=optdigest(o)
 with torch.no_grad():target=mean_prediction(t,d,'AB').detach().clone()
 o.zero_grad(set_to_none=True);L=divergence(s,d,target);b=float(L.detach());L.backward();torch.nn.utils.clip_grad_norm_(list(s.values()),1);o.step();tent={n:v.detach().clone() for n,v in s.items()};tenthash=optdigest(o);directional_derivative=float(sum((s[n].grad*(tent[n]-old[n])).sum() for n in s));trials=[];alpha=0.;accepted=False
 with torch.no_grad():
  for k in range(1 if rule=='original' else 9):
   scale=2.**(-k)
   for n,v in s.items():v.copy_(tent[n] if k==0 else old[n]+scale*(tent[n]-old[n]))
   val=float(divergence(s,d,target));trials.append(val)
   if rule=='original' or (math.isfinite(val) and val<=b-max(1e-12,1e-6*abs(b))):accepted=True;alpha=scale;break
  if not accepted:
   for n,v in s.items():v.copy_(old[n])
   o.load_state_dict(oldopt)
  if accepted and kind=='ema':
   for n,v in t.items():v.copy_(.99*oldt[n]+.01*s[n])
  final=float(divergence(s,d,target))
 o.zero_grad(set_to_none=True)
 return {'directional_derivative':directional_derivative,'KL_before':b,'KL_after_same_detached_target':final,'accepted':accepted,'scale':alpha,'trials':trials,'student_after_hash':digest(s),'teacher_after_hash':digest(t),'optimizer_before_hash':oldhash,'optimizer_proposed_hash':tenthash,'optimizer_after_hash':optdigest(o),'target_sha256':hashlib.sha256(target.numpy().tobytes()).hexdigest()}
def compare_step(actual,independent,*,compare_teacher=True):
 for key in ['accepted','scale','student_after_hash','optimizer_before_hash','optimizer_proposed_hash','optimizer_after_hash','target_sha256']:
  assert actual[key]==independent[key],('step',key,actual[key],independent[key])
 if compare_teacher:assert actual['teacher_after_hash']==independent['teacher_after_hash']
 assert len(actual['trials'])==len(independent['trials'])
 error=max(abs(actual['KL_before']-independent['KL_before']),abs(actual['KL_after_same_detached_target']-independent['KL_after_same_detached_target']))
 for v,w in zip(actual['trials'],independent['trials']):
  assert v['KL'] is not None and math.isfinite(w);error=max(error,abs(v['KL']-w))
 assert error<1e-10,('KL',error)
 return error
