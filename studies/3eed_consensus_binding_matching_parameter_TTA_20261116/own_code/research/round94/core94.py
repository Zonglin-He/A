"""Dense calibration-specific visual evidence; only existing final match projection adapts."""
from pathlib import Path
import sys,hashlib,copy
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round93'))
from core93 import *
H94=ROOT/'research/round94';B94=ROOT/'crossplatform/round94';P93=B93/'prediction336_attempt001'
REFS94=['source','ABC','R59','R91_frozen','R91_mean3','R93_bt','R93_raw']
DIRECT94=['binding_direct','binding_optimal_mixture','reverse_direct','shuffle_direct','viewmean_direct','viewmean_optimal_mixture']
NEW94=['binding_raw1','binding_bt1','binding_bt3','shuffle_bt1','reverse_bt1','viewmean_bt1'];ARMS94=REFS94+DIRECT94+NEW94

def permutation94(r,d,repeat):
 # Visible-only fixed cyclic derangement; no point changes visibility or physical coordinates.
 visible=torch.nonzero(d['visible'],as_tuple=True)[0].cpu().numpy();seed=int(hashlib.sha256(('R94binding|'+r['id']).encode()).hexdigest()[:16],16);order=np.random.default_rng(seed).permutation(visible);perm=np.arange(1024)
 if len(order)>1:perm[order]=np.roll(order,(1 if repeat==0 else max(1,len(order)//2)))
 return perm

def bound94(d,perm):
 p=torch.as_tensor(perm,device='cuda');inv=torch.argsort(p);x=dict(d);x['p2i']=d['p2i'][p];x['p2i_weight']=d['p2i_weight'][p];x['i2p']=torch.where(d['i2p']>=0,inv[d['i2p'].clamp_min(0)],d['i2p']);return x

def qperm94(r):return np.random.default_rng(int(hashlib.sha256(('R94querycontrol|'+r['id']).encode()).hexdigest()[:16],16)).permutation(256)

def targets94(correct,wrong,r0,r):
 # All256 retain source score; correction is gauge invariant and does not select a positive class.
 b0=correct[0];s0=b0+r0;delta=correct.mean(0)-wrong.mean((0,1));delta=delta-delta.mean();target=(s0+delta).softmax(0);reverse=(s0-delta).softmax(0);shuffle=(s0+delta[torch.as_tensor(qperm94(r),device='cuda')]).softmax(0);mean=(correct.mean(0)+r0).softmax(0);p0=s0.softmax(0)
 return {'binding':target,'shuffle':shuffle,'reverse':reverse,'viewmean':mean},p0,delta

def head94(m):return list(m.match[-1].parameters())
def patch94(m):return {n:p.detach().cpu().numpy().copy() for n,p in m.match[-1].named_parameters()}
def put94(m,s):
 with torch.no_grad():
  for n,p in m.match[-1].named_parameters():p.copy_(torch.as_tensor(s[n],device=p.device))
def enable94(m):
 m.requires_grad_(False)
 for p in head94(m):p.requires_grad_(True)
def read94(m,f,r0):
 q=F.normalize(m.match(f['h'][None])[0]+f['late'],dim=-1);s=native_scores(q,f['u'],f['mask'].bool()).double()+r0;return s.softmax(0)
def loss94(p,target,p0):return (target*(target.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()+(p0*(p0.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum()

def optimize94(m,f,r0,target,p0,steps,S):
 put94(m,S);enable94(m);ps=head94(m);opt=torch.optim.Adam(ps,lr=.0005);records=[];states={};probs={}
 for step in range(1,steps+1):
  opt.zero_grad();p=read94(m,f,r0);before=loss94(p,target,p0);before.backward();torch.nn.utils.clip_grad_norm_(ps,1.);g=[p.grad.detach().clone() for p in ps];old=patch94(m);os=copy.deepcopy(opt.state_dict());opt.step();proposal=patch94(m);delta={k:proposal[k]-v for k,v in old.items()};trials=[];accepted=False
  for k in range(9):
   scale=2.**(-k);ss={n:old[n]+scale*delta[n] for n in old};put94(m,ss)
   with torch.no_grad():pp=read94(m,f,r0);after=float(loss94(pp,target,p0))
   dot=sum(float((gg.double()*torch.as_tensor(ss[n]-old[n],device='cuda').double()).sum()) for (n,_),gg in zip(m.match[-1].named_parameters(),g));bound=float(before)+1e-4*dot;ok=after<=bound+1e-10;key='step'+str(step)+'_trial'+str(k);states[key]=ss;trials.append({'scale':scale,'loss':after,'bound':bound,'accepted':ok,'state_key':key,'parameter_L2':float(np.sqrt(sum(np.square(ss[n]-S[n]).sum() for n in ss)))})
   if k==0:probs['raw'+str(step)]=pp.detach().cpu().numpy()
   if ok:accepted=True;break
  if not accepted:put94(m,old);opt.load_state_dict(os)
  with torch.no_grad():pp=read94(m,f,r0)
  key='endpoint'+str(step);states[key]=patch94(m);probs['bt'+str(step)]=pp.detach().cpu().numpy();records.append({'step':step,'before':float(before.detach()),'after':float(loss94(pp,target,p0)),'accepted':accepted,'accepted_scale':trials[-1]['scale'] if accepted else 0.,'endpoint_key':key,'trials':trials,'parameter_L2':float(np.sqrt(sum(np.square(states[key][n]-S[n]).sum() for n in S)))})
 return probs,records,states
