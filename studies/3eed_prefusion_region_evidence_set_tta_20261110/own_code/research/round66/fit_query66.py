"""One native matching-MLP update per input, A/B conditional set loss only.
Same source initialization each input; C is not opened or part of stopping.
"""
import sys,time,copy,torch
from common66 import *
sys.path.insert(0,str(ROOT/'research/round58'));import numerics58 as n

def data(ab,e):
 d={v:{'h':torch.tensor(ab[v+'_h'])[None],'tokens':torch.tensor(ab[v+'_tokens'])[None],'mask':torch.tensor(ab[v+'_token_mask']).bool()} for v in 'AB'};d['base']=torch.tensor(e['base']).double();d['B'].update(W=torch.tensor(e['B_W']).double(),mass=torch.tensor(e['B_fixed_mass']).double());return d

def objective(m,d,e,prior):
 losses=[];keep=[];S=torch.tensor(e['positive_query_mask']);C=torch.tensor(e['tested_query_mask'])
 for v in 'AB':
  p=n.distribution(m,d,v);losses.append(-torch.log(p[S].sum().clamp_min(1e-300)/p[C].sum().clamp_min(1e-300)));q=prior[v];keep.append((q*(q.clamp_min(1e-300).log()-p.clamp_min(1e-300).log())).sum())
 return sum(losses)/2+sum(keep)/2

def one_step(m,d,e):
 prior={v:n.distribution(m,d,v).detach().clone() for v in 'AB'};opt=n.optimizer(m);old={k:v.detach().clone() for k,v in m.named_parameters()};beforehash=n.phash(m);before=float(objective(m,d,e,prior).detach());opt.zero_grad(set_to_none=True);L=objective(m,d,e,prior);L.backward();grad=float(torch.sqrt(sum(v.grad.square().sum() for v in m.parameters())));torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();proposal={k:v.detach().clone() for k,v in m.named_parameters()};trials=[];accepted=False;scale=0.
 with torch.no_grad():
  for k in range(9):
   alpha=2.**(-k)
   for name,val in m.named_parameters():val.copy_(proposal[name] if k==0 else old[name]+alpha*(proposal[name]-old[name]))
   val=float(objective(m,d,e,prior));trials.append(val)
   if n.permitted(before,val):accepted=True;scale=alpha;break
  if not accepted:
   for name,val in m.named_parameters():val.copy_(old[name])
  after=float(objective(m,d,e,prior));stepnorm=float(torch.sqrt(sum((val-old[name]).square().sum() for name,val in m.named_parameters())))
 return {'accepted':accepted,'scale':scale,'loss_before':before,'loss_after':after,'trials':trials,'gradient_norm':grad,'step_L2':stepnorm,'before_hash':beforehash,'after_hash':n.phash(m),'C_used':False}

def main():
 H=ROOT/'research/round66';p=read(H/'experiment/FIT_QUERY_PLAN_LOCK_v2.json');pin(p);assert not read(H/'STATE.json')['C_supervision_authorized'];rows=read(ROOT/p['manifest']);ev=ROOT/p['evidence_root'];es=read(ev/'SEALED.json');out=ROOT/p['fit_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();(out/'cases_PRIVATE').mkdir();initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');start=time.perf_counter();counts={'eligible':0,'accepted':0,'rejected':0,'no_feedback':0};torch.set_num_threads(4)
 for i,r in enumerate(rows):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(ev/rel)==es['files'][rel];e=load(ev/rel);ab=load(ROOT/p['AB_root']/rel);m=n.head(initial).requires_grad_(True)
  if bool(e['update_allowed']):rec=one_step(m,data(ab,e),e);counts['eligible']+=1;counts['accepted' if rec['accepted'] else 'rejected']+=1
  else:rec={'accepted':False,'scale':0.,'loss_before':0.,'loss_after':0.,'trials':[],'gradient_norm':0.,'step_L2':0.,'before_hash':n.phash(m),'after_hash':n.phash(m),'C_used':False};counts['no_feedback']+=1
  np.savez_compressed(out/rel,**n.state(m));dump(out/'cases_PRIVATE'/(sid+'.json'),rec)
  if (i+1)%64==0:print('FIT66',i+1,counts,flush=True)
 dump(out/'COSTS.json',{'n':336,'counts':counts,'C_read_or_used':False,'seconds':time.perf_counter()-start});seal(out,'R66_A_B_ONLY_MATCHING_HEAD_FITS_SEALED_BEFORE_C_READOUT',n=336);print(read(out/'COSTS.json'))
if __name__=='__main__':main()
