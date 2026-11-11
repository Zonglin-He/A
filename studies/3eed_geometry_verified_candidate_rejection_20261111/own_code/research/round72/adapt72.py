"""Actual episodic native matching-MLP updates; no external feedback or labels."""
from common72 import *
import sys,time,torch
sys.path.insert(0,str(ROOT/'research/round58'));from numerics58 import head,state,optimizer,phash

def logodds(m,h,t,mask):
 q=torch.nn.functional.normalize(m(h),dim=-1);l=q@t.T/.07
 return torch.logsumexp(l[:,mask],-1)-torch.logsumexp(l[:,~mask],-1)
def distribution(m,h,t,mask,z0,p0):
 return (p0.log()+logodds(m,h,t,mask).double()-z0).softmax(-1)
def objective(p,p0,N):
 return -torch.log1p(-p[N].sum().clamp(max=1-1e-14))+(p0*(p0.log()-p.clamp_min(1e-300).log())).sum()
def main():
 p=read(H/'experiment/ADAPT_PLAN_LOCK.json');pin(p);out=ROOT/p['adapt_root'];assert not out.exists();out.mkdir(parents=True)
 for f in ['samples_PRIVATE','cases_PRIVATE','actual_head_states_PRIVATE']:(out/f).mkdir()
 initial=load(ROOT/p['head_initial']);e=ROOT/p['evidence_root'];es=read(e/'SEALED.json');start=time.perf_counter();counts={'n':0,'graph_active':0,'RGBshuffle_active':0,'Nshuffle_active':0,'actual_optimizer_steps':0};torch.set_num_threads(4)
 for r in rows():
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(e/rel)==es['files'][rel];ev=load(e/rel);z=prefix(r,('h','tokens','token_mask','scores'));p0=torch.as_tensor(z['scores'].astype(float)).clamp_min(1e-300);p0=p0/p0.sum();outputs={};records={}
  for tag,key in [('graph','main_N'),('RGBshuffle','RGB_N'),('Nshuffle','Nshuffle_N')]:
   N=torch.as_tensor(ev[key][ev['query_group']]).bool();active=bool(N.any() and not N.all() and p0[N].sum()>1e-12);counts[tag+'_active']+=active;steps=[]
   if active:
    m=head(initial);opt=optimizer(m);h=torch.as_tensor(z['h']);t=torch.as_tensor(z['tokens']);mask=torch.as_tensor(z['token_mask']).bool()
    with torch.no_grad():z0=logodds(m,h,t,mask).double().clone()
    for step in range(1,4):
     before=phash(m);opt.zero_grad(set_to_none=True);prob=distribution(m,h,t,mask,z0,p0);L=objective(prob,p0,N);L.backward();gn=float(torch.sqrt(sum(v.grad.square().sum() for v in m.parameters())));assert np.isfinite(gn);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();counts['actual_optimizer_steps']+=1
     with torch.no_grad():pp=distribution(m,h,t,mask,z0,p0);afterloss=float(objective(pp,p0,N));arr=pp.numpy().copy()
     assert np.isfinite(arr).all();steps.append({'step':step,'loss_before':float(L.detach()),'loss_after':afterloss,'gradient_norm':gn,'before_sha256':before,'after_sha256':phash(m),'rejected_mass':float(arr[N.numpy()].sum()),'parameter_change_L2':float(np.sqrt(sum(np.square(v-initial[k]).sum() for k,v in state(m).items())))})
     if step in (1,3):
      np.savez_compressed(out/'actual_head_states_PRIVATE'/(sid+'_'+tag+str(step)+'.npz'),**state(m));outputs[tag+'_tta'+str(step)]=arr
   else:
    for step in (1,3):outputs[tag+'_tta'+str(step)]=z['scores'].astype(float).copy()
   records[tag]={'active':active,'N_groups':int(ev[key].sum()),'N_queries':int(N.sum()),'initial_rejected_mass':float(p0[N].sum()),'steps':steps}
  np.savez_compressed(out/rel,**outputs);dump(out/'cases_PRIVATE'/(sid+'.json'),records);counts['n']+=1
  if counts['n']%256==0:print('ADAPT72',counts,flush=True)
 counts.update(seconds=time.perf_counter()-start,updated_parameters=sum(x.size for x in initial.values()),new_external_model_calls=0,new_GT_lookups=0);dump(out/'COUNTS.json',counts);seal(out,'R72_FIXED_NEGATIVES_NATIVE_MATCHING_UPDATES_SEALED_BEFORE_TASK_LOOKUP',n=2640);print(counts)
if __name__=='__main__':main()
