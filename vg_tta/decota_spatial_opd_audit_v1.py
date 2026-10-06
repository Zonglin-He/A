"""Independent float64 Gaussian/control-variate/IoU/Adam/state arithmetic."""
import numpy as np
from vg_tta.decota_fixed_full_audit_v1 import top1_support,overlap,vector

def audit(z,expert):
 names=list(z['initial']);hist=z['path'];support=top1_support(expert)
 assert z['selected_step']==10 and len(hist)==11 and len(z['rounds'])==10
 assert z['positions']==[p for p,_ in support] and z['active_parameters']==1792 and not z['GT_used']
 assert np.array_equal(vector(z['state'],names),vector(hist[10]['state'],names))
 assert np.array_equal(z['final'].numpy(),hist[10]['boxes'].numpy())
 m=np.zeros(1792);v=np.zeros(1792);updates=0;errgrad=erradam=errreward=0.
 for k,(h,r) in enumerate(zip(hist,z['rounds'])):
  a=vector(h['state'],names);nxt=vector(hist[k+1]['state'],names)
  if not support:
   assert not r['updated'] and np.array_equal(a,nxt);continue
  rr=r['rollout'];samples=rr['samples'].numpy().astype(float);mu=rr['mean'].numpy().astype(float)
  w=rr['weights'].numpy().astype(float);used=rr['used_rewards'].numpy().astype(float)
  assert samples.shape==(len(support),32,4)
  original=rr['rewards'].numpy().astype(float);actions=1/(1+np.exp(-samples))
  recomputed=np.asarray([[overlap(x,e) for x in aa] for aa,(_,e) in zip(actions,support)])
  error=float(abs(original-recomputed).max());assert error<3e-6;errreward=max(errreward,error)
  if rr['permutations'] is not None:assert np.array_equal(used,np.take_along_axis(original,rr['permutations'].numpy(),1))
  else:assert np.array_equal(used,original)
  logits=used/.25;ew=np.exp(logits-logits.max(1,keepdims=True));ew/=ew.sum(1,keepdims=True)
  assert abs(ew-w).max()<3e-7
  current=r['mean_before'].numpy().astype(float)
  expected=((current-mu)/.25**2-((w-1/32)[:,:,None]*(samples-current[:,None,:])).sum(1)/.25**2)/len(support)
  error=float(abs(expected-r['autograd_mean_gradient'].numpy()).max());assert error<3e-5;errgrad=max(errgrad,error)
  assert r['updated']==rr['informative'] and r['no_information']==(not rr['informative'])
  if not r['updated']:assert np.array_equal(a,nxt);continue
  u=h['update'];g=u['gradient'].numpy().astype(float);updates+=1
  m=.9*m+.1*g;v=.999*v+.001*g*g
  delta=-.03*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
  raw=u['raw'].numpy();error=float(abs(delta-raw).max());assert error<4e-6;erradam=max(erradam,error)
  assert np.array_equal(nxt-a,raw) and u['names']==names
 assert updates==z['gradient_calls']
 return dict(status='pass',max_mean_gradient_error=errgrad,max_Adam_error=erradam,max_detached_reward_error=errreward,
  gradient_calls=updates,rounds=10,final_step=10,GT_read=False,independent_decoder_Jacobian=False)
