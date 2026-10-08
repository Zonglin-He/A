"""Independent float64 Gaussian/control-variate/IoU/Adam/state arithmetic."""
import numpy as np
from vg_tta.decota_fixed_full_audit_v1 import top1_support,overlap,vector

def giou(a,b):
 alo=a[:,:2]-a[:,2:]/2;ahi=a[:,:2]+a[:,2:]/2
 blo=b[:,:2]-b[:,2:]/2;bhi=b[:,:2]+b[:,2:]/2
 inter=np.maximum(np.minimum(ahi,bhi)-np.maximum(alo,blo),0).prod(1)
 union=np.maximum(a[:,2:],0).prod(1)+np.maximum(b[:,2:],0).prod(1)-inter;union=np.maximum(union,1e-7)
 enclosing=np.maximum(np.maximum(ahi,bhi)-np.minimum(alo,blo),0).prod(1);enclosing=np.maximum(enclosing,1e-7)
 return inter/union-(enclosing-union)/enclosing

def audit(z,expert):
 cfg=z['config'];steps=cfg['steps'];allnames=list(z['initial']);names=z.get('optimizer_parameter_names',allnames);hist=z['path'];support=top1_support(expert)
 assert z['selected_step']==steps and len(hist)==steps+1 and len(z['rounds'])==steps
 assert z['positions']==[p for p,_ in support] and z['active_parameters']==len(vector(z['initial'],names)) and not z['GT_used']
 assert np.array_equal(vector(z['state'],names),vector(hist[steps]['state'],names))
 assert np.array_equal(z['final'].numpy(),hist[steps]['boxes'].numpy())
 m=np.zeros(z['active_parameters']);v=np.zeros(z['active_parameters']);updates=0;errgrad=erradam=errreward=0.
 for k,(h,r) in enumerate(zip(hist,z['rounds'])):
  a=vector(h['state'],names);nxt=vector(hist[k+1]['state'],names)
  if not support:
   assert not r['updated'] and np.array_equal(a,nxt);continue
  for name in allnames:
   if name not in names:assert np.array_equal(h['state'][name].numpy(),hist[k+1]['state'][name].numpy())
  if z.get('direct_objective'):
   pred=h['boxes'].numpy().astype(float);target=np.array([e for _,e in support]);selected=pred[[p for p,_ in support]]
   error=abs(float((5*np.abs(selected-target).sum(1)+2*(1-giou(selected,target))).mean())-r['loss']);assert error<8e-6
   original=used=w=samples=mu=current=None
  else:
   rr=r['rollout'];samples=rr['samples'].numpy().astype(float);mu=rr['mean'].numpy().astype(float)
   w=rr['weights'].numpy().astype(float);used=rr['used_rewards'].numpy().astype(float)
   assert samples.shape==(len(support),32,4)
   original=rr['rewards'].numpy().astype(float);actions=1/(1+np.exp(-samples))
   recomputed=np.asarray([[overlap(x,e) for x in aa] for aa,(_,e) in zip(actions,support)])
   error=float(abs(original-recomputed).max());assert error<3e-6;errreward=max(errreward,error)
   if rr['permutations'] is not None:assert np.array_equal(used,np.take_along_axis(original,rr['permutations'].numpy(),1))
   else:assert np.array_equal(used,original)
   logits=used/cfg['tau'];ew=np.exp(logits-logits.max(1,keepdims=True));ew/=ew.sum(1,keepdims=True)
   assert abs(ew-w).max()<3e-7
   current=r['mean_before'].numpy().astype(float)
   expected=((current-mu)/cfg['sigma']**2-((w-1/32)[:,:,None]*(samples-current[:,None,:])).sum(1)/cfg['sigma']**2)/len(support)
   error=float(abs(expected-r['autograd_mean_gradient'].numpy()).max());assert error<3e-5;errgrad=max(errgrad,error)
   assert r['updated']==rr['informative'] and r['no_information']==(not rr['informative'])
  if not r['updated']:assert np.array_equal(a,nxt);continue
  u=h['update'];g=u['gradient'].numpy().astype(float);updates+=1
  m=.9*m+.1*g;v=.999*v+.001*g*g
  delta=-cfg['lr']*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
  raw=u['raw'].numpy();error=float(abs(delta-raw).max());assert error<4e-6;erradam=max(erradam,error)
  assert np.array_equal(nxt-a,raw) and u['names']==names
 assert updates==z['gradient_calls']
 return dict(status='pass',max_mean_gradient_error=errgrad,max_Adam_error=erradam,max_detached_reward_error=errreward,
  gradient_calls=updates,rounds=steps,final_step=steps,GT_read=False,independent_decoder_Jacobian=False)
