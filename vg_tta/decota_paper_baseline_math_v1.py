"""Independent NumPy Adam/SGD arithmetic from recorded live gradients.

This checks parameter execution, not an independent model Jacobian replay.
"""
import numpy as np
def audit_updates(result):
 maximum=0.;n=0
 for ev in result['update_evidence']:
  opt=ev['optimizer_before'];group=opt['param_groups'][0];ids=group['params'];states=opt['state']
  for before,grad,after,idx in zip(ev['parameters_before'],ev['gradients'],ev['parameters_after'],ids):
   x=before.numpy().astype(np.float64);y=after.numpy().astype(np.float64)
   if grad is None:expected=x
   else:
    g=grad.numpy().astype(np.float64);old=states.get(idx,{})
    if result['method']=='TENT':
     b1,b2=group['betas'];step=int(old['step'].item())+1 if old else 1
     m0=old['exp_avg'].numpy().astype(np.float64) if old else np.zeros_like(x)
     v0=old['exp_avg_sq'].numpy().astype(np.float64) if old else np.zeros_like(x)
     m=b1*m0+(1-b1)*g;v=b2*v0+(1-b2)*g*g
     expected=x-group['lr']*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+group['eps'])
    else:
     g=g+group['weight_decay']*x
     if group['momentum']:
      buf=group['momentum']*old['momentum_buffer'].numpy().astype(np.float64)+(1-group['dampening'])*g if 'momentum_buffer' in old else g
      g=g+group['momentum']*buf if group['nesterov'] else buf
     expected=x-group['lr']*g
   err=float(np.max(np.abs(expected-y)));maximum=max(maximum,err);n+=x.size
   assert np.allclose(y,expected,rtol=2e-6,atol=2e-6),('baseline optimizer arithmetic',result['method'],idx,err)
 assert len(result['update_evidence'])==result['counts']['optimizer_steps']
 return dict(status='pass',scalar_checks=n,max_parameter_error=maximum,live_updates=len(result['update_evidence']),independent_NumPy=True,decoder_Jacobian_independently_replayed=False,GT_read=False)
