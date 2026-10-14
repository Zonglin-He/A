"""Finite CPU scalar diagnostics of saved original-step rewards and actual displacements."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_instance_identity_common_v1 import BASE,OLD,read,write,sha,guard

def main():
 import torch
 sys.addaudithook(guard);rows=[]
 for r in read(BASE/'STEP_DIAGNOSTIC_RUNTIME.json')['records']:
  assert sha(ROOT/r['path'])==r['sha256'];d=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);cid=r['case_id'];alias=('HC'if cid.startswith('hc2')else'V')+f'{int(cid[-3:])+1:02d}'
  for name,f in d['fits'].items():
   for i,e in enumerate(f['path']):
    before=e['state'];after=e.get('update_state',f['path'][i+1]['state']if i+1<len(f['path'])else f['final_state']);delta=torch.cat([(after[k]-before[k]).flatten().double()for k in before])
    u=e.get('update');grad=u['gradient'].double()if u is not None else None
    if u is not None:assert torch.allclose(delta,u['raw'].double(),atol=1e-7,rtol=1e-5)
    p=e.get('rollout');rew=None if p is None else p['rewards'].double();weights=None if p is None else p['weights'].double()
    rows.append(dict(case=alias,arm=name,step=i,parameter_displacement_l2=float(delta.norm()),parameter_displacement_linf=float(delta.abs().max()),gradient_l2=None if grad is None else float(grad.norm()),gradient_dot_actual_displacement=None if grad is None else float(grad@delta),gradient_record_status='saved_gradient'if grad is not None else'not_saved_for_direct'if name=='direct'else'no_information_no_update',reward_range_by_anchor=None if rew is None else(rew.max(-1).values-rew.min(-1).values).tolist(),weight_ess_by_anchor=None if weights is None else(1/weights.square().sum(-1)).tolist(),no_information=None if p is None else not p['informative']))
 assert not torch.cuda.is_initialized()
 write(BASE/'SAVED_GRADIENT_DIAGNOSTIC.json',dict(status='pass',GT_read=False,GPU_used=False,rows=rows,count=len(rows),scope='saved gradients and actual displacement; no new autograd; Direct gradients not stored and not invented'))
 print('pass',len(rows))
if __name__=='__main__':main()
