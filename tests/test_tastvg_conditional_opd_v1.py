import torch
from vg_tta.tastvg_conditional_opd_v1 import create,scores,objective,update

def item(position=1):
 g=torch.Generator().manual_seed(3)
 return dict(position=position,phi=torch.randn(8,768,dtype=torch.float64,generator=g),base=torch.arange(8,dtype=torch.float64)/10,teacher=torch.arange(7,-1,-1,dtype=torch.float64)/10)

def test_zero_initialization_and_loss_identity():
 m=create();x=item();assert torch.equal(scores(m,x),x['base'])
 z={**x,'teacher':x['base'].clone()};assert abs(float(objective(m,z,'reverse_kl').detach()))<1e-14

def test_reverse_kl_shift_invariance():
 m=create();x=item();a=objective(m,x,'reverse_kl');b=objective(m,{**x,'base':x['base']+5,'teacher':x['teacher']-3},'reverse_kl')
 torch.testing.assert_close(a,b,atol=1e-13,rtol=0)

def test_replay_is_detached_and_current_not_duplicated():
 for mode in ['pairwise','reverse_kl']:
  m=create();x=item();old=item(0);h=x['phi'].clone();d,g=update(m,x,[old],mode)
  assert d['replay_positions']==[0] and d['after']['total']<d['before']['total']
  assert torch.equal(h,x['phi']) and x['phi'].grad is None
  assert g['2.weight'].norm()>0 and g['0.weight'].norm()==0
