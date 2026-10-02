"""Meaningful model-free checks: pairing/sign, zero/ties, actuation and norm."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from vg_tta.tastvg_directional_preference_v1 import apply_direction,direction_from_ranks,nested_directions
from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
def run():
 basis=torch.eye(4,dtype=torch.float64)
 ranks=average_ranks(np.array([.4,.9,.1,.2,.8,.5,.6,.7,.3]))
 v,c,top=direction_from_ranks(basis,ranks,'rank_directional')
 assert c[0]>0 and c[1]<0 and c[2]<0 and c[3]>0 and np.array_equal(v.numpy(),c)
 f,_,top=direction_from_ranks(basis,ranks,'top_directional')
 assert top==0 and torch.equal(f,torch.tensor([1.,0,0,0],dtype=torch.float64))
 flat=average_ranks(np.ones(9));zero,_,_=direction_from_ranks(basis,flat,'rank_directional');assert not zero.any()
 for mode in ['rkl','rank_directional','top_directional']:
  p=torch.nn.Parameter(torch.tensor([1.,2.,3.,4.],dtype=torch.float32));g=torch.tensor([.3,-.4,.2,.1])
  before=p.detach().clone();m=apply_direction([('p',p)],[g],basis,ranks,.02,mode)
  assert abs(m['actual_step_norm']-.02*float(g.double().norm()))<2e-7
  if mode=='rkl':assert torch.equal(p,before.add(g,alpha=-.02))
  else:assert torch.dot((p-before).double(),v if mode=='rank_directional' else f)>0
  p=torch.nn.Parameter(before.clone());z=apply_direction([('p',p)],[g],basis,flat,.02,mode)
  if mode!='rkl':assert torch.equal(p,before) and z['no_op_reason']=='no_rank_direction'
  p=torch.nn.Parameter(before.clone());z=apply_direction([('p',p)],[torch.zeros_like(g)],basis,ranks,.02,mode);assert torch.equal(p,before)
 u=nested_directions(1792,4);torch.testing.assert_close(u@u.T,torch.eye(4,dtype=torch.float64),atol=1e-12,rtol=0)
 assert not torch.cuda.is_initialized()
 print('PASS: rank pairing/sign, stable top tie, exact zero direction, 3 actuation/norm checks, zero gradient, original orthonormal basis')
if __name__=='__main__':run()
