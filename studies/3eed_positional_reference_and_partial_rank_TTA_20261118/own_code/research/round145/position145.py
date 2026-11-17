"""Actual model-state reference, unchanged physical coordinates/graphs/outputboxes."""
from common145 import *
import torch,numpy as np

def normalize145(base,z,delta):
 zz=dict(z)
 if not delta:return zz
 x=z['xyz'].clone();x[:,:,2]+=float(delta)
 zz['pos0']=base.pos_embed(x).transpose(1,2).contiguous()
 for i in range(6):
  k='position'+str(i);zz[k]=z[k].clone();zz[k][:,:,2]+=float(delta)
 assert torch.equal(zz['xyz'],z['xyz']) and torch.equal(zz['vis0'],z['vis0'])
 return zz

def savepos145(out,z,z0,delta):
 out['delta']=np.array(delta);out['physical_xyz']=z0['xyz'];x=z0['xyz'].copy();x[:,:,2]+=delta;out['pos_input']=x;out['pos0']=z['pos0'].detach().cpu().numpy();out['positions']=np.stack([z['position'+str(i)].detach().cpu().numpy() for i in range(6)])
