"""Independent CPU Conv/BN/soft-token evaluator, no source framework imports."""
import torch,numpy as np
import torch.nn.functional as F
def ce_cpu108(h,mask,S):
 x=torch.from_numpy(h).T[None]
 for j in [0,4]:
  x=F.conv1d(x,torch.from_numpy(S[f'net.{j}.weight']));b=j+1;x=F.batch_norm(x,torch.from_numpy(S[f'net.{b}.running_mean']),torch.from_numpy(S[f'net.{b}.running_var']),torch.from_numpy(S[f'net.{b}.weight']),torch.from_numpy(S[f'net.{b}.bias']),training=False,momentum=0.,eps=1e-5).relu()
 x=F.conv1d(x,torch.from_numpy(S['net.8.weight']),torch.from_numpy(S['net.8.bias']))[0].T;positive=torch.zeros(256,dtype=torch.bool);positive[:len(mask)]=torch.from_numpy(mask).bool();mass=x.double().log_softmax(-1).masked_fill(~positive[None],-torch.inf).logsumexp(-1);return mass.softmax(0).numpy(),x.numpy()
