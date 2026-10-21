import ast,json,importlib
from pathlib import Path
import numpy as np
import torch
from stvg.predictor import output_writeback,Predictor
from stvg.initialization import initialize
from stvg.policy import feedback,chart,unchart,mixture_loss,mean_gradient,log_components
from stvg.positions import observation_positions
from stvg.temporal import final_mbr
ROOT=Path(__file__).resolve().parents[1]

def test_frozen_parameters():
 c=json.loads((ROOT/'configs/default.json').read_text())
 assert c['method']=='BoxFixed' and c['temporal']['default']=='strong'
 old=json.loads((ROOT/'configs/legacy_opd.json').read_text())
 for ds in ['hc2','vidstg']:
  p=Predictor(ds);assert p.source is None
  assert not {'lr','tau','steps','samples','scales','state_shape'}&set(p.config)
  for k in ['source_checkpoint','E_calibration','initialization_gamma']:assert p.config[k]==old['datasets'][ds][k]

def test_zero_initialization_and_writeback():
 src=np.tile([.5,.5,.2,.3],(16,1));pos=list(range(0,16,2));c=json.loads((ROOT/'configs/default.json').read_text())['datasets']['hc2']['E_calibration'];e,_=initialize(src,pos,src[pos],c);assert np.array_equal(e,src)
 raw=src.copy();raw[:,0]+=.1;raw[:,2]*=1.1;assert np.array_equal(output_writeback(src,raw,0),src);assert np.array_equal(output_writeback(src,raw,1),raw)

def test_policy_fixed_support_gradient():
 mu=torch.tensor([[.5,.3,np.log(.2),np.log(.3)]],dtype=torch.float64,requires_grad=True);D=torch.tensor([[.2,.3,1.,1.]],dtype=torch.float64);target=unchart(mu.detach());rr=feedback(mu.detach(),D,target,'regression',41,0,(.05,.25),.14);loss=mixture_loss(mu,rr['actions'],D,rr['weights'],(.05,.25));grad=torch.autograd.grad(loss,mu)[0];expected=mean_gradient(mu.detach(),rr['actions'],D,rr['weights'],(.05,.25));assert torch.allclose(grad,expected,atol=1e-10,rtol=1e-10)
 for j in range(4):
  a=mu.detach().clone();a[0,j]+=1e-6;v=mixture_loss(a,rr['actions'],D,rr['weights'],(.05,.25));a[0,j]-=2e-6;u=mixture_loss(a,rr['actions'],D,rr['weights'],(.05,.25));assert abs(float((v-u)/2e-6)-grad[0,j])<1e-6
 assert torch.equal(rr['eps'][:,:8],-rr['eps'][:,8:16])and torch.equal(rr['eps'][:,16:24],-rr['eps'][:,24:])
 density=torch.stack([torch.distributions.Independent(torch.distributions.Normal(mu.detach(),s*D),1).log_prob(rr['actions'])for s in [.05,.25]],-1);assert torch.allclose(density,log_components(mu.detach(),rr['actions'],D,(.05,.25)),atol=1e-11,rtol=0)

def test_physical_positions_and_mbr():
 ids=[0,2,3,5,8,11,14,18,23,29];pos=observation_positions(ids,[2,6]);assert len(pos)==len(set(pos))==8
 p=np.zeros((10,10));p[2,7]=1;assert final_mbr(ids,p)['interval']==[3,19]

def test_no_legacy_runtime_imports():
 for f in (ROOT/'stvg').glob('*.py'):
  tree=ast.parse(f.read_text())
  for n in ast.walk(tree):
   if isinstance(n,ast.ImportFrom):assert not (n.module or '').startswith(('methods.','vg_tta.','scripts.'))
  if f.stem not in ['__main__','teacher_worker']:importlib.import_module('stvg.'+f.stem)
