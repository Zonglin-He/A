"""CPU engineering checks only; these are not GREAT task results."""
import torch,random,numpy as np,sys,json,pathlib
from tta import adapt
class Fake(torch.nn.Module):
 def __init__(self):
  super().__init__();self.decoder=torch.nn.Module();self.decoder.cmff=torch.nn.Module();self.decoder.cmff.cross_atten1=torch.nn.Module();self.decoder.cmff.cross_atten1.layernorm=torch.nn.LayerNorm(3)
 def forward(self,img,point,human,obj):
  x=self.decoder.cmff.cross_atten1.layernorm(point.transpose(1,2));position=torch.linspace(-.4,.4,point.shape[-1]).to(point.device)
  return torch.sigmoid(x[:,:,0]+position).unsqueeze(-1)
def check():
 m=Fake().eval();x=torch.randn(1,3,16);before={k:v.clone() for k,v in m.state_dict().items()};random.seed(13);np.random.seed(14);torch.manual_seed(15)
 pr=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state().clone();req={n:p.requires_grad for n,p in m.named_parameters()}
 r=adapt(m,torch.zeros(1),x,['human'],['object'],steps=3,lr=.01)
 assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items());assert pr==random.getstate();assert np.array_equal(nr[1],np.random.get_state()[1]);assert torch.equal(tr,torch.get_rng_state());assert req=={n:p.requires_grad for n,p in m.named_parameters()}
 assert r['diagnostics']['same_input_repeat_max_abs']==0;assert r['diagnostics']['parameter_delta_l2']>0;assert all(x['gradient_norm']>0 for x in r['diagnostics']['loss'])
 print(json.dumps({'kind':'engineering_only','all_state_restore':True,'nonzero_updates':True,'same_input_repeat':True,'diagnostics':r['diagnostics']},indent=2))
if __name__=='__main__':check()
