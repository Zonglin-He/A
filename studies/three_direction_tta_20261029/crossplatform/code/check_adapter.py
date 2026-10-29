"""Synthetic engineering check only; never a task score."""
import json,torch,sys
from pathlib import Path
from tta import Adapter,boxes
class Fake(torch.nn.Module):
 def __init__(self):
  super().__init__();self.cross_encoder=torch.nn.Sequential(torch.nn.LayerNorm(4));self.register_buffer('fixed',torch.tensor([[1.,2,3,4],[-1.,2,0,3]]));self.register_buffer('tokens',torch.tensor([[[1.,0,0,0],[0.,1,0,0],[0.,0,1,0]]]))
 def _run_backbones(self,inputs):return {'x':self.fixed+inputs['point_clouds'].mean()*0.01}
 def forward(self,inputs):
  c=self._run_backbones(inputs);q=torch.nn.functional.normalize(self.cross_encoder(c['x']),dim=-1)[None]
  return {'last_center':torch.stack([q[0,0,:3],q[0,1,:3]])[None],'last_pred_size':torch.ones(1,2,3),'last_proj_queries':q,'proj_tokens':self.tokens}
m=Fake();state={k:v.clone() for k,v in m.state_dict().items()};inputs={'point_clouds':torch.randn(1,50,3),'text':['car']};mask=torch.tensor([False,True,False]);adapter=Adapter(m);rng=torch.get_rng_state().clone()
report={}
for mode in ['entropy','consistency','gated']:
 outputs,teacher,diag=adapter.run(inputs,mask,3,mode=mode);assert all(torch.equal(v,m.state_dict()[k]) for k,v in state.items());assert torch.equal(rng,torch.get_rng_state());assert all(p.grad is None for p in m.parameters())
 report[mode]={'trace':diag['trace'],'reset_exact':True,'rng_restored':True,'grad_cleared':True}
Path(__file__).with_name('adapter_checked.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
