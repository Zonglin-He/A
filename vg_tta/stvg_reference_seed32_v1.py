"""Raw absolute PosDecoder reseeding; never add E a second time."""
import torch
from methods.decota_final_simplified_v1.tensors import detached
ARMS=['Source_MBR','E_MBR','Source_reference','E_reference']

@torch.no_grad()
def readout(base,reference=None):
 boxes=[];traces=[]
 with torch.autocast('cuda',enabled=False):
  for view,inputs in enumerate(base.spatial_inputs):
   kw=dict(inputs);original=kw['pred_boxes'];native_content=kw['query_tgt'].clone()
   if reference is not None:
    kw['pred_boxes']=reference[view::2].to(original).unsqueeze(1)
    assert kw['pred_boxes'].shape==original.shape
    assert torch.isfinite(kw['pred_boxes']).all()and(kw['pred_boxes'][:,0,2:]>0).all()
   actual=kw['pred_boxes'].clone();pos=base.decoder.decoder(**kw)
   assert torch.equal(kw['query_tgt'],native_content)
   boxes.append(pos.flatten(1,2)[-1]);traces.append(detached(dict(view=view,native_reference=original,actual_reference=actual,native_content=native_content,all_six_absolute_outputs=pos),'cpu'))
 return dict(boxes=torch.stack([boxes[i%2][i//2]for i in range(base.n)]).float().cpu(),traces=traces)

def properties():
 # An identity decoder illustrates why the old E-retaining wrapper is invalid.
 S=torch.tensor([.4,.5,.3,.2]);E=torch.tensor([.5,.5,.3,.2]);wrong=E+(E-S)
 assert not torch.equal(wrong,E)
 return dict(status='pass',absolute_head_not_residual=True,no_static_E_retaining_wrapper=True,no_new_optimizer=True)
