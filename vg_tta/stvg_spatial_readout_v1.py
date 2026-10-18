"""One-factor DINO ranking consistency and whole-trajectory decoder-layer readout."""
import copy
import torch
from methods.decota_final_simplified_v1.observations import candidate_stages,pool_gate
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou

def context_probe(detection,position,frame_id,ranking):
 assert ranking in ['mixed','target'];d=dict(detection)
 if ranking=='target':d['all_phrase_scores']=detection['target_all_scores']
 stage=candidate_stages(d);idx=stage['old_top3']
 return pool_gate(dict(boxes=stage['boxes'][idx],target_scores=detection['target_all_scores'][idx],candidate_ids=idx,position=position,frame_id=frame_id)),stage

def select_layer(layers,anchors):
 assert layers.ndim==3 and layers.shape[0]==6 and layers.shape[-1]==4 and torch.isfinite(layers).all()
 if not anchors:return dict(layer=5,scores=None,boxes=layers[-1].detach().clone(),empty=True)
 pos=torch.tensor([a['position']for a in anchors],device=layers.device);ref=torch.tensor([a['box']for a in anchors],device=layers.device,dtype=layers.dtype)
 score=aligned_iou(layers[:,pos],ref[None]).mean(1);k=int(score.argmax())
 return dict(layer=k,scores=score.detach().cpu(),boxes=layers[k].detach().clone(),empty=False)

@torch.no_grad()
def full_layers(base,ids,records,state):
 """Capture actual second spatial Decoder pass in each original offset."""
 from scripts.stvg_pipeline_speed_v1 import full_prefix_readout
 from vg_tta.c1_enabling_tricks_v1 import TrickReplay
 from methods.decota_final_simplified_v1.tensors import detached
 original=base.state();captured=[]
 def hook(module,args,out):captured.append(out.detach().clone())
 h=base.decoder.decoder.register_forward_hook(hook)
 try:full=full_prefix_readout(base,ids,records,state)
 finally:h.remove()
 assert len(captured)==4
 second=[captured[1].flatten(1,2),captured[3].flatten(1,2)];layers=torch.stack([second[i%2][:,i//2]for i in range(len(ids))],1).float()
 assert layers.shape==(6,len(ids),4)and torch.equal(layers[-1].cpu(),full['boxes'])
 try:
  base.restore(state);cached=TrickReplay(base,ids,'layer_readout',{}).values()['layers'];assert torch.equal(layers,cached)
 finally:base.restore(original)
 return layers.detach().cpu(),full
