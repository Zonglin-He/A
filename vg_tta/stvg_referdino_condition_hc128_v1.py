"""Frozen initial-reference intervention with pruning and matching lineage.

One slot per frame is the native winning track's initial proposal ancestor.
Slots are locked from a no-intervention pass, never chosen using ground truth.
All other query contents/references and native tracking/temporal/mask work remain.
"""
import hashlib
import numpy as np
import torch
from vg_tta.stvg_referdino_native32_v1 import NativeRefer

class ConditionalRefer(NativeRefer):
 def run(self,frames,caption,boxes=None,slots=None):
  trace={};matches=[];handles=[];original=self.model.match_from_embds
  def match(*args,**kw):
   out=original(*args,**kw);matches.append(np.asarray(out).copy());return out
  def before(module,args,kw):
   ref=kw['refpoints_unsigmoid'];tgt=kw['tgt'];trace['initial']=ref.detach().sigmoid().cpu();trace['native_tgt_hash']=hashlib.sha256(tgt.detach().cpu().numpy().tobytes()).hexdigest();trace['initial_count']=ref.shape[0]
   if boxes is not None:
    b=torch.as_tensor(boxes,device=ref.device,dtype=ref.dtype);ids=torch.as_tensor(slots,device=ref.device,dtype=torch.long)
    assert b.shape==(ref.shape[1],4) and ids.shape==(ref.shape[1],) and torch.isfinite(b).all() and (b[:,2:]>0).all()
    # Native initial references have sigmoid range. Explicitly report projection.
    bounded=b.clamp(1e-5,1-1e-5);trace['reference_projection_elements']=int((bounded!=b).sum());trace['injected_reference']=bounded.cpu();new=ref.clone();new[ids,torch.arange(len(ids),device=ref.device)]=torch.logit(bounded);kw=dict(kw,refpoints_unsigmoid=new)
    keep=torch.ones(ref.shape[:2],device=ref.device,dtype=torch.bool);keep[ids,torch.arange(len(ids),device=ref.device)]=False;assert torch.equal(new[keep],ref[keep]);trace['other_references_bitexact']=True
   return args,kw
  def after(module,args,output):
   hs,refs,selected=output;ids=torch.arange(trace['initial_count'],device=selected[0].device)[:,None].expand(-1,selected[0].shape[1]);stages=[]
   for chosen in selected:
    ids=ids.gather(0,chosen);stages.append(ids.detach().cpu().clone())
   trace['pruning_lineages']=stages;trace['final_lineages']=ids.transpose(0,1).detach().cpu();trace['spatial_final_content_hash']=hashlib.sha256(hs[-1].detach().cpu().numpy().tobytes()).hexdigest();trace['spatial_references']=refs[-2].detach().cpu()
  self.model.match_from_embds=match
  handles.append(self.model.transformer.decoder.register_forward_pre_hook(before,with_kwargs=True));handles.append(self.model.transformer.decoder.register_forward_hook(after))
  def final(module,args,out):
   trace['all_boxes']=out['pred_boxes'][0].detach().cpu();trace['all_logits']=out['pred_logits'][0].detach().cpu()
  handles.append(self.model.register_forward_hook(final)) # infer bypasses Module.__call__, capture below instead
  # Native wrapper calls infer directly, so wrap infer for final-output audit.
  infer=self.model.infer
  def infer_capture(*args,**kw):
   out=infer(*args,**kw);trace['all_boxes']=out['pred_boxes'][0].detach().cpu();trace['all_logits']=out['pred_logits'][0].detach().cpu();trace['full_masks_hash']=hashlib.sha256(out['pred_masks'].detach().cpu().numpy().tobytes()).hexdigest();return out
  self.model.infer=infer_capture
  try:
   out=super().__call__(frames,caption)
  finally:
   for h in handles:h.remove()
   self.model.match_from_embds=original;self.model.infer=infer
  raw=trace['final_lineages'];assert len(matches)==len(raw)-1
  tracked=torch.stack([raw[0]]+[raw[t+1,torch.as_tensor(m[0],dtype=torch.long)]for t,m in enumerate(matches)])
  trace['matching_permutations']=[m.tolist()for m in matches];trace['tracked_lineages']=tracked;idx=out['selected_query'];trace['selected_lineages']=tracked[:,idx];trace['selected_slots_native']=trace['selected_lineages'].tolist()
  if slots is not None:
   wanted=torch.as_tensor(slots,dtype=torch.long);trace['injected_slots']=wanted;trace['survives_per_layer']=[(x==wanted[None]).any(0).tolist()for x in trace['pruning_lineages']];trace['selected_is_intervened']=(trace['selected_lineages']==wanted).tolist();trace['intervened_track_indices']=[torch.where(tracked[t]==wanted[t])[0].tolist()for t in range(len(raw))]
  assert torch.equal(out['boxes'],trace['all_boxes'][:,idx]);out['trace']=trace;return out


def continue_fit(rp,target,cfg,start_state):
 """Equal-budget direct feedback uptake, not a full OPD experiment."""
 from vg_tta.stvg_hc_anchor_content16_v1 import local_cost
 with torch.no_grad():rp.state.copy_(start_state.to(rp.state))
 opt=torch.optim.Adam([rp.state],lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
 target=torch.as_tensor(target,device='cuda',dtype=torch.float64);path=[];updates=[]
 for k in range(cfg['steps']+1):
  v=rp.values();loss=local_cost(v['boxes'][rp.positions],target).mean();path.append(dict(state=rp.state.detach().cpu().clone(),boxes=v['boxes'].detach().cpu(),loss=float(loss.detach())))
  if k==cfg['steps']:break
  opt.zero_grad(set_to_none=True);grad=torch.autograd.grad(loss,rp.state)[0];assert torch.isfinite(grad).all();rp.state.grad=grad.detach();old=rp.state.detach().clone();opt.step();updates.append(dict(gradient=grad.detach().cpu(),step=(rp.state.detach()-old).cpu()));rp.backwards+=1
 assert all(not p.requires_grad and p.grad is None for p in rp.decoder.parameters())
 return dict(path=path,updates=updates,final=path[-1]['boxes'],final_state=path[-1]['state'],actual_updates=len(updates))
