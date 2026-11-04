"""Expert-guided episodic spatial-decoder adaptation, no shared correction or DP."""
import gc,hashlib,time,json
import numpy as np,torch
from model_utils import spatial_modules,spatial_parameters,state_snapshot,restore

def to_device(value,device):
 if torch.is_tensor(value):return value.to(device)
 if isinstance(value,dict):return {k:to_device(v,device)for k,v in value.items()}
 if isinstance(value,(list,tuple)):return type(value)(to_device(v,device)for v in value)
 return value

def cxcywh_to_xyxy(b):return torch.cat([b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2],dim=1)

def expert_loss(boxes,target,positions,lambda_l1,lambda_giou):
 from torchvision.ops import generalized_box_iou
 p=boxes[positions];assert p.shape==target.shape==(8,4)
 l1=(p-target).abs().sum(1).mean();giou=(1-torch.diagonal(generalized_box_iou(cxcywh_to_xyxy(p),cxcywh_to_xyxy(target)))).mean()
 return lambda_l1*l1+lambda_giou*giou,dict(L1=float(l1.detach()),one_minus_GIoU=float(giou.detach()))

def decoder_boxes(model,ctx):
 # Cached encoder outputs are constant; pos_fc + entire PosDecoder execute afresh.
 out=[]
 for kw,cls in zip(ctx['spatial_inputs'],ctx['frames_cls']):
  args={k:v for k,v in kw.items()};args['pred_boxes']=model.ground_decoder.pos_fc(cls).sigmoid().unsqueeze(1)
  # Exact historical native zero-residual addition/layout, not a learnable residual.
  q=args['query_tgt'];args['query_tgt']=q+torch.zeros(256,device=q.device,dtype=q.dtype)[None,None,:]
  pos=model.ground_decoder.decoder(**args);out.append(pos.flatten(1,2)[-1])
 n=len(ctx['frame_ids']);return torch.stack([out[t%2][t//2]for t in range(n)])

def state_digest(state):
 h=hashlib.sha256()
 for mod,values in state.items():
  for k,v in values.items():h.update((mod+'/'+k).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def select_trainable(model):
 model.eval().requires_grad_(False);named=spatial_parameters(model)
 for name,p in named:p.requires_grad_(True)
 ids={id(p)for n,p in named};assert all(not p.requires_grad for p in model.parameters()if id(p)not in ids)
 return named

def adapt(model,cpu_context,expert_boxes,positions,config,*,checkpoints=(5,20),scratch=None):
 # One fresh AdamW state per query/config stream. No temporal/GT loss or parameter update.
 torch.cuda.synchronize();begin=time.perf_counter();initial=state_snapshot(model);initial_sha=state_digest(initial);ctx=to_device(cpu_context,'cuda');teacher=torch.as_tensor(expert_boxes,device='cuda',dtype=torch.float32);pos=torch.as_tensor(positions,device='cuda',dtype=torch.long)
 for mod in spatial_modules(model).values():mod.cuda()
 with torch.no_grad():before=decoder_boxes(model,ctx)
 reference=torch.as_tensor(cpu_context['Source'],device='cuda',dtype=before.dtype);assert torch.equal(before,reference),'Initial spatial-decoder replay must be bitwise native Source before fitting'
 named=select_trainable(model);ids={id(p)for n,p in named};frozen_versions={id(p):p._version for p in model.parameters()if id(p)not in ids};opt=torch.optim.AdamW([p for n,p in named],lr=config['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False);trace=[];saved={}
 try:
  for step in range(1,max(checkpoints)+1):
   with torch.enable_grad():
    opt.zero_grad(set_to_none=True);pred=decoder_boxes(model,ctx);loss,parts=expert_loss(pred,teacher,pos,config['lambda_l1'],config['lambda_giou']);assert loss.requires_grad and torch.isfinite(loss)
    loss.backward();active=[(n,p)for n,p in named if p.grad is not None];assert active and all(torch.isfinite(p.grad).all()for n,p in active);norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],config['clip_norm']);assert torch.isfinite(norm)and float(norm)>0;opt.step()
   assert all(p.grad is None and p._version==frozen_versions[id(p)]for p in model.parameters()if id(p)not in ids)
   trace.append(dict(step=step,loss=float(loss.detach()),**parts,gradient_norm_before_clip=float(norm),gradient_parameter_tensors=len(active),gradient_parameter_scalars=sum(p.numel()for n,p in active)))
   if step in checkpoints:
    with torch.no_grad():after=decoder_boxes(model,ctx);post,pp=expert_loss(after,teacher,pos,config['lambda_l1'],config['lambda_giou'])
    assert torch.isfinite(after).all();current=state_snapshot(model);delta=sum(float((current[m][k]-initial[m][k]).double().square().sum())for m in current for k in current[m])**.5;changed=(after!=reference).any(1);sha=state_digest(current)
    # Optional successful-state checkpoint/cold readback, outside any deployment timer.
    cold=True
    if scratch is not None:
     path=scratch/('step'+str(step)+'.pt');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();torch.save(current,path);reread=torch.load(path,map_location='cpu',weights_only=False,mmap=True);assert state_digest(reread)==sha
     for mod,obj in spatial_modules(model).items():obj.load_state_dict(reread[mod],strict=True)
     with torch.no_grad():again=decoder_boxes(model,ctx)
     assert torch.equal(after,again);cold=True
     # Caller owns disposal ONLY after its receipt/prediction commits, never here.
    observed=torch.zeros(len(after),dtype=torch.bool,device='cuda');observed[pos]=True
    update_norms={m+'/'+k:float((current[m][k]-initial[m][k]).double().norm())for m in current for k in current[m]}
    saved[step]=dict(boxes=after.detach().cpu().numpy(),parameter_update_norms=update_norms,unobserved_changed_frames=int((changed&~observed).sum()),max_box_coordinate_change=float((after-reference).abs().max()),trace=list(trace),parameter_delta_L2=delta,adapted_parameter_sha256=sha,parameter_tensors=len(named),parameter_scalars=sum(p.numel()for n,p in named),spatial_changed_frames=int(changed.sum()),all_sampled_frames=len(after),pseudo_loss_before=float(expert_loss(before,teacher,pos,config['lambda_l1'],config['lambda_giou'])[0].detach()),pseudo_loss_after=float(post),post_loss_parts=pp,gradient_names=[n for n,p in active],checkpoint_cold_forward_exact=cold,initial_parameter_sha256=initial_sha,optimizer_steps=step)
  torch.cuda.synchronize();seconds=time.perf_counter()-begin
 finally:
  opt.zero_grad(set_to_none=True);restore(model,initial)
  with torch.no_grad():reset=decoder_boxes(model,ctx)
  assert torch.equal(reset,reference),'Episodic reset must recover exact original all-frame Source boxes'
 for z in saved.values():z.update(reset_all_parameters_and_Source_boxes_exact=True,shared_stream_total_seconds=seconds,scope='Conditional cached encoder context + transfer + exact spatial replay/updates/postprediction; includes experimental trace/checkpoint if scratch enabled, not end-to-end deployment.')
 return saved
