"""Fixed actor+argument motion-key contexts; no label/score acceptance."""
import torch
from research.native_event_support_view_v1.core import keep_region

def quota_mask(target,proposal,quota,valid):
 available=valid&~target;assert 0<=quota<=int(available.sum())
 score=proposal.flatten().double();index=torch.arange(score.numel(),dtype=torch.float64)
 order=sorted(torch.nonzero(available.flatten()).flatten().tolist(),key=lambda j:(-float(score[j]),j))
 extra=torch.zeros_like(target).flatten();extra[order[:quota]]=True
 return target|extra.reshape_as(target)

def contexts(actor,argument,subject,offset,valid):
 keys={k:[]for k in ['Pair','TargetOnly','ReflectedArgumentExtra','WrongTimeArgumentExtra','SubjectSourceContext']};stats=[]
 for j,v in enumerate(valid):
  ix=offset+2*j;t,tf=keep_region(actor[ix],v);a,af=keep_region(argument[ix],v);s,sf=keep_region(subject[ix],v);w,wf=keep_region(argument[len(argument)-1-ix],v)
  from research.native_event_support_view_v1.core import reflected
  c=reflected(a,v);quota=int((a&~t).sum());pair=t|a
  keys['Pair'].append(pair);keys['TargetOnly'].append(t)
  for k,m in [('ReflectedArgumentExtra',c),('WrongTimeArgumentExtra',w),('SubjectSourceContext',s)]:keys[k].append(quota_mask(t,m,quota,v))
  assert all(int((keys[k][-1]&~t).sum())==quota for k in keys if k!='TargetOnly')
  stats.append(dict(frame_index=ix,extra=quota,target=int(t.sum()),valid=int(v.sum()),fallbacks=[tf,af,sf,wf],argument_reflected_overlap=float((a&c).sum()/a.sum())))
 return {k:torch.stack(v).flatten(1)for k,v in keys.items()},stats

def keymask(original,keep):
 count=keep.shape[1];assert len(original)==len(keep)and bool(keep.any(1).all())
 if bool(keep.all()):return original
 out=original.clone();out[:,-count:]|=~keep.to(out.device)
 assert torch.equal(out[:,:-count],original[:,:-count]);return out
