"""Matched frozen probability and learned-feature readout; same three native views."""
from core90 import *
def frozen_reads90(m,base,z,d,pm,im):
 ps=[];qs=[];us=[];ins=[];e0=None
 for a,b in [(None,None)]+list(zip(pm,im)):
  hh=m.interface.out.register_forward_pre_hook(lambda _m,args:ins.append(args[0].detach())) if m.use_interface else None
  try:
   p,q,e=m.main(base,z,d,'bidirectional',point_mask=a,image_mask=b)
   if e0 is None:e0=e
   ps.append(p);qs.append(q);us.append(e['projected_text'])
  finally:
   if hh is not None:hh.remove()
 q=F.normalize(torch.stack(qs).mean(0),dim=-1);u=F.normalize(torch.stack(us).mean(0),dim=-1);s=native_scores(q,u,z['token_mask'].bool()).double()
 if m.use_interface:assert len(ins)==3;s=s+m.interface.out(torch.stack(ins).mean(0)).squeeze(-1).double()
 return ps[0],torch.stack(ps).mean(0),s.softmax(0),e0
