"""Historical task-trained foreground projection, without fitted parameters."""
import torch
from methods.decota_final_simplified_v1.objectives import legal_logp,time_cells

def project(logits,actions,records):
 result={k:[]for k in ['Full','Fluct']};audit=[]
 for z,x,r in zip(logits,actions,records):
  lp,ij=legal_logp(z);x=x.detach().float().double().reshape(-1)
  assert len(x)==z.shape[1] and torch.isfinite(x).all()
  med=torch.quantile(x,.5);mad=torch.quantile((x-med).abs(),.5)
  u=(x-med)/(mad+1e-6);_,w=time_cells(r['frame_ids'],lp.device)
  assert abs(float(w.sum())-1)<1e-12
  bar=(w*u).sum();meta={}
  for name,f in [('Full',u),('Fluct',u-bar)]:
   pref=torch.cat([f.new_zeros(1),(w*f).cumsum(0)])
   score=pref[ij[1]+1]-pref[ij[0]]+.1*lp
   k=int(score.argmax());i,j=ij[:,k].tolist()
   result[name].append([r['frame_ids'][i],r['frame_ids'][j]+1])
   # Independent sliced sums; compare values and decision, not only loss.
   direct=torch.stack([(w[i:j+1]*f[i:j+1]).sum()+.1*lp[k]for k,(i,j)in enumerate(ij.T.tolist())])
   assert float((direct-score).abs().max())<1e-10 and int(direct.argmax())==k
   meta[name]=dict(target=k,interval=result[name][-1],max_direct_error=float((direct-score).abs().max()),candidates=len(lp))
  audit.append(dict(median=float(med),MAD=float(mad),weighted_mean=float(bar),**meta))
 return {k:dict(interval=[min(x[0]for x in v),max(x[1]for x in v)],offset_intervals=v)for k,v in result.items()},audit
